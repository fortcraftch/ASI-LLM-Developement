# Desarrollo en paralelo al entrenamiento

Estas funciones son opcionales y se activan con `python -m asi adaptive`. No modifican el entrenador, las recetas de 124M ni el checkpoint en curso. Los nuevos módulos reutilizan el formato del modelo local. Se han probado con modelos pequeños; su utilidad y degradación en la LLM entrenada siguen pendientes de medir.

## Qué se puede hacer ahora

| Área | Implementación | Qué depende del entrenamiento |
|---|---|---|
| Decisión por hardware/calidad (2.4) | Estimación de memoria, consulta de VRAM libre y elección entre residencia densa o con caché INT8; filtro opcional por NLL observada | Calibrar las políticas sobre el checkpoint definitivo; el planificador no inventa una calidad |
| Varias etiquetas (2.4) | Unión fija de los pools seleccionados; presupuesto conjunto y rechazo si no caben todos | Medir degradación en preguntas realmente multitema |
| Clasificador integrado (1.1/2.4) | Cabeza pequeña entrenable, embeddings de la LLM congelados y compartidos, BCE multietiqueta | Entrenar la cabeza final usando los embeddings definitivos y validarla |
| Expertos comprimidos (3.3) | RAM original, expertos INT8 residentes en GPU y expertos flotantes activos; promoción al cambiar de contexto | Medir pérdidas, latencia de promoción y memoria a escala real |

Esto es una primera política dinámica para **un checkpoint**. No elige entre todas las arquitecturas ni encuentra la configuración óptima de calidad/latencia. El margen de memoria cubre temporales de forma aproximada: una estimación no garantiza ausencia de OOM. No se han demostrado mejoras de velocidad.

## Varias etiquetas: qué cambia

Con la receta de dos expertos por pool, seleccionar `health,biology` activa **cuatro expertos enrutados por capa**. Todos participan con pesos uniformes que suman la escala de routing; E no se ejecuta. Los expertos y sus pesos de mezcla permanecen fijos durante la generación. Los expertos compartidos siguen activos.

La LLM categorizada se entrena inicialmente con una pareja por entrada. Mezclar dos parejas es una variante de inferencia fuera de esa condición de entrenamiento: puede ayudar o empeorar. No se promete conservar calidad. En arquitecturas con más de dos expertos por pool, esta política ejecuta todos los expertos de la unión, y por tanto también cambia el cómputo frente a top-2.

El programa no descarta etiquetas silenciosamente para cumplir el presupuesto. Si no caben, devuelve `status: blocked`. Este runtime no divide por sí solo la pregunta ni intercambia expertos por token. El comando separado `asi tasks` planifica subtareas explícitas y sus dependencias; véase [CLASSIFIER_DATA_TASKS.md](CLASSIFIER_DATA_TASKS.md). La función `ContextMixture` instala el cambio solo dentro de su contexto y restaura el routing original al salir.

## Planificar y ejecutar

En PowerShell, desde la raíz del proyecto y **cuando exista un checkpoint compatible**:

```powershell
$model = 'results/domain_124m/model_37595.pt'
$pools = 'configs/experiment_124m_v1/expert_pools.json'
python -m asi adaptive run --checkpoint $model --pool-manifest $pools --pools health,biology --prompt 'Explain how bacteria affect health.' --device cuda --warm-mib 64 --reserve-mib 1024 --output results/plan_health_biology.json
```

Sin `--execute` solo planifica: carga el checkpoint en RAM, inventaría tensores y consulta la VRAM libre. Cuenta backbone, compartidos, buffers KV ya configurados, expertos calientes, espacio para sus stacks temporales, descompresión y el margen indicado. `--memory-mib` limita adicionalmente la VRAM libre utilizable; no es un límite de memoria impuesto por CUDA. La CPU solo sirve aquí para pruebas lógicas y requiere ese presupuesto explícito.

Añadir `--execute` y otro nombre de salida para generar. El informe incluye selección, memoria y contadores antes/después de la generación. Estos deben mostrar cero cargas/transferencias de expertos durante la respuesta. El margen por defecto es una hipótesis conservadora que debe contrastarse con el pico real de CUDA.

Sin informe de calidad, la política se identifica como `unmeasured`. Con `--max-nll-delta`, solo acepta configuraciones con evidencia correspondiente al mismo checkpoint, manifiesto y conjunto de pools. Si falta evidencia, bloquea la ejecución. Los JSON de salida no se sobrescriben.

## Calibrar calidad sin reentrenar la LLM

```powershell
python -m asi adaptive evaluate --checkpoint $model --pool-manifest $pools --data-root D:/datasets/specialized_fineweb_10bt --contexts 'health;biology;health,biology' --seq-len 128 --windows-per-pool 16 --device cuda --output results/adaptive_quality_v1.json
```

Compara residencia densa e INT8 sobre las mismas ventanas de validación. Para la LLM categorizada, la referencia utiliza el pool correcto de cada ventana y su routing original; para una receta base, usa sus expertos sin restricción. Guarda NLL, diferencia de NLL por pool, acierto de siguiente token, memoria y contadores. Cada informe conserva las ventanas utilizadas.

```powershell
python -m asi adaptive run --checkpoint $model --pool-manifest $pools --pools health,biology --prompt 'Explain how bacteria affect health.' --quality-report results/adaptive_quality_v1.json --max-nll-delta 0.10 --warm-mib 64 --execute --output results/run_health_biology.json
```

El umbral 0,10 es solo un ejemplo de tolerancia que debe decidir el estudio. Se aplica a la **diferencia media de NLL observada** en las ventanas, no a una garantía sobre cada respuesta. Consultar las diferencias por pool para no ocultar degradación de un tema en la media.

Estos datos de validación pasan a ser datos de calibración de la política: reservar otras ventanas/documentos para la evaluación final. Dos pools se calibran aquí con ventanas de ambos dominios por separado; eso **no demuestra** calidad en preguntas mixtas. La evaluación mide forwards con texto conocido, no velocidad autoregresiva ni costes repetidos de cambio de contexto. Esos estudios siguen siendo necesarios.

## Clasificador integrado y presupuesto

El clasificador inicial toma la media de los embeddings de hasta los últimos 512 tokens y aplica LayerNorm y una pequeña red de dos capas. Reutiliza la matriz de embeddings; no ejecuta expertos para decidir cuáles cargar. Es un baseline de bolsa de embeddings: no representa el orden de las palabras ni sustituye una evaluación semántica.

La cabeza se guarda en un archivo separado. No se actualiza ningún peso de la LLM. Para dimensión 512, ocho etiquetas y capa oculta de 128, añade **67.720 parámetros**. La LLM de 122.325.504 más la cabeza suma **122.393.224**, aunque el informe mantiene los costes separados. La matriz compartida no se cuenta dos veces. La implementación CLI extrae características en CPU y permite entrenar la cabeza en CPU/GPU.

Con los shards categorizados:

```powershell
python -m asi adaptive train-classifier --checkpoint $model --pool-manifest $pools --data-root D:/datasets/specialized_fineweb_10bt --steps 1000 --batch-size 16 --output results/integrated_head_v1
```

Ese dataset asigna un único pool a cada ventana. El clasificador admite varias salidas positivas, pero **su capacidad multietiqueta no queda entrenada ni demostrada con esos ejemplos**. Para aprender etiquetas simultáneas, proporcionar ejemplos reales separados en entrenamiento/validación mediante `--examples ejemplos.jsonl`, en lugar de `--data-root`:

```json
{"text":"Explain the mathematics behind this learning algorithm.","labels":["technology","math_physics"],"split":"train"}
{"text":"Describe how forests store carbon.","labels":["biology","environment"],"split":"val"}
```

Son ejemplos del formato, no un dataset suficiente. Se rechazan entradas efectivas idénticas entre train y val; la separación documental y temática sigue siendo responsabilidad del dataset. El informe registra BCE, F1 micro, coincidencia exacta de etiquetas y memoria de la cabeza. Usa umbral fijo 0,5 y, si ninguna etiqueta lo alcanza, escoge la mayor puntuación.

Para utilizarla, sustituir `--pools` por `--classifier results/integrated_head_v1/head.pt` en `adaptive run`. Se verifica la identidad de los embeddings y la taxonomía. Una cabeza entrenada sobre un checkpoint intermedio debe volver a entrenarse si cambian sus embeddings. Por eso conviene esperar al checkpoint final para el clasificador definitivo.

El clasificador externo previo sigue disponible en `chat domain`; su memoria, tokenizer y tiempo deben medirse por separado. El nuevo modo integrado no lo descarga ni lo ejecuta.

## Cómo funciona INT8

`WarmExpertCache` cuantiza por fila de salida: pesos INT8 y escalas FP32. Conserva el respaldo original en RAM y otra representación cuantizada en RAM; ambos costes aparecen separados. El presupuesto warm guarda pools completos comprimidos en GPU. Los pools activos están descomprimidos, en el tipo de sus pesos originales.

Al cambiar de contexto, un acierto warm descomprime desde GPU; un fallo sube la representación cuantizada desde RAM y la descomprime. Las expulsiones restauran el respaldo RAM. La cuantización también se aplica a las cargas frías para que la precisión no dependa de si hubo cache hit. No hay un kernel de multiplicación INT8: **es almacenamiento comprimido con cómputo flotante**.

Durante la respuesta no se descomprime ni cambia ningún experto. La caché no reduce el tamaño del pool activo y no permitirá ejecutar un conjunto activo que ya exceda el presupuesto. Una reserva warm puede reducir futuras transferencias, pero consume memoria adicional; su beneficio debe medirse con sesiones de temas cambiantes. La RAM conserva todos los expertos originales en esta primera implementación: el arranque por shards desde SSD ya existente todavía no se integra en este nuevo runtime.

## Validación y siguientes medidas

`tests/test_adaptive.py` verifica unión y restauración del routing, ausencia de E, error de cuantización, independencia del historial, rechazo sin evidencia, entrenamiento aislado de la cabeza y calibración de extremo a extremo. La prueba CUDA verifica tensores INT8 físicamente en GPU, promoción sin transferir pesos desde RAM y contadores constantes durante prefill/decode. Estas pruebas usan pesos pequeños sintéticos, no resultados de calidad del modelo del TFG.

Después del entrenamiento: entrenar la cabeza, calibrar políticas, reservar el test final y medir preguntas multitema, cambios repetidos de contexto, RAM/VRAM máxima, bytes transferidos, latencia y calidad. Mantener los checkpoints y las recetas de entrenamiento originales para que todas las variantes partan de los mismos pesos.
