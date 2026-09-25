# Comparación de tres modelos: protocolo preparado, sin ejecutar

Esta batería corresponde a la opción 1. Su objetivo es medir cuánta calidad se
pierde al reducir la residencia de expertos y cuánto swapping queda. No busca
demostrar que la descarga de pesos acelera un modelo que ya cabe en GPU.

**Estado a 25/09/2026:** entradas y matriz preparadas en
`results/comparison_prepared_v1`. No se ha cargado ni evaluado ningún checkpoint
entrenado para esta batería. Las rutas de los tres modelos están deliberadamente
vacías. Los tests de software utilizan datos sintéticos y modelos diminutos.

## Qué se compara

| Modelo | Control | Intervenciones |
|---|---|---|
| `original` | Todos los expertos en GPU, E original | Caché exacta LRU con 2, 4 y 8 plazas por capa; conserva las decisiones de E |
| `classified` | Su propio checkpoint completo | Lo anterior; candidatos por clase con E (2, 4, 8); dos expertos fijos sin E; pareja global sin clase |
| `domain` | Su propio checkpoint con todos los pools accesibles | Caché exacta; candidatos del pool entrenado; dos expertos fijos sin E |

Las clases se prueban de dos maneras: etiqueta conocida del corpus (*oracle*) y
predicción del clasificador. Oracle no significa experto ideal: solo conoce la
categoría de la entrada. Las clases pueden compartir expertos.

La caché exacta cambia la residencia, pero debe conservar rutas, argmax y NLL
frente a su control completo. La restricción y la selección fija sí cambian la
función del modelo. En `fixed` cada experto recibe `route_scale/2`: la degradación
incluye el cambio de índices y de pesos de mezcla. El control de masa calibrada
de la opción 2 sigue disponible en `fixed-study`; esta matriz usa mezcla uniforme.

**No se inventan expertos para llenar una clase.** Si un pool entrenado tiene dos
expertos, las variantes con cuatro u ocho candidatos quedan como `skipped`, con
motivo. No se amplía silenciosamente con expertos de otros pools. Tampoco se
simula una capacidad superior al número real de expertos de la capa.

Para aislar la categorización posthoc, `original` y `classified` deberían apuntar
a los mismos pesos, con la clasificación como intervención. Si se entrenan por
separado, sus diferencias incluyen el entrenamiento. El contraste contra el
control residente del propio checkpoint sigue siendo válido. Ocho frente a
dieciséis expertos tampoco es una comparación causal de igual tamaño.

## Entradas y cobertura congeladas

El bundle contiene hashes, ventanas exactas, conversaciones, configuración y
orden aleatorio reproducible de los trabajos. No contiene resultados del modelo.

| Longitud de entrada | Ventanas disponibles | Tokens puntuados por ventana |
|---|---:|---:|
| 128 | 813 | 65 |
| 512 | 445 | 257 |

Se solicitaron hasta 256 ventanas por clase y longitud. El corpus actual aporta
**1.258 ventanas**, con **167.210 tokens objetivo** en total entre ambas longitudes.
Las longitudes se analizan por separado: sus fragmentos pueden solaparse. Dentro
de cada longitud el muestreo utiliza bloques disjuntos y no repite ventanas para
completar cuotas. No hay validación de IA; se informa como ausente, nunca se
sustituye por train. El promedio por token y el promedio entre dominios presentes
se publican por separado para mostrar el desequilibrio.

Para longitud L se guardan L+1 tokens. El clasificador ve únicamente los primeros
L/2 tokens; el resto es la continuación reservada. La clase queda fija mientras
se puntúa esa continuación mediante teacher forcing. Las ventanas y los tokens
objetivo son idénticos para todos los modelos. Se requiere la misma tokenización
GPT-2 del dataset actual; no es un adaptador universal para checkpoints externos.

Hay **40 conversaciones de seis turnos**: ocho dominios iniciales por cinco
escenarios —tema estable, cambio único, alternancia, regreso y preguntas mixtas—.
Son entradas redactadas para forzar cambios, no un benchmark de conocimientos.
En las mixtas, la etiqueta oracle es el primer tema declarado, no una solución
multilabel. La variante fija sigue utilizando una sola clase.

Las generaciones usan 32 y 128 tokens como máximo, tres repeticiones y estados
de caché frío/caliente. Frío significa caché de expertos vacía al comenzar cada
conversación; puede calentarse en sus turnos. Caliente significa una reproducción
previa completa de esa conversación, excluida de las métricas. Es reutilización
de una carga conocida, no predicción del futuro. EOS puede acortar la generación.

Todos reciben el mismo historial de **mensajes del usuario**, limitado a los
últimos 128 tokens; las respuestas generadas no se añaden. Así, una respuesta
distinta no altera los prompts posteriores. Cada turno reinicia KV y vuelve a
procesar su contexto. Durante el decode cada variante sigue sus propios tokens:
esa fase mide ejecución libre, mientras la comparación pareada de calidad usa
las ventanas reservadas. El prefill es serial en todos los brazos, necesario para
medir demanda token a token con solo dos plazas; no representa el máximo
rendimiento de un prefill batched.

La configuración genera **742 trabajos** antes de excluir variantes incompatibles.
Es una batería extensa y divisible: `run-job` ejecuta un trabajo, en su propio
proceso y carpeta. No se ha estimado su duración mediante ejecuciones reales.

## Qué significa cada resultado

| Medida | Pregunta que responde |
|---|---|
| NLL, perplexity y acierto del siguiente token | ¿Cuánto cambia la predicción sobre la misma continuación? No es un porcentaje de capacidad general. |
| Delta pareada contra el residente propio | ¿Qué pierde este checkpoint por la intervención de routing/residencia? |
| Delta contra el original | ¿Cómo difieren los sistemas completos? Revisar entrenamiento y tamaño antes de atribuir causas. |
| Precargas y bytes al preparar la clase | ¿Cuánto cuesta cambiar de contexto antes de responder? |
| Misses, bytes y fracción de pasos sin cargas | ¿Se sigue haciendo swapping durante prefill o decode? Un hit rate alto no garantiza respuestas sin swapping. |
| Clasificación, preparación, p50/p95 por paso | ¿Dónde se consume el tiempo? Las medidas incluyen instrumentación y sincronizaciones. |
| Pesos de expertos, tensores, CUDA asignada/reservada y RSS | ¿Qué memoria se ahorra realmente? RAM es el almacenamiento; CPU es el procesador. |

RSS es memoria residente observada al final de cada ventana/turno, no un máximo
continuo del proceso. CUDA sí registra el pico del allocator. Memoria reservada y
asignada no se suman; tampoco se duplican los alias CPU de la copia de respaldo.
Los expertos fríos se mantienen en RAM en esta batería. SSD y RAM limitada siguen
siendo una línea separada mediante `export-store` y el chat existente: aquí no se
presentarán lecturas lógicas como I/O físico ni resultados SSD inexistentes.

El resumen calcula un bootstrap descriptivo de ventanas para los contrastes de
NLL. No presupone documentos independientes. Las repeticiones temporales quedan
separadas y no multiplican el tamaño de muestra de calidad. Los jobs ausentes,
fallidos o excluidos nunca se convierten en ceros. Una comprobación de caché exacta
fallida debe investigarse antes de interpretar degradación.

## Preparación y registro posterior

La preparación ya realizada puede reproducirse, sin inferencia:

```powershell
python -m asi comparison prepare --config configs/comparison_suite.json --output results/comparison_prepared_v2
python -m asi comparison check --suite results/comparison_prepared_v1
```

`check` comprueba integridad y existencia de rutas; **no certifica que los modelos
estén entrenados** y no los carga. Ahora informa las rutas pendientes.

Cuando estén entrenados los tres modelos:

1. Copiar `configs/comparison_suite.json` a una configuración local final.
2. Completar solo `models`: rutas de checkpoints y arquitectura, mapping del
   modelo clasificado y metadatos de `training`. Conservar tokenizer, datos,
   tokens vistos, steps, seed, inicialización, optimizador y presupuesto de batch.
   Los hashes del checkpoint y el número de parámetros se registran al ejecutar.
   Los metadatos declarados no sustituyen la verificación de igualdad experimental.
3. Crear el mapping posthoc exclusivamente con train y conservar hasta ocho
   candidatos por clase. `posthoc calibrate --experts-per-label 8` ya exporta
   `provenance.calibration_split` y `global_layers`. Ese comando ejecuta modelos:
   queda para la fase futura. Los mappings antiguos sin esa procedencia deben
   regenerarse; no basta con renombrar un archivo o editar la etiqueta de split.
4. Registrar las rutas sin volver a muestrear las entradas:

```powershell
python -m asi comparison bind-models --suite results/comparison_prepared_v1 --config configs/comparison_trained.local.json --output results/comparison_ready_v1
python -m asi comparison check --suite results/comparison_ready_v1
```

El comando acepta únicamente cambios en los modelos. Si se cambia el protocolo,
hay que preparar un bundle nuevo. Las carpetas existentes no se sobrescriben.

## Ejecución futura, no realizada ahora

Consultar `jobs.json` para elegir el ID deseado. Empezar por los controles
residentes y de caché exacta, después las intervenciones con dos plazas y finalmente
el resto del barrido. No ejecutar simultáneamente varios jobs en la misma GPU.

```powershell
python -m asi comparison run-job --suite results/comparison_ready_v1 --job job_00000 --output results/comparison_runs_v1/job_00000
```

El ID anterior es un ejemplo; el orden está barajado y no garantiza que sea un
control. Para recorrer la matriz posteriormente, en un solo proceso GPU a la vez:

```powershell
$comparisonJobs = Get-Content results/comparison_ready_v1/jobs.json -Raw | ConvertFrom-Json
foreach ($comparisonJob in $comparisonJobs) {
    python -m asi comparison run-job --suite results/comparison_ready_v1 --job $comparisonJob.id --output "results/comparison_runs_v1/$($comparisonJob.id)"
    if ($LASTEXITCODE -ne 0) { throw "Trabajo fallido: $($comparisonJob.id)" }
}
```

No usar este bucle hasta revisar los controles. Para reanudar, seleccionar solo
IDs pendientes y usar una carpeta nueva para reintentos. No mezclar dos resultados
con el mismo ID en la carpeta a resumir.

```powershell
python -m asi comparison summarize --suite results/comparison_ready_v1 --results results/comparison_runs_v1 --output results/comparison_summary_v1.json
```

Cada job deja `report.json` con estado, hashes y hardware, y `records.jsonl` con
observaciones, rutas finales por capa y transferencias por token. El resumen es
offline: no lanza trabajos pendientes. Rechaza checkpoints, mappings o versiones
del runner mezclados; conserva las exclusiones y los trabajos que faltan.

## Correspondencia con el TFG y límites actuales

| Archivo | Pasos | Función |
|---|---|---|
| `configs/comparison_suite.json` | 1.2, 1.3, 1.5, 2.2, 2.3 | Presupuestos, modelos, longitudes y repeticiones |
| `asi/experiments/comparison.py` | 1.2, 1.5, 2.2, 2.3 | Congelar datos, sesiones y matriz; registrar los modelos posteriormente |
| `asi/experiments/comparison_run.py` | 1.2, 1.3, 1.5, 2.2, 2.3 | Ejecución explícita, calidad y trazas RAM/GPU por token |
| `asi/experiments/comparison_report.py` | 1.2, 1.3, 2.2 | Agregación por dominio, contrastes pareados y métricas de swapping |
| `tests/test_comparison_suite.py` | Verificación transversal | Congelación, procedencia, cuotas, matriz y agregaciones con datos sintéticos |

No se considera demostrado ningún resultado nuevo. Faltan la ejecución con los
tres modelos terminados, revisión de controles exactos, tareas externas con
respuestas verificables y, para conclusiones sobre entrenamiento, semillas y
presupuestos emparejados. La clasificación de IA necesita validación reservada
adicional. Esas carencias permanecen visibles aunque el runner esté preparado.
