# Evaluación del modelo categorizado terminado

Checkpoint local: `model_37595.pt`, paso 37.595 (37.596 actualizaciones), 122.325.504 parámetros. Entrenado en homelab con batch 8, contexto 1.024, batch global 262.144 y `uniform_pool`. Sus ocho pools son technology, math_physics, biology, health, environment, society_economy, history_arts y education_language. Las mediciones de esta evaluación corresponden a la **RTX 3060 local**, no a la 5060 Ti de entrenamiento.

## Qué significa liberar los expertos en este checkpoint

Durante el entrenamiento se ejecutaban los dos expertos de un pool con pesos iguales. Los parámetros del router E no recibían gradientes. Por eso se distinguen tres pruebas sobre exactamente los mismos pesos:

| Nombre en los JSON | Comportamiento | Interpretación |
|---|---|---|
| classified | Clasificador de entrada elige un pool; sus dos expertos permanecen en GPU y se activan con igual peso | Uso previsto del modelo categorizado |
| all_uniform | Los 16 residen y se ejecutan; cada uno recibe 1/16 del peso de mezcla | Ablación de mezclar todas las especialidades; cambia también el cómputo |
| all_router | Los 16 residen, E escoge dos por token | Diagnóstico de un router **sin entrenar**; no es el control de un MoE preentrenado libre |

El nuevo entrenamiento base será el control correcto para estudiar libertad aprendida. Que una ablación empeore no prueba que los expertos no tengan conocimiento; que mejore no demuestra conservación de todas las capacidades.

## Benchmarks comparables con el proyecto GPT-2

**Actualización:** la ejecución rápida mantiene todos los expertos en VRAM y activa
solo el pool seleccionado. Véase [BENCHMARKS_RAPIDOS.md](BENCHMARKS_RAPIDOS.md) para
el comando completo optimizado. Los resultados históricos de memoria y tiempos
de esta página corresponden a la configuración antigua, con residencia limitada.

Se importan directamente las funciones de `Creating-GPT-2-From-0/eval.py`: HellaSwag, LAMBADA, PIQA, ARC-Easy, BoolQ y aritmética sintética. Se conservan prompts, tokenización GPT-2, normalización de respuestas y selección de opciones. Se registra el hash del evaluador y la huella de los datasets de Hugging Face.

La primera ejecución usa los **primeros 100 casos de cada tarea**, con la misma semilla 42 para la aritmética. Es un diagnóstico exploratorio de seis tareas, no la evaluación completa de todos los benchmarks del proyecto GPT-2. La clasificación ve únicamente el contexto; nunca la respuesta correcta ni las opciones a puntuar. Sus decisiones se reutilizan entre las opciones de una pregunta. El clasificador no se ha recalibrado utilizando estos tests.

```powershell
python -m asi checkpoint-eval --checkpoint model_37595.pt --gpt2-evaluator ../Creating-GPT-2-From-0/eval.py --output results/checkpoint_37595_new/report.json
```

Usar una salida nueva. Para ampliar la muestra, cambiar `--limit`; para otras tareas, `--tasks`. Se requiere disponer del proyecto GPT-2 indicado. El evaluador es código Python local de confianza: no apuntar este argumento a código descargado sin revisar. No se modifican archivos del proyecto vecino.

**Batería completa en Bash**, sin truncar datasets y con todos los benchmarks registrados en el evaluador:

```bash
python -u -m asi checkpoint-eval --checkpoint model_37595.pt --gpt2-evaluator ../Creating-GPT-2-From-0/eval.py --pool-manifest configs/experiment_124m_v1/expert_pools.json --tasks all --limit 0 --modes classified all_uniform all_router --device cuda --output "results/eval_completa_$(date +%Y%m%d_%H%M%S)/report.json"
```

`--limit 0` usa todos los ejemplos de cada split; la aritmética sintética conserva el valor predeterminado del evaluador (500). Algunas tareas requieren descargar datasets. Los fallos de acceso quedan en el informe y este no se marca completo si hay errores. Esta ejecución puede ser muy larga. Las métricas reproducen el script GPT-2: por ejemplo, TriviaQA/WebQuestions usan inclusión de texto y GSM8K extracción del último número, no necesariamente los protocolos oficiales. El informe se conserva tras cada tarea, pero no existe reanudación automática de una tarea interrumpida.

Para repetir el GPT-2 propio sobre idéntica muestra:

```powershell
python ../Creating-GPT-2-From-0/eval.py --model ../Creating-GPT-2-From-0/models/gpt2Base.pt --tasks hellaswag,lambada,piqa,arc_easy,boolq,arithmetic --limit 100 --seed 42 --device cuda --output results/checkpoint_37595_eval_verified/gpt2_matched.json
```

Los resultados completos anteriores de GPT-2 tienen otro número de preguntas y no se mezclan con esta tabla. GPT-2 tiene 124.475.904 parámetros y una arquitectura diferente: es una referencia de calidad, no una ablación causal del modelo ASI. Su cálculo FP32/TF32 y su implementación difieren de la autocast BF16 de ASI; no comparar sus tiempos como una mejora exclusiva del routing.

## Generación y rendimiento

Ocho prefijos de texto, uno por ámbito, generación greedy hasta 64 tokens y tres repeticiones temporales. Se usa decodificación incremental con KV, pesos FP32, autocast BF16 y backend bmm. Se miden tokens/s, pesos residentes, pico asignado por CUDA, transferencias y cargas de expertos. El tiempo de generación empieza después de clasificar y preparar la caché. Los tiempos de tareas incluyen clasificación, carga, puntuación y Python: **no son tokens/s de decodificación**.

El número de expertos residentes se refiere a cada capa MoE; existen además expertos compartidos y backbone. Mantener 16 residentes no implica activar 16: all_router activa dos. La memoria publicada distingue pesos de pico del asignador; no se ha medido aquí un pico RSS continuo del proceso ni la memoria del sistema operativo. La caché conserva respaldo RAM de todos los expertos; este modelo cabe completo en la GPU, de modo que esta prueba no valida el rendimiento con SSD bajo presión de memoria.

Control de categorías conocidas para las ocho continuaciones (sin benchmarks y con una sola repetición):

```powershell
python -m asi checkpoint-eval --checkpoint model_37595.pt --gpt2-evaluator ../Creating-GPT-2-From-0/eval.py --tasks= --modes classified --oracle-probes --probe-repeats 1 --new-tokens 64 --output results/checkpoint_37595_oracle/report.json
```

Aquí **oracle** solo significa indicar manualmente el tema correcto; no proporciona la respuesta al modelo. Este control separa fallos de clasificación de las limitaciones del generador. No extrapolar ocho prefijos a una tasa de coherencia general.

## Resultados y gráficas

**Medido el 04/10/2026**, 100 casos por tarea:

| Variante | HellaSwag normalizado | LAMBADA | PIQA | ARC-Easy | BoolQ | Aritmética |
|---|---:|---:|---:|---:|---:|---:|
| Pareja por categoría | 38% | 14% | 66% | 42% | 57% | 0% |
| Mezcla uniforme de 16 | 38% | 23% | 67% | 36% | 53% | 0% |
| E sin entrenar, top-2 libre | 39% | 19% | 65% | 35% | 57% | 0% |
| GPT-2 propio, misma muestra | 36% | 23% | 70% | 40% | 52% | 0% |

LAMBADA tiene perplejidad objetivo 121,64 / 117,14 / 125,50 / 49,03 en ese orden. Igualar el porcentaje de aciertos no significa igualar la distribución de probabilidades. Los resultados son descriptivos; esta muestra no permite declarar un ganador general ni significación de cada diferencia.

| Variante ASI | Expertos residentes GPU por capa | Pesos GPU | Pico CUDA asignado | Mediana de generación |
|---|---:|---:|---:|---:|
| Pareja por categoría | 2 | 358,35 MiB | 707,96 MiB | 9,21 tokens/s |
| Mezcla uniforme de 16 | 16 | 466,63 MiB | 815,93 MiB | 5,90 tokens/s |
| E sin entrenar, top-2 libre | 16 | 466,63 MiB | 816,10 MiB | 8,24 tokens/s |

No hubo cargas de expertos durante ninguna de las 72 generaciones medidas. Sí hay cargas al preparar cambios de categoría. El ahorro de pesos residentes es aproximadamente 23,2%; el backbone y los expertos compartidos dominan el tamaño de esta arquitectura pequeña.

La inspección de las ocho continuaciones muestra repeticiones y errores en las variantes, incluyendo una derivada incorrecta. No hay evidencia de que abrir los expertos resuelva la coherencia. En el control manual de categorías, «Inflation is an increase in…» pasa del tema salud a precios, pero sigue degenerando en repeticiones y afirmaciones erróneas. Corregir la clasificación ayuda a ese caso, pero no resuelve por sí solo la generación. Informe separado: `results/checkpoint_37595_oracle/report.json`; no se ha usado ese control para ajustar el clasificador sobre el test.

Informe medido: `results/checkpoint_37595_eval_verified/report.json`; comparación con GPT-2: `gpt2_matched.json` en la misma carpeta. El intento anterior `results/checkpoint_37595_eval` quedó interrumpido por permisos de escritura de la caché de datasets y no constituye la evaluación final.

```powershell
python -m asi checkpoint-report --report results/checkpoint_37595_eval_verified/report.json --gpt2-report results/checkpoint_37595_eval_verified/gpt2_matched.json --output results/checkpoint_37595_summary
```

Genera tabla legible, gráficos PNG, textos y casillas de revisión vacías. Los intervalos Wilson son descriptivos, no pruebas pareadas. No se calcula una supuesta puntuación de inteligencia a partir de pérdida, velocidad o repeticiones.

El [estado del arte](ESTADO_ARTE_2026-10-04.md) sitúa estas pruebas frente a los trabajos del tutor. El [siguiente entrenamiento](TRAINING_124M.md) y la [auditoría de profesores](RECOVERY_V3.md) mantienen separados los resultados medidos y las etapas pendientes.

## Fine-tuning supervisado del modelo categorizado

El piloto del 05/10/2026 añade una variante del checkpoint 37595 ajustada con respuestas de OpenAssistant/oasst1. **200 actualizaciones no son 200 épocas**: con cuatro ejemplos por actualización y 1024 ejemplos de entrenamiento, son aproximadamente 0,78 recorridos. No se presupone que más entrenamiento mejore los benchmarks.

`asi/experiments/domain_sft.py` reutiliza las funciones de preparación y tokenización de `../Creating-GPT-2-From-0/finetune_oasst.py`. Se necesitan ese archivo y los datos Hugging Face disponibles o descargables. Conserva los marcadores `<|user|>`/`<|assistant|>`, el tokenizador GPT-2 y la pérdida solo sobre tokens del asistente, incluido EOS. También conserva el truncado izquierdo original: respuestas largas pueden perder contexto del usuario; el piloto usa 256 tokens.

Se seleccionan 1024 conversaciones de entrenamiento y 128 de validación, con semilla 1337, desde sus splits originales y verificando raíces distintas. Los manifiestos guardan identificadores, hashes y distribución por pools. No se usan los benchmarks para elegir checkpoint. El clasificador congelado selecciona un pool a partir de los mensajes del usuario, sin ver las respuestas; se mantiene durante cada conversación.

La muestra contiene 176077 tokens supervisados de entrenamiento y 22011 de validación. No está equilibrada por categoría: solo hay 10 ejemplos de medioambiente en entrenamiento y ninguno en validación; salud tiene dos en validación y biología cuatro. La selección por pérdida agregada no garantiza mejora en todos los pools. Ampliar y equilibrar esta cobertura será necesario antes de conclusiones por categoría.

Se ajustan pesos completos del backbone, expertos compartidos y expertos elegidos por pool. El router queda congelado porque `uniform_pool` no lo utiliza. **No es LoRA ni el experimento de recuperación del OLMoE público**. Entrenamiento con AdamW, lr 1e-5, betas 0,9/0,95, weight decay 0,01, clip 1, BF16 autocast y backend bmm. En entrenamiento todos los pesos están residentes; el ahorro de memoria se mide posteriormente en inferencia.

La validación pondera por tokens del asistente. Se evalúa antes de entrenar y cada 50 actualizaciones; se conserva el mejor checkpoint, incluido el inicial si ningún ajuste mejora. Hay parada tras tres validaciones sin mejora. Solo se guardan `best.pt` y `last.pt`, de forma atómica, manteniendo intacto `model_37595.pt`. Este piloto no implementa reanudación del optimizador ni autorestart; utilizar carpetas nuevas para repetirlo.

Desde la raíz del repositorio, con el proyecto GPT-2 al lado:

```bash
python -m asi domain-sft prepare --data data/domain_sft_oasst_pilot
python -u -m asi domain-sft train --data data/domain_sft_oasst_pilot --checkpoint model_37595.pt --steps 200 --output results/domain_sft_oasst_pilot
python -u -m asi checkpoint-eval --checkpoint results/domain_sft_oasst_pilot/best.pt --gpt2-evaluator ../Creating-GPT-2-From-0/eval.py --modes classified --limit 100 --output results/domain_sft_oasst_pilot/eval.json
python -m asi checkpoint-report --report results/checkpoint_37595_eval_verified/report.json --sft-report results/domain_sft_oasst_pilot/eval.json --gpt2-report results/checkpoint_37595_eval_verified/gpt2_matched.json --output results/domain_sft_comparison
```

Los seis benchmarks usan las mismas preguntas y formatos que el control original; solo cambia el checkpoint. `checkpoint-report --sft-report` comprueba procedencia, evaluador, datasets, límites y número de ejemplos antes de combinarlos. El GPT-2 original queda como referencia externa; su corpus y presupuesto de entrenamiento son distintos.

Para observar seguimiento de instrucciones, `checkpoint-eval --tasks= --modes classified --chat-probes --probe-repeats 1` permite ocho preguntas con el formato OASST. Ejecutarlo tanto en el checkpoint original como en el ajustado, en informes distintos. Estos textos requieren revisión humana; no constituyen una métrica automática de coherencia. No mezclar sus tiempos con los de prefijos de continuación. Para ampliar después los benchmarks usar `--tasks all --limit 0` **en ambas variantes**.

### Entrenamiento medido del piloto

El piloto completó 200 actualizaciones (800 ejemplos presentados, 0,78125 épocas efectivas). El tiempo de las actualizaciones, validaciones intermedias y guardados fue 211,4 segundos en la RTX 3060; excluye preparación, carga y validación inicial. No es una comparación de velocidad con GPT-2.

| Actualización | Pérdida de validación OASST, solo asistente |
|---:|---:|
| 0 | 3,10858 |
| 50 | 2,88454 |
| 100 | 2,84228 |
| 150 | 2,82084 |
| 200 | 2,81075 |

Se seleccionó la actualización 200. Se ajustaron 122235392 parámetros (routers congelados); no cambió el número de expertos. Esta pérdida no se debe comparar directamente con la validación FineWeb del preentrenamiento. Informe: `results/domain_sft_oasst_pilot/report.json`; modelo seleccionado: `best.pt` en esa carpeta.

Los mismos seis benchmarks (100 casos cada uno) muestran un resultado mixto:

| Variante, dos expertos por categoría | HellaSwag norm. | LAMBADA | PIQA | ARC-Easy | BoolQ | Aritmética |
|---|---:|---:|---:|---:|---:|---:|
| Original | 38% | 14% | 66% | 42% | 57% | 0% |
| SFT, actualización 200 | 38% | 19% | 69% | 40% | 64% | 0% |

La perplejidad objetivo de LAMBADA baja de 121,64 a 103,64. Esto no demuestra significación estadística ni recuperación general: el prefijo de la derivada sigue generando una afirmación incorrecta y repetida. Se conserva el original como control y como alternativa. No se usaron estos resultados para seleccionar la actualización 200.

La variante SFT mantiene 2 expertos residentes por capa, 358,35 MiB de pesos GPU y un pico asignado de 707,96 MiB. Las 24 continuaciones no cargaron expertos durante la respuesta. Se midieron 11,63 tokens/s de mediana, frente a 9,21 en la sesión anterior del original; **no atribuir esta diferencia al SFT**, ya que las ejecuciones no se intercalaron bajo condiciones de carga controladas. La arquitectura y el coste teórico permanecen iguales. Comparación con gráficas: `results/domain_sft_comparison/summary.md`.

### Revisión de conversación antes y después

Se ejecutaron ocho preguntas idénticas con marcadores OASST, hasta 64 tokens greedy y la misma clasificación de entrada. Informes: `results/domain_sft_oasst_pilot/chat_before.json` y `chat_after.json`; lectura emparejada: `results/domain_sft_comparison/chat_review.md`.

El original frecuentemente copia la pregunta o repite los marcadores de conversación. El ajustado produce frases de respuesta y en dos casos llega a EOS, pero sigue sin dar el código de ordenación solicitado, falla la derivada y repite afirmaciones incorrectas. Hay adaptación visible al formato, **no evidencia de recuperación suficiente de corrección o seguimiento de instrucciones**. Estas ocho observaciones no son una tasa general de calidad. El paso siguiente sería ampliar cobertura y exposición de entrenamiento manteniendo validación y benchmarks separados, no asumir que repetir cientos de épocas resolverá los errores.
