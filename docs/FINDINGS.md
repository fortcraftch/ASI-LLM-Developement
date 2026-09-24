# Resultados actuales e interpretación

Son pilotos locales anteriores al refactor. Los originales permanecen en
`results/`, excluido de Git. Reorganizar código no constituye un nuevo experimento.

## Entrenamiento por pools — 1.1–1.3

El modelo local tiene 16 expertos en ocho pools y top-2. El checkpoint de diez
pasos verificó el flujo. `results/expert_audit/report.json` contiene 36 casos
(12 prompts por tres modos), sin violaciones de máscara observadas.

Esto respalda aislamiento de rutas, no expertos competentes ni conocimiento
exclusivo. Los tests comprueban que AdamW no actualiza expertos inactivos;
backbone y expertos compartidos siguen compartiendo aprendizaje.

## Modelo original entrenado — 2.1 y 2.2

`results/expert_audit_training/model_04750.pt` declara step 4750 y validación
de entrenamiento 3,3454. Tiene 12 capas: una densa y 11 MoE con ocho expertos y
top-2 por capa; **88 instancias enrutadas**, más expertos compartidos. No es
el DeepSeek oficial completo de 671B.

Evidencia local: [resumen](../results/posthoc_04750/SUMMARY.md),
[informe](../results/posthoc_04750/report.json) y
[etiquetas candidatas](../results/posthoc_04750/expert_labels.json).

Calibración: 32 ventanas de 128 tokens por ocho dominios (32.768 tokens).
Evaluación: ocho ventanas por siete dominios (7.168 tokens); **AI no tiene
validación disponible**. Es el mismo corpus del entrenamiento y no se ha
certificado independencia documental.

| Comparación con las mismas ventanas | Resultado |
|---|---:|
| NLL nativa, todos residentes | 3,7581 |
| NLL de caché prefetch conservando routing | 3,7581 |
| NLL restringiendo a cuatro candidatos por dominio/capa | 4,0932 |
| Incremento de NLL al restringir | +0,3352 |
| Cobertura semántica K=4 | 76,31% |
| Cobertura por popularidad global K=4 | 76,13% |

La precarga conservó rutas y argmax, y coincidieron las 64 columnas de logits
muestreadas por token. No se verificó todo el vocabulario. Restringir empeoró
NLL. La ventaja temática es solo de unos **0,18 puntos porcentuales de cobertura**:
no demuestra una ventaja práctica sólida frente a popularidad global. Las
etiquetas son asociaciones provisionales, no profesiones probadas.

La NLL de ventanas no debe compararse directamente con el 3,3454 del entrenamiento:
son protocolos distintos. Tampoco demuestra corrección en tareas externas.

## Memoria y tiempo — 1.3, 1.4 y 2.2

Los 88 expertos pesan 115,5 MiB FP32. Un límite de 52 deja como máximo 68,25 MiB
de esos pesos en GPU: **47,25 MiB menos, 40,9% de los pesos de expertos**. No es
40,9% de VRAM total. Backbone/shared ocupan unos 356,1 MiB y buffers unos
216,25 MiB; faltan temporales, allocator y clasificador. Se mantiene backing RAM.

El piloto observó aproximadamente 0,116 s por ventana residente, 0,131 s con
caché exacta y 0,093 s con restricción, sin sumar aquí precarga. Es una ejecución,
no un benchmark repetido: la caché exacta ahorra memoria y añade transferencias;
el modo restrictivo pierde calidad. No permite prometer aceleración general.

`results/posthoc_04750/prompt_sessions.json` contiene 12 entradas con clasificador
real y ocho tokens generados por entrada. Comprueba integración, no calidad ni
conversaciones largas. El generador original reprocesa contexto por token;
estos tiempos no representan decode incremental optimizado.

## Evidencia pendiente tras el primer piloto

Comparar precarga semántica, popularidad y LRU con igual capacidad GPU, sesiones
separadas de calibración y coste total por turno. Incluir bytes y precargas junto
al hit rate. Para 1.1–1.2 hace falta entrenamiento suficiente y control emparejado.

Los pasos 3.1–3.3 siguen pendientes. El exportador INT8 no demuestra compresión
en GPU ni ejecución de modelos veinte veces mayores que la capacidad disponible.

## Verificación del refactor — 24 de septiembre de 2026

Se ejecutaron **17 tests**, incluidos CUDA, equivalencia de caché, aislamiento
de expertos, shards, procedencia y ayuda de todos los comandos. También se
ejecutó la auditoría aleatoria de 36 casos y un prompt real con el checkpoint
4750, clasificador y caché GPU. El manifiesto reconstruido coincide con el
existente. Se comprobaron sintaxis e imports tras retirar los scripts antiguos.

Estos son controles de regresión, no una repetición del estudio científico.
Sus salidas están en `results/refactor_backup/smoke_audit/` y
`results/refactor_backup/checkpoint_chat.json`. No se alteraron los checkpoints,
el dataset ni los informes del piloto anterior.

## Predictor de caché — nuevo desarrollo de 1.4 y 1.5

Ahora existe un predictor estadístico de demanda por etiqueta actual y anterior,
con suavizado hacia popularidad global, serialización y adaptación online opcional
después de cada turno. El estudio congela el aprendizaje antes de evaluar.

Se ejecutó `cache-study` con el checkpoint 4750, GPU, clasificador real, capacidad
de 52 expertos y hasta 32 candidatos precargados. Seis sesiones train aportaron
18 turnos; tres sesiones test de tres turnos se repitieron tres veces con cinco
políticas: **135 observaciones**, pero solo **tres sesiones de evaluación distintas**.
Los casos son redactados, cubren tema estable, workflow y cambio brusco, y acumulan
prompts sin generar respuestas. No constituyen conversaciones reales ni evaluación
externa de calidad. Las listas semánticas usan la calibración anterior; popularidad
y predictor aprenden de las mismas sesiones train.

Evidencia: [resumen del estudio](../results/cache_study_04750_v1/SUMMARY.md),
[informe](../results/cache_study_04750_v1/report.json),
[turnos individuales](../results/cache_study_04750_v1/turns.jsonl) y
[predictor congelado](../results/cache_study_04750_v1/predictor.json).

| Política | Mediana por turno, ms | p95, ms | Hits | Transferencias totales, MiB |
|---|---:|---:|---:|---:|
| Todo residente | 208,67 | 243,74 | No aplica | 0 durante turnos |
| LRU | 229,23 | 273,23 | 0,0% | 2579,06 |
| Popularidad | 231,08 | 264,39 | 43,2% | 1842,75 |
| Semántica | 243,45 | 268,35 | 43,5% | 1834,88 |
| Predictor aprendido | 222,84 | 257,24 | 43,2% | 1842,75 |

Los tiempos incluyen clasificación y selección/precarga, además del forward;
excluyen carga inicial, comprobación e informes. Las transferencias suman nueve
turnos por tres repeticiones e incluyen precargas. El control residente carga sus
pesos antes de medir, por eso no tiene transferencias durante los turnos.

Todas las políticas dieron NLL 5,18495 sobre estos contextos y cero diferencias
en rutas, argmax y 64 logits muestreados por token. La NLL no es comparable con
la de las ventanas del corpus del piloto anterior.

El predictor transfirió **28,5% menos que LRU**, pero coincidió con popularidad
en bytes y aciertos. La diferencia observada de mediana no prueba una ventaja
del contexto: con esta muestra no podemos separarla de variabilidad temporal.
El 0% de LRU es compatible con recorrer sucesivamente más expertos que la
capacidad de caché; no significa que su implementación esté desactivada.
El contexto completo por turno favorece ese patrón y no representa decode
incremental. Hace falta un corpus mayor y evaluación por sesión/escenario.

Se verificaron **23 tests**, incluidos CPU y CUDA. El chat con predictor aprendido
y adaptación online ejecutó el checkpoint real y guardó una copia con 19 turnos
observados; el predictor congelado conserva sus 18 observaciones de entrenamiento.
Esto verifica integración, no demuestra mejora del aprendizaje online.

## Dos expertos enrutados por capa — objetivo de memoria y swapping

El objetivo se concreta en ejecutar modelos mayores que la memoria disponible,
sin transferencias continuas durante la respuesta. No se busca superar la velocidad
del modelo residente. Se implementó un adaptador KV incremental de la arquitectura
original y un límite de **dos expertos enrutados por capa**, también durante el
prefill serial. El router conserva sus decisiones; los expertos compartidos y
backbone permanecen activos y se contabilizan aparte.

Se repitieron las cinco políticas con el mismo checkpoint: tres sesiones test,
tres turnos, tres repeticiones y 16 tokens generados por turno. Son **405 forwards
de decode por política**; el primer token generado procede del último prefill.
El contexto acumula las respuestas del baseline y se limita a 64 tokens de prompt.
El predictor es el congelado del estudio anterior, sin volver a aprender en test.

Evidencia: [resumen de dos expertos](../results/decode_two_experts_04750_v1/SUMMARY.md),
[informe y memoria](../results/decode_two_experts_04750_v1/report.json),
[traza por token](../results/decode_two_experts_04750_v1/turns.jsonl).

| Política | Expertos enrutados GPU | Pasos decode sin cargas | H2D por paso | Hits por experto |
|---|---:|---:|---:|---:|
| Control residente | 88 | 100% | 0 | No aplica |
| LRU por capa | Máximo 22 | 0% | 15,419 MiB | 46,6% |
| Popularidad | Máximo 22 | 0% | 15,419 MiB | 46,6% |
| Semántica | Máximo 22 | 0% | 15,419 MiB | 46,6% |
| Predictor aprendido | Máximo 22 | 0% | 15,419 MiB | 46,6% |

Las 11 capas MoE respetaron el máximo de dos. Los pesos enrutados residentes
pasaron de **115,5 a 28,875 MiB**, un **75% menos de pesos enrutados**, no de VRAM
total. Se conservaron las respuestas greedy, todas las rutas y los logits
muestreados, sin diferencias frente al control incremental. El adaptador también
se contrastó con el forward original en FP32 sobre todo el vocabulario de un
prefijo de nueve tokens: diferencia máxima aproximada 7,63e-6.

**La reducción de residencia funciona; evitar swapping por token no está resuelto.**
Con dos plazas y dos selecciones activas, tras procesar un token solo puede quedar
su pareja en cada capa. Si la pareja siguiente cambia, se debe cargar lo que
falta. La precarga de contexto ocurre antes del prefill y no crea plazas extra
durante decode; por eso estas cuatro políticas terminan con igual tráfico en la
generación. Un hit rate del 46,6% no significa que haya tokens completos sin cargas.

El prefill también transfiere pesos y se informa aparte; no se ha escondido su
coste. Las latencias son orientativas: ejecución de escritorio sin aislamiento
de actividad CPU/I/O, no un ensayo para afirmar aceleración. En este primer
informe, `peak_allocated_bytes` puede incluir fases anteriores porque no se
reinició por sesión; usar residencia de parámetros y trazas para el límite de
expertos. El código posterior reinicia el contador CUDA por sesión para nuevos
ensayos. La conclusión sobre transferencias no depende de ese contador.

## Desarrollo siguiente: disco → RAM limitada → GPU

Se añadió exportación sin compresión por experto, carga inicial en `meta` y una
caché RAM LRU. Al arrancar se carga el backbone y se dejan marcadores vacíos para
los expertos, cuyos pesos se leen de archivos bajo demanda. La inferencia desde
el almacén no necesita cargar el checkpoint original ni todos sus expertos en RAM.

Se exportaron los 88 expertos del checkpoint 4750. En una prueba real con
**4 MiB de presupuesto RAM para expertos** y dos por capa en GPU:

- Se retuvieron como máximo **3,9375 MiB** de tensores de expertos en RAM.
- Se mantuvieron **28,875 MiB** de pesos enrutados en GPU.
- La salida de 16 tokens y los contadores de demanda coincidieron con el modo
  de respaldo RAM completo para el mismo prompt.
- Hubo **273 lecturas de shards** durante prefill y generación; el presupuesto
  pequeño demuestra el límite, pero no es una configuración recomendada.

Evidencia: [ejecución desde disco](../results/expert_store_04750/chat_disk_check.json)
y [referencia RAM](../results/expert_store_04750/chat_ram_reference.json).
Los bytes leídos son lógicos: el sistema operativo puede servirlos desde su
caché de páginas. El presupuesto tampoco incluye clasificador, backbone,
temporales ni RSS del proceso. Los shards no están cuantizados.

Pasaron **29 tests**, incluyendo integridad, carga sin abrir expertos al arrancar,
límites RAM, residencia física de parámetros en CUDA y equivalencia incremental.
Este desarrollo resuelve el mecanismo de carga por niveles del prototipo, no
demuestra todavía ejecución práctica de un DeepSeek oficial enorme.

El siguiente estudio debe contrastar el coste de conservar expertos adicionales
comprimidos frente a fijar una pareja por contexto. Fijar la pareja cambia el
router y exige medir degradación de calidad; no debe presentarse como caché exacta.
La predicción para la RAM también necesita evaluación con un corpus mayor.
