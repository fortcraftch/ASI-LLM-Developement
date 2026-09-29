# Fine-tuning de recuperación con expertos fijos

## Protocolo actual v2: ajustar pesos bajo varias selecciones

La CLI permite `--adaptation router`, `expert-lora` o `router-expert-lora`.
LoRA aprende una corrección de bajo rango de la proyección de salida de cada
experto: el peso efectivo pasa de W a W + B×A. Esto modifica su transformación,
aunque el checkpoint base permanece congelado. No es entrenamiento completo de
los 7B parámetros ni un adaptador de atención.

Se revisó el proyecto vecino `Creating-GPT-2-From-0/finetune_oasst.py`: su objetivo
supervisado puntúa tokens del asistente y EOS. El modo `--supervised` aplica aquí
esa idea: pregunta enmascarada en la pérdida, respuesta y EOS entrenables. No usa
probabilidades del profesor ni mide KL; el flujo de distilación sigue disponible
sin esa opción. Recuperar sintaxis no demuestra que el conocimiento permanezca
accesible o que las respuestas sean correctas.

`configs/recovery_language_v2.json` amplía el corpus a **60 preguntas originales
sintéticas: 36 train, 12 dev y 12 test**, en las tres categorías disponibles.
Incluye tareas explicativas y algunas respuestas numéricas exactas. Sigue siendo
un piloto, no un benchmark externo ni una muestra representativa de cada dominio.

### Cómo proteger el ajuste al cambiar de selección

- Calibrar solo con train, usando preguntas y respuestas completas; congelar
  después los IDs de expertos por capa y categoría.
- Entrenar las parejas ejemplo/selección de forma equilibrada. Con 36 ejemplos,
  dos selecciones y 144 pasos, cada pareja se visita dos veces.
- Mantener **un ajuste por identidad capa/experto**, compartido por las categorías
  donde participa. Los parámetros LoRA y el optimizador sobreviven a las expulsiones
  de la caché del experto base; no se crean copias divergentes por categoría.
- Elegir por el **peor cociente entre pérdida dev actual e inicial** entre todos
  los grupos categoría/selección. El paso cero sigue siendo candidato: una mejora
  media no puede ocultar una regresión media en otra categoría o tamaño.
- Evaluar el retorno A→B→A y verificar ausencia de cargas dentro de cada respuesta.
  El checkpoint vincula fuente, corpus y banco de selecciones mediante hashes.

Esto valida únicamente las selecciones declaradas, inicialmente **16 y 32
residentes por capa con 8 activos**. Otra combinación de expertos no queda cubierta
por compartir el mismo tamaño. Hay que entrenarla o validarla antes de desplegarla.
La recuperación también puede sobreajustarse a estos pocos ejemplos de dev.

Los pequeños parámetros LoRA de la unión de expertos entrenables permanecen en
GPU, incluso cuando sus expertos base no son residentes. Su coste se registra en
`adapter_parameter_bytes` y `cache_final.adapter_bytes`; no debe confundirse con
el contador de expertos base residentes. No hay swapping por token de adaptadores.
El rango por defecto es 8 y su factor de escala es 1 (alpha LoRA igual al rango).

### Comandos de la comparación v2

Usar directorios nuevos para otro ensayo. Calibración de cuatro ejemplos train
por categoría, incluyendo respuestas y EOS:

```bash
python -m asi recovery calibrate --examples configs/recovery_language_v2.json --mask-sizes 32 16 --calibration-per-category 4 --max-length 128 --output results/recovery_calibration_v2
python -m asi recovery train --supervised --adaptation router --examples configs/recovery_language_v2.json --study results/recovery_calibration_v2/report.json --mask-sizes 32 16 --max-length 128 --steps 144 --eval-interval 36 --generation-per-category 2 --output results/recovery_language_router_v2
python -m asi recovery train --supervised --adaptation router-expert-lora --rank 8 --examples configs/recovery_language_v2.json --study results/recovery_calibration_v2/report.json --mask-sizes 32 16 --max-length 128 --steps 144 --eval-interval 36 --generation-per-category 2 --output results/recovery_language_lora_v2
python -m asi recovery evaluate --checkpoint results/recovery_language_lora_v2/adaptation_best.pt --examples configs/recovery_language_v2.json --study results/recovery_calibration_v2/report.json --max-length 128 --generation-per-category 2 --output results/recovery_language_reloaded_v2
python -m asi graphs --results results/recovery_language_lora_v2 --output results/recovery_language_graphs_v2
python -m asi recovery compare --reports results/recovery_language_router_v2/report.json results/recovery_language_lora_v2/report.json --output results/recovery_language_comparison_v2
python -m asi graphs --results results/recovery_language_comparison_v2 --output results/recovery_language_comparison_graphs_v2
```

El checkpoint recupera automáticamente modo, rango y banco de tamaños al evaluar.
Con LoRA se guardan `adaptation_best.pt` y `adaptation_last.pt`, que contienen
routers, correcciones de expertos y metadatos. Se aplican mediante `recovery evaluate`
sobre el checkpoint original; no son una exportación completa para cargar directamente
con `transformers.from_pretrained`.
`evaluate --skip-generation` permite verificar pérdidas y cambios de selección
tras recargar sin volver a producir los mismos textos.
La comparación conserva datos, selecciones, tasa de aprendizaje y número de pasos;
LoRA añade parámetros y cómputo, cuyos costes se miden. No cambiar hiperparámetros
según test. El nuevo corpus y objetivo impiden atribuir diferencias con v1 a una
sola causa: la comparación controlada es entre las dos ejecuciones **v2**.

En v2, `test_before` y `test_after` separan resultados por selección, y cada registro
incluye categoría. `dev_best_worst_group_ratio` contiene el criterio de selección.
`generations` añade repetición de trigramas y coincidencia exacta solo cuando la
pregunta exige una respuesta literal. EOS detiene la generación; menos repeticiones
puede deberse a respuestas demasiado cortas y no implica más corrección.
`transition_checks` verifica la pérdida al volver a una categoría tras otra.
Cada pregunta comienza sin KV previo: este control no demuestra estabilidad de
una conversación larga que conserve historial al cambiar de categoría.
Se generan los dos primeros ejemplos test de cada categoría, fijados por el corpus;
todos los ejemplos test se puntúan con teacher forcing.

### Resultados reales v2 — 28–29/09/2026

Ambos entrenamientos terminaron en la RTX 3060 de 12 GB: 144 pasos, misma semilla,
datos, selecciones y exposición. Dev eligió el paso 144 para routers y el 72 para
routers + LoRA; seguir entrenando LoRA empeoró su criterio de validación.

| Expertos residentes por capa (8 activos) | CE sin ajuste | CE routers | CE routers + LoRA |
| --- | ---: | ---: | ---: |
| 16 | 7,3730 | 5,8888 | 4,3605 |
| 32 | 4,9507 | 3,3839 | 2,5748 |

CE mide el error al predecir las respuestas de referencia: menor es mejor, pero
no equivale a exactitud de respuestas libres. Son 12 ejemplos test por selección.
El punto de partida es el modelo **ya restringido**, no el routing nativo.

En las seis generaciones test por selección, la fracción media de trigramas
repetidos con 32 residentes pasa de 8,33% sin ajuste a 4,61% con routers y 0,57%
con LoRA. Con 16 pasa de 41,67% a 22,22% y 26,19% respectivamente: **LoRA mejora
CE pero repite más que el control de routers**. Persisten errores de contenido y
redacción. Con 32, LoRA responde correctamente «4» a la longitud de una lista y
da 54 para 9×6, aunque en este segundo caso incumple el formato de solo un número.
Su coincidencia literal es 1/3 con 32 y 0/3 con 16; no es exactitud semántica.

Todas las generaciones registran cero cargas de expertos durante la respuesta.
El retorno A→B→A conserva exactamente la pérdida en ambas selecciones. Esto no
valida selecciones arbitrarias ni conversaciones con KV persistente.

LoRA añade 16.293.888 parámetros (62,16 MiB FP32) a los 2.097.152 de routers.
El pico de memoria GPU asignada durante entrenamiento/validación fue 7,29 GiB
frente a 7,07 GiB del control. Los bucles de entrenamiento con validación duraron
23,62 y 22,18 minutos respectivamente, excluyendo evaluaciones iniciales/finales;
no son un benchmark de latencia de inferencia.

Informes: `results/recovery_language_router_v2/report.json`,
`results/recovery_language_lora_v2/report.json` y
`results/recovery_language_comparison_v2/report.json`.
[Gráficas comparativas](../results/recovery_language_comparison_graphs_v2/index.html)
y [curva de validación LoRA](../results/recovery_language_graphs_v2/index.html).

La evaluación independiente del checkpoint recargado terminó correctamente el
29/09/2026 (`results/recovery_language_reloaded_v2/report.json`): todos los registros
de pérdidas test antes/después coinciden exactamente con el entrenamiento, y
A→B→A mantiene diferencia cero. Se ejecutó con `--skip-generation`, sin repetir
los textos. La batería completa pasó 94 pruebas, incluyendo gradientes LoRA,
congelación del modelo base, expulsión/recarga y compatibilidad del checkpoint.

Conclusión: recuperación parcial, más prometedora con 32 residentes. Una semilla
y 60 ejemplos sintéticos no permiten demostrar recuperación general ni conservación
del conocimiento. El siguiente estudio requiere datos representativos, varias
semillas, referencia nativa sobre el mismo corpus y evaluación independiente de
contenido; no reajustar este ensayo usando sus resultados test.

## Protocolo inicial v1 y resultados históricos

Este experimento corresponde a los pasos **2.2 y 2.3 del TFG**: comprobar si un
modelo público puede recuperar calidad después de restringir sus expertos.
`asi/experiments/recovery.py` implementa el entrenador, la preparación del profesor
y la comparación final. `configs/recovery_pilot_v1.json` define los ejemplos;
`tests/test_recovery.py` verifica gradientes, congelación y recarga de parámetros.

## Qué aprende

El alumno es OLMoE con **32 expertos residentes de 64 por capa**, seleccionados por
categoría mediante el estudio de routing previo. En cada token el router elige
**8 de esos 32** y calcula sus pesos. Los residentes permanecen fijos durante
la respuesta; cambiar de categoría sí puede requerir transferencias.

Solo se entrenan los routers: **2.097.152 parámetros**. Los expertos, atención,
embeddings y demás pesos permanecen congelados. Sus pesos originales no se
sobrescriben. El router mantiene parámetros FP32 y el cálculo usa autocast BF16.
En aquel piloto no se entrenaron adaptadores LoRA ni matrices densas de expertos.

El profesor es el mismo checkpoint con routing original, sin restricción temática.
Primero se guardan sus probabilidades en disco, ejecutándolo con respaldo en disco
y caché RAM limitada. Profesor y alumno no necesitan coexistir en GPU ni en RAM.

Para cada pregunta y respuesta de referencia se minimiza:

`pérdida = alpha × CE(respuesta) + (1 − alpha) × KL(profesor || alumno)`

CE mide cuánto favorece el alumno las palabras de la respuesta escrita. KL mide
cuánto difiere su distribución de la del modelo original en esas mismas posiciones.
Se calcula únicamente sobre la respuesta, dando al modelo los tokens anteriores
correctos (*teacher forcing*). Por defecto alpha es 0,5. Una reducción de estas
pérdidas no garantiza respuestas libres coherentes: también se guardan sus textos.

## Ejecutar el piloto

Requiere el checkpoint público descargado y el estudio de routing completado.
Usa la misma venv que el resto del proyecto y directorios de salida nuevos:

```bash
python -m asi recovery prepare --output results/recovery_prepare_v1
python -m asi recovery train --output results/recovery_router_v1
python -m asi recovery evaluate --checkpoint results/recovery_router_v1/router_best.pt --output results/recovery_reloaded_v1
```

La preparación crea `results/recovery_teacher_v1/manifest.json` y un archivo por
ejemplo. Valida hashes del modelo, corpus, selección de expertos y archivos del
profesor. Una preparación interrumpida puede reanudarse con `--resume-teacher`
y un directorio `--output` nuevo si existe su manifiesto parcial verificado.

Valores iniciales: 24 pasos, batch de un ejemplo, learning rate 0,0001, validación
cada 6 pasos, longitud máxima 96 tokens y 32 tokens de generación. `--experts`,
`--active`, `--examples`, `--teacher-dir`, `--steps` y `--lr` permiten otros ensayos;
K debe tener un mapeo ya calibrado en `--study`. La identidad del profesor incluye
estas restricciones y el corpus: otro experimento incompatible requiere prepararlo
en un directorio distinto. No se truncan ejemplos demasiado largos.

La RAM se adapta a la disponibilidad con una caché de expertos de hasta 256 MiB.
`--ram-gib` limita RSS; no reserva ni exige esa cantidad. Se verifica capacidad
GPU antes de cargar residentes, aunque la memoria de activaciones del entrenamiento
depende de la longitud y puede exigir reducirla. Este flujo no reanuda el optimizador.

## Separación de datos y lectura del informe

El piloto contiene 12 ejemplos sintéticos originales de programación, matemáticas
y biología: **6 train, 3 dev y 3 test**. El optimizador solo usa train. En v1, dev elegía
el checkpoint con menor pérdida, incluyendo los routers originales del paso cero.
Test se consulta al final para comparar la misma restricción antes y después.
Las categorías se proporcionan explícitamente: aquí no se mide el clasificador.

`report.json` incluye:

- `dev_before`, `dev_checks`, `selected_step`: elección del ajuste sin usar test.
- `teacher_test_ce`: CE del modelo original sobre las respuestas reservadas.
- `test_before`, `test_after`: CE, KL y pérdida del alumno en esos ejemplos.
- `generations`: textos libres antes/después, tiempos y contadores de caché.
- `peak_cuda_bytes`: pico del bucle de entrenamiento y sus validaciones.
- `training_with_validation_seconds`: tiempo de ese bucle, incluidos cambios de
  categoría; excluye preparación del profesor y comparación final.
- `cache_final`, RSS e identidad de los artefactos para revisar memoria y procedencia.

`router_best.pt` contiene solo routers seleccionados y metadatos;
`router_last.pt` conserva el último paso, aunque no sea el mejor. La generación
usa una longitud fija y **no se detiene en EOS**; puede mostrar tokens especiales
o continuación después del fin de respuesta. Las respuestas de referencia no
incluyen un objetivo EOS. El piloto estudia recuperación local, no entrena todavía
el comportamiento completo de finalización de un asistente.

Los contadores antes/después permiten distinguir las precargas entre categorías
de las cargas dentro de una respuesta. El programa aborta si encuentra una carga
de expertos durante el forward de evaluación, entrenamiento o generación fija.

Para visualizar la comparación final y la curva de validación, sin cargar modelos:

```bash
python -m asi graphs --results results/recovery_router_v1 --output results/recovery_graphs_v1
```

Las medias de CE/KL dan el mismo peso a cada ejemplo, aunque sus respuestas tengan
distinta longitud. Los gráficos separan el modelo original de la restricción sin
ajuste y con ajuste.

## Qué puede concluirse

Este piloto comprueba el mecanismo y busca una señal inicial de recuperación.
Doce ejemplos, una semilla y respuestas cortas no bastan para aprobar o descartar
la categorización de modelos públicos. Tampoco son comparables directamente con
la NLL del estudio previo, que puntuaba otras continuaciones y otros prompts.

Una investigación posterior requiere un corpus independiente más amplio,
calibración representativa de expertos, varios K/A y seeds, controles de igual
presupuesto de adaptación, tareas con corrección verificable y conversaciones.
LoRA persistente se investiga en v2, sin asumir de antemano que recupere calidad.
La selección temática de cada ensayo se mantiene congelada para aislar el
efecto de este entrenamiento.

## Resultado local del primer piloto

Ejecutado en RTX 3060 de 12 GiB con el checkpoint público OLMoE, K=32/A=8,
24 pasos y los 12 ejemplos descritos. El paso 18 gana en dev: pérdida combinada
2,5573 antes y 2,3670 después; el paso 24 sube a 2,3896.

| Variante | CE en test | KL respecto al original |
|---|---:|---:|
| Routing original, profesor | 2,9791 | 0 por definición |
| Residentes fijos sin ajuste | 4,2515 | 1,7479 |
| Mismos residentes, routers ajustados | 3,6830 | 1,5143 |

CE y KL mejoran en cada uno de los tres ejemplos test, pero las generaciones
de 32 tokens **siguen mostrando repeticiones y errores**. Por ejemplo, tras el
ajuste la pregunta sobre la mediana continúa con «is is is…». No se ha recuperado
una capacidad conversacional fiable ni se ha demostrado generalización amplia.

Las seis generaciones antes/después mantuvieron **512 instancias residentes**
(32 × 16 capas), cero cargas, cero bytes H2D de expertos y cero lecturas lógicas
de expertos durante la generación. Las precargas entre categorías sí transfieren
pesos. El pico asignado por PyTorch durante entrenamiento/validación fue
**7,085 GiB**; RSS al terminar, **2,135 GiB** (instantánea, no pico). El bucle de
24 pasos con sus validaciones tardó **66,12 s**, excluyendo preparación del
profesor, validación inicial y comparación final. No es un benchmark de velocidad.

Artefactos locales: [informe completo](../results/recovery_router_v1/report.json),
[routers seleccionados](../results/recovery_router_v1/router_best.pt) y
[gráficas](../results/recovery_graphs_v1/index.html).

La evaluación en un proceso nuevo con `router_best.pt` reprodujo exactamente
las métricas test y los tokens de las seis generaciones. Su informe está en
`results/recovery_reloaded_v1/report.json`. El checkpoint de routers ocupa unos
8 MiB. Pasaron los 89 tests del proyecto, incluidos backward BF16 con parámetros
FP32, congelación de pesos y guardado/recarga del ajuste.

El resultado justifica comprobar la hipótesis con más datos y una selección de
expertos mejor calibrada; no justifica escalar todavía a un modelo enorme ni
atribuir el resultado a un fine-tuning completo. Los expertos y backbone no se
han adaptado, y el corpus sigue siendo un piloto técnico muy pequeño.
