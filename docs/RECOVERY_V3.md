# Recuperación ampliada: datos públicos, profesor y tres ajustes

**Actualización 07/10/2026:** terminó el piloto de 36 referencias verificables,
los tres ajustes con 32 y los tres calendarios 32→16. Se conservaron los archivos
antiguos. La pérdida mejora, pero la revisión con hasta 256 tokens encuentra
errores de tarea y repeticiones; no se declara recuperación general. Se mantiene
pospuesta la ampliación a 4.093 referencias. Resultados, gráficas y calibración
de memoria están en [CONTINUACION_9_13.md](CONTINUACION_9_13.md).
El estado descrito a continuación al 05/10 es histórico, previo a esta ejecución.

La comparación de textos completos del original y las recuperaciones de 32 y 16,
con tiempos, memoria y transferencias, se reproduce con
[RECOVERY_GENERATION.md](RECOVERY_GENERATION.md).
El [análisis de las repeticiones](RECOVERY_REPETITION_ANALYSIS.md) separa los
bucles observados de las hipótesis sobre routing, decodificación y entrenamiento.

## Protocolo siguiente: recuperación controlada (11/10/2026)

Este apartado fija la siguiente comparación de desarrollo; no describe una
nueva ejecución. La hoja de ruta vigente es [ROADMAP_PRINCIPAL.md](ROADMAP_PRINCIPAL.md).
El piloto de escala ya terminó: cambiar únicamente la masa seleccionada no
recuperó el lenguaje. Se conserva **la escala original restringida**, 32 residentes
por capa y ocho activos, con la misma máscara en todos los ajustes.

Primera lectura completada el 12/10: las **240/240 referencias** están anotadas
con huellas en `results/recovery_reference_review_20261011/report.json`.
Lectura no significa aprobación: faltan verificaciones factuales, resolución de
ambigüedades y pruebas funcionales de código. `ready_for_training=false`.
Se han propuesto por separado dos respuestas matemáticas corregidas y tres de
biología contrastadas con fuentes. Una pregunta de plantas tenía una explicación
sobre estrellas; otras contienen imágenes ausentes o explicaciones truncadas.
Los defectos no se atribuyen causalmente a la degeneración del modelo sin un
experimento controlado. Corpus, profesor y resultados anteriores permanecen intactos.
Toda revisión conserva el ID padre y su exposición; no convierte datos usados en
test nuevo. Cambiar texto exige retokenizar y recalcular profesor si se usa KL.

### Comprobación funcional de referencias de código

Preparado `python -m asi reference-code-check`: ocho referencias con aserciones
escritas después de la revisión (seis diagnósticos y dos controles positivos).
Es una comprobación de referencias seleccionadas, no un benchmark ni una prueba
exhaustiva de todo MBPP. Los casos y respuestas están fijados por SHA-256.

```bash
python -m asi reference-code-check --dry-run --output results/reference_cases_prepared.json
# En un equipo que ya tenga Docker Linux y una imagen Python local:
python -m asi reference-code-check --image python:3.11-slim --output results/reference_cases_executed.json
```

El ejecutor resuelve la etiqueta a un ID inmutable de imagen local; no descarga
imágenes. Cada referencia se ejecuta como usuario sin privilegios, sin red,
sin carpetas del host montadas y con raíz de solo lectura. Límites: 256 MiB RAM,
sin swap adicional, una CPU, 32 procesos, tmpfs de 16 MiB y 10 segundos por caso
(incluido arranque). Descarta stdout/stderr del código para no llenar memoria o
disco del host. Comprueba la retirada del contenedor incluso tras un timeout.
Las aserciones se evalúan hasta el primer fallo de cada referencia. No es un
juez robusto frente a código que intente falsificar deliberadamente el resultado.

Estado local del 12/10: preparación de ocho referencias y cinco pruebas unitarias
del ejecutor aprobadas; **cero referencias ejecutadas**. Docker no está disponible
en esta sesión y WSL deniega la enumeración. El intento deja informe explícito
`code_check_environment.json` en la carpeta de revisión y no ejecuta código en el
host como alternativa. Las pruebas unitarias usan simulación de subprocess;
no certifican el aislamiento de un contenedor real. Falta ejecutar y verificar
el recorrido completo en un host Linux con Docker antes de usar sus métricas.

### Versión derivada en preparación

`results/recovery_reference_review_20261011/training_draft_v1.json` contiene las
240 referencias de entrenamiento, con **17 correcciones propuestas aplicadas solo
a esta copia**: dos explicaciones matemáticas, doce aclaraciones/ajustes de
matemáticas y tres respuestas de biología. Las demás observaciones no están
resueltas por crear el archivo. No es el conjunto definitivo ni contiene dev/test.

El comando reutilizable `python -m asi data recovery-revise` valida las huellas
anteriores, rechaza padres desconocidos o repetidos, conserva `parent_id` y
`train_only`, y elimina conteos de tokens obsoletos cuando cambia texto. El lector
de recuperación rechaza `ready_for_training=false` y referencias train-only
reclasificadas como dev/test. Los conjuntos históricos sin este nuevo campo
conservan su comportamiento anterior.

```bash
python -m asi data recovery-revise --corpus results/recovery_corpus_v3/corpus.json --review results/recovery_reference_review_20261011/report.json --replacements results/recovery_reference_review_20261011/answer_replacements_20261012.json results/recovery_reference_review_20261011/biology_replacements_20261012.json results/recovery_reference_review_20261011/math_clarifications_20261012.json --output results/recovery_reference_review_20261011/training_draft_next.json
```

Comprobación de tokenización local: 240/240 caben en 384 tokens, máximo 275,
con EOS incluido. Informe `draft_v1_tokenization.json`, con huellas del borrador,
tokenizador y secuencias. Esto no demuestra corrección factual ni funcional y no
activa entrenamiento. Se mantienen pendientes el cierre de referencias, la
comprobación funcional y los conjuntos de desarrollo/test.

Verificación adicional del entorno: la consulta WSL fuera de la restricción de
sesión confirma que **no hay distribuciones instaladas**. El acceso denegado
inicial no era evidencia de una distribución parada; no hay un entorno Linux
local listo para ejecutar los contenedores.

### Pregunta y comparación

¿Adaptar las salidas de los expertos mejora respuestas completas frente a
adaptar solo el router, sin cargar otros expertos durante la respuesta?

- Controles: original sin adaptación y restringido sin adaptación.
- Tres ajustes: router, LoRA de `down_proj` de expertos, router + esa misma LoRA.
- Primera comparación con CE, rango 8, 200 actualizaciones, mismo entrenamiento,
  orden, semilla 1337 y puntos de validación 0/50/100/150/200. Los parámetros
  entrenables difieren por diseño; registrar su número, memoria y tiempo.
- Congelar backbone, atención, embeddings, normas y cabeza de vocabulario;
  también los pesos base de los expertos. No ampliar proyecciones a la vez.
- Guardar identidad del modelo, tokens y máscaras, estado inicial de los
  adaptadores y orden real de lotes. El router debe partir de los mismos pesos;
  los brazos que contienen LoRA deben compartir su inicialización.
- La destilación CE+KL será una comparación posterior con el mismo ajuste y
  datos, si la primera demuestra utilidad. Así no se mezclan el efecto de los
  pesos adaptados y el del profesor en una única comparación.

No reutilizar los 424 shards de procedencia no certificada. Los 276 del piloto
posterior solo pueden usarse después de verificar que corresponden exactamente
al conjunto y tokenización elegidos; no son un sustituto del profesor para otros
inputs. No ampliar automáticamente la generación a las 4.093 referencias.

### Datos y evaluación antes de ejecutar

1. Revisar las 240 referencias de entrenamiento del piloto anterior, conservando
   procedencia y excluyendo respuestas defectuosas. Si cambian, versionar el
   conjunto y regenerar las probabilidades afectadas antes de cualquier KL.
2. Objetivo de desarrollo: 90 preguntas, 30 por categoría (programación,
   matemáticas, biología), separadas por familias semánticas y plantillas del
   entrenamiento. Son un objetivo de preparación, **no un conjunto congelado**.
3. Objetivo confirmatorio: 300 preguntas nuevas, 100 por categoría, sin consultar
   sus respuestas generadas al seleccionar variantes. Los 128 candidatos del
   inventario anterior no bastan y no están certificados. No reducir el test ni
   reciclar ejemplos inspeccionados para cumplir artificialmente este objetivo.
4. Revisar temas y referencias por su significado, además de IDs, duplicados y
   plantillas. Mantener en cuarentena los tres problemas identificados. La
   incertidumbre de preentrenamiento del modelo público se declara siempre.
5. Publicar el manifiesto con hashes, referencias, familias, rúbricas y separación
   antes de lanzar los ajustes. El inventario corrupto sigue incompleto: ningún
   plan debe presentarse como test independiente por pasar solo un filtro de IDs.

La métrica principal será **respuesta útil completa**: solución central correcta,
explicación válida cuando se pide, cumplimiento de instrucciones esenciales y
sin contradicciones o degeneración que invaliden la tarea. Formato cosmético,
EOS, repetición y CE se informan aparte. No exigir explicación a prompts que
solicitan solo código o una respuesta breve.

- Matemáticas: respuesta numérica y validez de los pasos, por separado.
- Biología: conceptos y explicación correctos; una palabra clave no basta.
- Código: pruebas funcionales solo en un entorno aislado con límites de tiempo,
  memoria y red. La revisión estática no equivale a superar esas pruebas.
- Anotación: ocultar nombres de brazos/checkpoints y aleatorizar el orden;
  registrar hash del texto, rúbrica, autor y desacuerdos. Si solo anota el
  asistente, el resultado se etiqueta exploratorio; no evaluación humana ciega.

Generación primaria greedy, EOS o 256 tokens, mismo prompt/tokenizador. Registrar
truncamiento; cualquier repetición a 512 tokens será una prueba secundaria
predefinida para todos los brazos, no una corrección selectiva de respuestas.

### Criterios de avance fijados para este próximo desarrollo

Seleccionar por proporción macro de respuestas útiles en las tres categorías;
en empate, menor CE de desarrollo y después checkpoint más temprano. No usar
el test para resolver empates ni elegir checkpoints. Registrar también paso cero.

Pasar a repetición con semillas 2027 y 3407 solo si el mejor ajuste supera al
restringido sin adaptar en **al menos 10 puntos porcentuales macro**, mejora en
al menos dos categorías y no pierde más de 10 puntos en ninguna. Son umbrales
de decisión de ingeniería para este desarrollo, no una prueba de significación
ni un criterio elegido con resultados de este nuevo experimento.

Antes del test, fijar una única variante y regla de checkpoint. Mostrar intervalos
pareados al 95% por remuestreo de familias de preguntas (no tokens ni respuestas
repetidas), categorías y semillas por separado. No sumar las semillas como si
fueran nuevas preguntas independientes. El objetivo de 300 preguntas no garantiza
potencia para demostrar equivalencia: depende de la discordancia y las familias.

Solo llamar «recuperación parcial» a una mejora de utilidad frente al restringido
con límite inferior del intervalo por encima de cero. Para sostener calidad
cercana al original, fijamos un margen práctico de 5 puntos macro: el límite
inferior de recuperado menos original debe superar −5 puntos; además, publicar
cada categoría y el resultado absoluto del original. Un intervalo inconcluso no
es equivalencia. No cambiar el margen después de ver el test.

En todos los casos se exige residencia de 32 por capa, ocho activos, cero cargas
de expertos durante la respuesta y presupuesto de memoria respetado. Medir la
preparación aparte. Cualquier incumplimiento invalida la afirmación de recuperación
bajo esa restricción aunque la respuesta mejore.

Solo después de esa validación de 32 se prepara 32→16 con controles propios.
Si baja CE sin mejorar utilidad, detener esta ampliación y estudiar datos,
máscaras y cooperación antes de aumentar pesos o duración. Las categorías nuevas
son generalización adicional; no se infiere cobertura universal de estas tres.

**Estado:** protocolo escrito; datos, revisión independiente, aislamiento de código
y ejecución aún pendientes. No hay un nuevo entrenamiento iniciado por este plan.

## Explicación sencilla y estado al 05/10/2026

El profesor es el **OLMoE original**, con su selección nativa entre todos los expertos.
El alumno parte de ese mismo modelo, pero solo puede elegir dentro de 32 expertos
residentes por capa (activa ocho por token). No entrenamos varios profesores nuevos:
guardamos referencias del original para no tener que cargarlo junto al alumno.

Para cada ejemplo, el profesor calcula probabilidades de los próximos tokens.
Los archivos `.pt` guardan esas distribuciones y sus metadatos; no son checkpoints
de miles de modelos distintos. El alumno combina dos objetivos: aprender la respuesta
de referencia y acercarse a la distribución del profesor bajo la restricción.
Esta transferencia de comportamiento se llama destilación.

Compararemos, con los mismos ejemplos, ajustar solo el router, solo LoRA de los
expertos, o ambos. LoRA adapta matrices de los expertos con menos pesos entrenables;
esta etapa no actualiza todas sus matrices completas. Después evaluaremos respuestas
y memoria frente al original, y más adelante la transición de 32 a 16 residentes.

Estado local comprobado: siguen existiendo 424 archivos del profesor de 4093 previstos;
el plan reconstruido solo tiene `plan.json`, sin una nueva ejecución teacher. El
manifiesto antiguo está dañado y sus shards no contienen identidad verificable del
profesor. Las correcciones de guardado atómico y procedencia están implementadas,
pero el conjunto completo y la comparación V3 multisemilla **no están terminados**.
No confundir este trabajo con el SFT OASST de 200 pasos del modelo categorizado de
124M: ese ajuste sí se ha ejecutado y no utiliza al profesor OLMoE.

## Auditoría tras la interrupción — 04/10/2026

`results/recovery_study_v3/teacher` contiene **424 de 4.093 referencias (10,36%)**, 7.074.131.776 bytes (6,59 GiB). Las 424 superan la comprobación de texto, tokens, EOS, dimensiones, valores finitos y normalización. El manifiesto JSON contiene bytes nulos y no es legible. Como los archivos antiguos no incluían la identidad del profesor, esta auditoría verifica su contenido estructural, **no permite certificar su procedencia ni reanudar de forma estrictamente verificable**. No se han borrado ni modificado.

Informe: `results/recovery_study_v3/teacher_audit_20261004.json`. Para repetir la auditoría con un archivo de salida nuevo:

```bash
python -m asi teacher-audit --teacher-dir results/recovery_study_v3/teacher --examples results/recovery_corpus_v3/corpus.json --model-dir public_models/olmoe --output results/teacher_audit_new.json
```

Un manifiesto histórico intacto permitiría verificar los hashes y aprovechar las referencias. Sin él, la opción rigurosa es regenerarlas o recalcular el profesor para verificar cada archivo; una comprobación de unas pocas muestras no autentica los 424. No se ha reconstruido un manifiesto ficticio.

Está preparado un plan equivalente en una carpeta nueva, preservando todo lo anterior:

```bash
python -m asi recovery-study run --plan results/recovery_study_v3_rebuild_20261004/plan.json --stage teacher
python -m asi recovery-study run --plan results/recovery_study_v3_rebuild_20261004/plan.json --stage native
```

Después, ejecutar `router_1337`, `expert-lora_1337` y `router-expert-lora_1337` sobre ese mismo plan, y las otras dos semillas cuando se compruebe el proceso. El plan conserva 32 residentes y top-8. La regeneración larga no se ha lanzado durante la evaluación del modelo pequeño. El total estimado de referencias sigue siendo unos 66 GiB; al preservar los archivos antiguos hay que añadir sus 6,59 GiB.

Los nuevos shards incluyen `teacher_identity`. Shards y manifiesto se guardan mediante temporal, flush/fsync y reemplazo atómico. Esto reduce el riesgo de corrupción por interrupción; no sustituye copias de seguridad. Al reanudar la etapa teacher desde un manifiesto válido, su informe incompleto se conserva en `teacher_report.attemptNNNN` y se crea un informe nuevo en la ruta del plan. El entrenamiento de recuperación aún no tiene reanudación de optimizador como `asi train`.

## Verificación histórica del 29/09

Estado verificado el 29/09/2026: corpus construido, calibración terminada con 24
ejemplos train y cuatro referencias reales del profesor guardadas/verificadas.
Estas cuatro referencias ocuparon 74,47 MiB y su pasada agrupada tardó 70,71 s
en la RTX 3060 con caché RAM de 256 MiB, sin incluir carga inicial del modelo.
No es una estimación fiable del tiempo en homelab con una caché RAM mayor.
Los nueve entrenamientos ampliados, el control nativo completo y la revisión
humana siguen pendientes; el perfil parcial no es un resultado de recuperación.

La batería pasó **110 tests**. Dos actualizaciones técnicas de router + LoRA en
la RTX 3060 también terminaron: una con CE+KL real (169 tokens, pico 7,59 GiB) y
otra con CE sobre el ejemplo train más largo (384 tokens, pico 8,53 GiB). Ambas
mantuvieron cero cargas de expertos dentro del paso. Los pesos se descartaron;
no son resultados de calidad ni garantía de memoria para toda ejecución. Registro:
`results/recovery_v3_gpu_probe.json`.

Continuación de los pasos 2.2 y 2.3 del TFG. El piloto v2 mostró recuperación
parcial con 32 residentes; **este protocolo no contiene todavía resultados de los
entrenamientos ampliados**. El objetivo es averiguar qué se recupera y a qué coste,
sin aumentar el rango de LoRA ni cambiar su proyección de salida.

## Datos y separación

`python -m asi data recovery-corpus` construye el corpus desde versiones fijadas:

| Fuente | Uso | Licencia declarada |
| --- | --- | --- |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Problemas con explicación y resultado numérico | MIT |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | Instrucciones para escribir Python; referencias y tests | CC-BY-4.0 |
| [SciQ](https://huggingface.co/datasets/allenai/sciq) | Preguntas y explicaciones de biología seleccionadas por vocabulario | CC-BY-NC-3.0 |

Se conservan repositorio, revisión, hashes de archivos, ID y split originales.
El corpus derivado incluye material SciQ no comercial. No se ejecuta código de
los datasets al importarlos: solo se leen archivos Parquet.

La construcción local produce **3.745 train, 168 dev y 180 test**:

| Categoría | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| Matemáticas | 1.500 | 60 | 60 |
| Programación | 745 | 48 | 60 |
| Biología | 1.500 | 60 | 60 |

Los grupos reservados son geometría/velocidades, ordenación/cadenas y
genética/ecología. Se asignan por reglas léxicas transparentes; una pregunta que
coincida con varios grupos usa la primera regla. Se eliminan duplicados de la
pregunta normalizada sustituyendo cifras, incluidos grupos que crucen particiones.
Se rechazan ejemplos que excedan 384 tokens completos, sin truncarlos.

**No son los splits oficiales de esos benchmarks.** Los splits originales se
reagrupan deliberadamente; no publicar estos resultados como puntuaciones GSM8K,
MBPP o SciQ oficiales. Los grupos léxicos y la normalización no prueban ausencia
de paráfrasis o contaminación semántica. Tampoco podemos excluir que el modelo
original viera estas fuentes durante su preentrenamiento. El filtrado SciQ puede
incluir temas fronterizos de química/medioambiente y explicaciones con referencias
a figuras ausentes; revisar la calidad del corpus antes de extraer conclusiones.

```bash
python -m asi data recovery-corpus --download --output results/recovery_corpus_v3/corpus.json
```

Omitir `--download` para reconstruir a partir de `data/recovery_sources_v3`.
Usar una salida nueva. El corpus y las fuentes están en directorios de datos/resultados
ignorados por Git; copiar el corpus al otro equipo o reconstruirlo con este comando.

## Calibración y profesor compartido

```bash
python -m asi recovery calibrate --examples results/recovery_corpus_v3/corpus.json --mask-sizes 32 16 --calibration-per-category 8 --reference-batch-size 4 --max-length 384 --output results/recovery_calibration_batched_v3
python -m asi recovery-study plan --output results/recovery_study_v3
```

La calibración solo utiliza train. Los tamaños 16 y 32 se fijan antes de ajustar
pesos. La CLI comprueba la identidad del corpus y que ningún ID de calibración
proceda de dev/test. El batching derecho usa máscara de atención; el relleno no
se cuenta al clasificar expertos. No cambia el batch del alumno, que sigue siendo 1.

El profesor usa el routing **nativo sin restricciones** sobre respuestas de referencia.
Con `--teacher-eos` también predice el final de respuesta. La pérdida del alumno es
0,5 × CE de referencia + 0,5 × KL profesor/alumno sobre tokens del asistente.
Las probabilidades se guardan completas en FP32: no se introduce aproximación top-k.
No generamos respuestas sintéticas del profesor para sustituir automáticamente las
referencias; imitarlo no garantiza que su respuesta o la referencia sean correctas.

Los tres ajustes comparten exactamente los mismos archivos del profesor. La identidad
v3 depende de fuente, corpus y objetivo; no de la selección de expertos del alumno.
Los checkpoints del alumno sí quedan vinculados a sus selecciones. El lector guarda
solo metadatos en RAM y carga **una distribución por ejemplo** al calcular la pérdida.
Verifica hashes e identidad antes de entrenar; un manifiesto parcial no es válido.

Estimación local del almacenamiento de probabilidades: **65,96 GiB**, más archivos
auxiliares y checkpoints. El modelo público ocupa otros aproximadamente 13 GiB.
Estar por debajo de 650 GB no garantiza velocidad: con poca RAM puede dominar la
lectura del disco. No confundir las lecturas lógicas con tráfico físico de SSD.

El plan permite hasta 16 GiB de caché de expertos en RAM, dentro del presupuesto
de proceso de 32 GiB (`--expert-ram-gib 16 --ram-gib 32`). No los reserva ni exige:
el almacén reduce su caché según memoria disponible y RSS. En un equipo con poca
RAM cabrán menos expertos; en homelab se podrá aprovechar más RAM para el profesor.
El valor histórico por defecto, fuera del plan, sigue siendo 0,25 GiB. Esto cambia
el coste de lecturas, no la selección matemática del profesor.

Para medir cuatro referencias antes del trabajo completo, una vez calibrado:

```bash
python -m asi recovery prepare --examples results/recovery_corpus_v3/corpus.json --study results/recovery_calibration_batched_v3/report.json --teacher-eos --max-length 384 --reference-batch-size 4 --prepare-limit 4 --teacher-dir results/recovery_study_v3/teacher --output results/recovery_teacher_profile_v3
```

El informe registra tiempo y número de ejemplos nuevos; su `completed` permanece
falso porque es un perfil parcial. No extrapolar cuatro preguntas como un benchmark
estable. Para completar, `recovery-study run` detecta ese manifiesto y añade
`--resume-teacher`. Ante otra interrupción, `recovery prepare --resume-teacher`
requiere el mismo teacher-dir y un output nuevo; conserva y verifica lo ya guardado.

## Comparación inicial: solo 32 residentes

El plan contiene tres semillas (1337, 2027 y 31415), una época por variante y 8
expertos activos por token. Las variantes son **router**, **expert-lora** y
**router-expert-lora**, con LoRA de rango 8 en la salida de los expertos. Cada una
ve los mismos ejemplos en el mismo orden dentro de una semilla. Los tamaños de
categoría son distintos: no se presenta este corpus como balanceado.

```bash
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage teacher
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage native
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage router_1337
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage expert-lora_1337
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage router-expert-lora_1337
```

Repetir las tres últimas etapas para 2027 y 31415. Las etapas son explícitas y
secuenciales: el comando `plan` no inicia nueve entrenamientos. Una salida existente
no se sobrescribe. El autostart de `asi train` pertenece al entrenador categorizado;
no se aplica a `recovery train`, que todavía no reanuda su optimizador.

Dev selecciona el checkpoint por el peor cociente de pérdida entre categorías,
incluyendo el modelo sin ajustar. Test se reserva para la comparación final. Las
generaciones tienen EOS y un máximo de 256 tokens; registrar respuestas truncadas,
no suponer que todas finalizaron. El control nativo usa los mismos prompts test y
presupuesto, con sus propias transferencias de expertos. Las comparaciones de
latencia son exploratorias sin repeticiones ni control de caché fría/caliente.

```bash
python -m asi recovery-study summarize --reports results/recovery_study_v3/native/report.json results/recovery_study_v3/router_1337/report.json results/recovery_study_v3/expert-lora_1337/report.json results/recovery_study_v3/router-expert-lora_1337/report.json --output results/recovery_summary_v3
python -m asi graphs --results results/recovery_summary_v3 --output results/recovery_graphs_v3
```

Añadir los informes de las otras semillas al mismo comando para obtener media y
desviación entre semillas. Se comprueban exposiciones, profesor, corpus, fuente y
preguntas generadas. No mezclar resultados de diferentes semillas como si fueran
repeticiones de latencia.

## Corrección, coherencia y seguimiento de instrucciones

El sistema puntúa automáticamente el número final de matemáticas, formato exigido,
sintaxis Python y presencia de la respuesta corta SciQ. **Sintaxis válida o mencionar
una palabra no significa responder correctamente.** No ejecuta código generado en
el equipo. Los tests de referencia se entregan para revisión en un entorno aislado.

```bash
python -m asi recovery-study review --examples results/recovery_corpus_v3/corpus.json --reports results/recovery_study_v3/native/report.json results/recovery_study_v3/router_1337/report.json results/recovery_study_v3/expert-lora_1337/report.json results/recovery_study_v3/router-expert-lora_1337/report.json --output results/recovery_review_v3
```

`review.jsonl` presenta pregunta, referencia y respuesta sin variante identificable
por metadatos. La rúbrica pide puntuar 0/1/2 en corrección, coherencia y cumplimiento.
Los valores permanecen vacíos hasta una revisión real; no se inventan puntuaciones.
Mantener `key.json` y los chequeos automáticos fuera de la vista del revisor hasta
terminar. Guardar también longitud, repetición, tiempo, memoria, transferencias y
la comprobación A→B→A de los informes originales.

## Después: bajar a 16

Ya existe el calendario `--mask-sizes 32 16 --curriculum-32-steps N`: comienza con
32 durante N pasos y después alterna 32/16 con exposición reproducible. Compararlo
con alternancia desde el principio, con el mismo total de pasos y presupuesto;
documentar que la exposición a cada máscara difiere por diseño. Es un estudio
separado, no una búsqueda de hiperparámetros guiada por los resultados test de arriba.
No se han ampliado las proyecciones LoRA ni entrenado todas las matrices densas.
## Revisión de objetivos de evaluación (12/10/2026)

Posteriormente, el borrador `training_draft_v3.json` añade seis propuestas de
código (primalidad, producto con signo, subsecuencias con ceros y umbral
estricto, ángulos de triángulo, tuplas anidadas y búsqueda léxica con regex).
Son 23 correcciones entre 240 filas; todos los textos caben en 384 tokens
con EOS, máximo 275. Las 27 aserciones nuevas están preparadas con hashes
en `code_replacement_cases_v3.json`; solo se ha analizado su sintaxis.
No se afirma validación funcional ni se promueve el borrador a entrenamiento.

En homelab con Docker Linux y la imagen ya disponible:

```bash
python -m asi reference-code-check \
  --corpus results/recovery_reference_review_20261011/training_draft_v3.json \
  --cases results/recovery_reference_review_20261011/code_replacement_cases_v3.json \
  --output results/recovery_reference_review_20261011/code_replacement_execution_v3.json
```

Los archivos `results/` deben copiarse también: no se presupone que un `git pull`
incluya los borradores. La ejecución requiere acceso a homelab aún no configurado.

El constructor de borradores ya no hereda silenciosamente `reference_tests`,
`expected_number` o `expected_phrase` al cambiar un enunciado o respuesta.
Conserva esos valores con prefijo `parent_`; una propuesta puede aportar
`new_reference_tests`, `new_expected_number` o `new_expected_phrase`, siempre
pendientes de validación. Así evitamos evaluar una corrección contra una
respuesta antigua incompatible.

`results/recovery_reference_review_20261011/training_draft_v2.json` conserva
los mismos 240 textos de v1 y sus 17 correcciones; los objetivos de las filas
corregidas quedan pendientes de revisión. V1 y el corpus original se conservan.
Ninguno de los dos borradores está aprobado para entrenamiento. Seis pruebas
de auditoría/revisión pasan, incluida la invalidación de objetivos antiguos.


### Paquete autocontenido para homelab

`output/recovery_reference_check_v3.zip` contiene el borrador v3, los seis
casos (27 aserciones), el ejecutor sin dependencias Python externas, un
`run.sh` y SHA256SUMS. No necesita copiar todo results ni cargar un modelo.
Descomprimir en una carpeta nueva, disponer de `python:3.11-slim` en Docker
y ejecutar `bash run.sh`. Devolver `code_replacement_execution_v3.json`.
Archivo ZIP comprobado por CRC, contenido y hashes contra los originales;
esto no sustituye la ejecución funcional pendiente en Linux.

Los 14 objetivos matemáticos revisados están comprobados mediante aritmética
racional en `numeric_targets_v3_review.json` junto al borrador. Permanecen
como propuestas separadas, ligadas por hashes; no se ha aprobado todo el corpus.

Borrador v4: 240 filas, 26 correcciones, 14 objetivos numéricos explícitos
verificados y tres explicaciones biológicas adicionales sin contexto ausente.
Tokenización real OLMoE comprobada en `draft_v4_tokenization.json`; máximo
275 tokens con EOS. Sigue sin aprobación global. V3 y su ZIP permanecen
intactos; las seis referencias de código conservan exactamente sus textos.

Correcciones de borde pendientes, separadas del ZIP v3:
`code_edge_replacements_20261012.json` propone contratos explícitos para
MCD (vacío, singleton, signos), suma de pares con enteros grandes, triángulos
escalenos válidos y bits de ancho arbitrario. Incluye 23 aserciones con sintaxis
comprobada; ninguna ejecutada. No se han integrado ni aprobado todavía.

Los tres ejercicios matemáticos marcados como condicionales tienen tasas y
duraciones expresamente estipuladas: sus objetivos 800, 4000 y 6 son correctos
bajo esos supuestos. `math_assumption_resolution.json` resuelve esta revisión
sin reescribir problemas válidos ni invertir unidades. La marca condicional
no era por sí misma un error aritmético; no implica validar hechos empíricos.
