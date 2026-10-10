# Recuperación controlada: original, restricción y finetuning

Esta continuación separa dos intervenciones que la comparación anterior de
tres variantes no aislaba: limitar candidatos y adaptar sus pesos/router.
El entrenamiento de DeepSeek es independiente y no se modifica.

## Piloto siguiente: 200 actualizaciones, CE frente a CE + KL

Plan vigente: `results/recovery_200_screened_20261008/plan.json`. Los borradores
`recovery_200_20261008` y `recovery_200_20261008_v2` no se ejecutaron y están
sustituidos por este plan tras revisar referencias.

- 240 ejemplos train (80 por categoría), 18 dev y 18 test; splits y separación
  por temas/plantillas heredados del corpus validado. Excluye los 36 IDs del piloto
  anterior y las nueve preguntas examinadas en la comparación de generación.
- Selección con semilla fija, longitud tokenizada máxima 384 y respuestas de
  hasta 160 tokens según el inventario. Comprobación de sintaxis de referencias
  Python, número final, presencia de frase y caracteres corruptos.
- Se excluyen explícitamente cuatro errores detectados por inspección: problema
  de donuts con día equivocado, «perímetro de cilindro» indefinido, referencia de
  respiración con cofactor mal escrito y astronomía etiquetada como biología.
  La muestra revisada está en `reference_review_sample.md`. **Esto no certifica
  semánticamente las 276 referencias** ni ejecuta sus tests Python.
- Dos entrenamientos desde el original: CE sola y 0,5 CE + 0,5 KL del profesor.
  Mismos datos, orden de ejemplos, semilla 1337, LR 1e-4, rango LoRA 8 y 200 pasos.
  Ambos usan las mismas máscaras de 32 residentes y ocho activos por token/capa.
- El profesor conserva routing libre y prepara sus probabilidades en disco;
  son distribuciones condicionadas a las referencias, no nuevas respuestas libres.
  Se utiliza un directorio nuevo, sin alterar las referencias anteriores.
- Solo se entrenan los routers y LoRA A/B de `down_proj` de los expertos de la
  unión de las máscaras. Pesos base de expertos, atención, embeddings, normas y
  cabeza de vocabulario congelados. Se registra el número de parámetros por grupo.
  Los adaptadores se comparten entre categorías cuando comparten un experto.
- Validación en los pasos 0, 50, 100, 150 y 200, con seis generaciones dev de
  hasta 256 tokens. Selección lexicográfica predefinida: error numérico, peor
  repetición media por categoría, error de formato y CE relativa. **Son proxies**,
  no una evaluación semántica del código o de biología; la revisión humana sigue
  siendo necesaria. CE es el desempate común aunque el objetivo de entrenamiento
  sea distinto. Se conserva también el paso cero.
- Guarda cinco snapshots de adaptación, además de best/last. Test se evalúa
  después de seleccionar cada checkpoint. La comparación final comprueba mismo
  orden de entrenamiento e igualdad exacta de los tensores iniciales.

La cola espera a que termine la comparación de memoria y después ejecuta
profesor → CE → CE+KL → original. No usa GPU mientras espera. La preparación del
profesor puede ser la parte más larga y escribir varios GiB de distribuciones.
Un fallo detiene la cola y conserva el trabajo parcial; no se reinicia ni se
sobrescribe automáticamente.

```bash
python -u -m asi.experiments.recovery_200 run --output results/recovery_200_screened_20261008 --wait-seconds 21600
```

No ejecutar otra copia si la cola ya está activa. `teacher.log`, `ce.log`,
`ce_kl.log` y `native.log` muestran cada etapa; `comparison.json` solo aparece
tras completarlas. Estos resultados aún no deben incorporarse como evidencia
medida al paper mientras la cola esté pendiente.

## Resultados validados del escenario de 8 GiB

### Control de memoria reducida completado

`results/recovery_memory_comparison_20261008.json` confirma igualdad exacta de
tokens en las 27 respuestas emparejadas: nueve del original, nueve de 16 sin
adaptar y nueve de 16 recuperado. Es igualdad de **cada variante consigo misma**
entre presupuestos, no igualdad entre el original y los modelos restringidos.
La auditoría adicional está en `results/recovery_memory_audit_20261008`.

| Variante | Pico GPU antes → después (GiB) | Pico RSS antes → después (GiB) | Mediana de respuesta antes → después (s) |
|---|---:|---:|---:|
| Original | 6,95 → 3,95 | 3,51 → 1,97 | 440,85 → 900,43 |
| 16 sin adaptar | 3,95 → 3,95 | 3,71 → 1,97 | 49,51 → 51,93 |
| 16 recuperado | 4,02 → 4,02 | 3,78 → 1,98 | 56,91 → 58,20 |

El original conserva sus respuestas usando un 43,2 % menos de pico GPU, con
aproximadamente el doble de tiempo mediano y 1.458,07 GiB acumulados de H2D durante
generación, frente a 657,04 GiB anteriores. Las cargas pasan de 56.067 a 124.422.
Los restringidos mantienen cero cargas durante la respuesta y 16,00 GiB de
preparación acumulada; el RSS cae casi a la mitad y los tiempos cambian poco.
No se interpreta la diferencia temporal pequeña como estadísticamente sólida.

La caché exacta permite sacrificar tiempo conservando salida; restringir candidatos
evita transferencias dentro de la respuesta, pero conserva también la degradación
ya observada: 16 recuperado sigue en 1/3 aciertos numéricos y 72,0 % de repetición
mediana. Menor RAM no la empeoró en estas preguntas. El problema de calidad no
queda resuelto por esta prueba. GPU y RAM cambiaron conjuntamente, una sola pasada
en el mismo equipo; no se aísla el coste individual de cada recurso.

La cola siguiente ya ha iniciado la preparación del profesor; no hay todavía
resultados del nuevo entrenamiento de 200 pasos.

Las cinco variantes completaron las nueve preguntas (45 respuestas). La auditoría
`results/recovery_validation_audit_20261008_v2/audit.json` comprueba hashes de las
fuentes, correspondencia con el corpus, separación de los 36 IDs del piloto,
máscaras, parada EOS, recálculo de métricas, límites medidos y transiciones.
Los informes originales se conservan intactos.

| Variante | Aciertos numéricos | Mediana de trigramas repetidos | Respuestas con EOS | Pico GPU GiB | Mediana preparar + generar (s) |
|---|---:|---:|---:|---:|---:|
| Original | 3/3 | 12,9 % | 5/9 | 6,95 | 440,85 |
| 32 sin adaptar | 1/3 | 28,0 % | 3/9 | 6,94 | 51,72 |
| 32 recuperado | 2/3 | 37,1 % | 5/9 | 7,01 | 54,05 |
| 16 sin adaptar | 0/3 | 97,2 % | 0/9 | 3,95 | 49,51 |
| 16 recuperado | 1/3 | 72,0 % | 2/9 | 4,02 | 56,91 |

**Recuperación parcial, no recuperación general de calidad.** Con 32 mejora una
respuesta matemática y el formato numérico, pero la repetición aumenta en 5 de
9 respuestas. Con 16 la repetición disminuye en 7 de 9, aunque sigue siendo alta.
El cambio mediano emparejado de repetición es +3,54 puntos en 32 y -7,48 en 16;
no debe confundirse con la diferencia entre las medianas de cada variante.
Nueve preguntas, una semilla y tres ejercicios numéricos no permiten establecer
una mejora estadísticamente sólida ni una puntuación general de inteligencia.

El original transfirió **657,04 GiB acumulados** RAM→GPU y realizó 56.067 cargas
de expertos durante la generación. Los cuatro restringidos registraron **cero**
cargas dentro de las respuestas, pero transfirieron 21,98 GiB (32) o 16,00 GiB
(16) durante la preparación y los ocho cambios de categoría. No es transferencia
cero de extremo a extremo. Estas cantidades acumuladas no son tamaño del modelo
ni lecturas físicas del SSD. Las longitudes EOS difieren; los tiempos no son una
comparación de trabajo o calidad equivalentes.

### Diagnóstico de código y revisión cualitativa

El indicador histórico `python_syntax_pass` analiza toda la respuesta si no
cumple el formato estricto. Por ello, código válido con explicación adicional
puede aparecer como fallo de sintaxis. El nuevo auditor separa ambos indicadores
sin recalcular ni reemplazar las métricas originales: el primer bloque completo
es sintácticamente válido en 3/3, 3/3, 3/3, 2/3 y 3/3, respectivamente. Ninguna
de estas cifras certifica que el programa resuelva la tarea.

Observaciones por inspección del asistente, **no evaluación humana ciega**:

- `gsm8k_train_1431`: el original y 32 recuperado obtienen 5 horas. 32 sin adaptar
  cambia indebidamente la velocidad; 16 deriva hacia repeticiones.
- `mbpp_test_154`: 16 recuperado compara una letra convertida a minúscula con
  `'A'`, lo que no implementa la posición alfabética solicitada. Es Python
  sintácticamente válido y una solución incorrecta.
- `mbpp_train_156`: el original y 32 recuperado contienen el bucle por parejas
  de índices adecuado para los ejemplos de cadenas distintas. 16 recuperado
  cuenta ambos sentidos de una pareja; no basta con que termine con EOS.
- `sciq_train_8267`: el original mezcla hábitats de agua dulce y marinos en su
  explicación; las variantes restringidas añaden palabras deformadas y bucles.
  Mencionar la frase de referencia no demuestra una explicación correcta.

También hay límites del material de evaluación: el enunciado MBPP no especifica
el nombre de función ni el retorno como cadena que exigen sus tests; por eso no
deben ejecutarse esos tests sin un contrato explícito. La referencia de
`gsm8k_train_1925` intercambia las etiquetas lago/océano en pasos intermedios,
aunque el total 11 es correcto. La pregunta sobre el ave «más grande» no fija
el criterio de tamaño: su concordancia con la referencia no se presenta como
verificación zoológica. Se conservan los nueve casos, sin retirarlos a posteriori
para mejorar puntuaciones. Las referencias deben revisarse antes de ampliar datos.

### Decisión de investigación

Primero terminar el control de memoria reducida y comprobar igualdad exacta de
tokens para cada variante. Después priorizar **32 residentes**, ampliar datos
revisados y seleccionar también por generación en validación. Los nueve ejemplos
ya inspeccionados pasan a ser diagnósticos: no usarlos para seleccionar ajustes
y después presentarlos como un test nuevo. No aumentar todavía el presupuesto de
16 ni afirmar que más pasos bastarán para recuperar la calidad.

Para reproducir la auditoría offline, usando una salida nueva:

```bash
python -m asi.analysis.recovery_validation_audit results/recovery_validation_20261008 --output results/mi_auditoria
```

Genera `audit.json`, `audit.md` y gráficas PNG/PDF. No carga el modelo ni ejecuta
las soluciones Python. Las fichas de revisión humana siguen sin puntuar.

## Protocolo congelado

Cinco procesos secuenciales: original, 32 sin adaptar, 32 recuperado, 16 sin
adaptar y 16 recuperado. Todos activan ocho expertos por token y capa. Los pares
antes/después usan exactamente las mismas máscaras; los adaptadores son los
seleccionados por validación del piloto del 06/10, sin nuevo entrenamiento.

Se eligen tres preguntas por categoría del split test del corpus local, con
semilla fija, excluyendo los 36 IDs del piloto y las generaciones archivadas
en `generation_recovery_*`. El split local puede diferir del split del dataset
original. La selección no consulta respuestas generadas ni sus puntuaciones.
Se alternan programación, matemáticas y biología: nueve preguntas independientes,
conservando la caché de expertos entre preguntas, **sin historial conversacional**.

La ejecución inicial usa greedy, EOS o 256 tokens, un límite de 8 GiB del asignador
CUDA de PyTorch, 2 GiB de caché RAM y presupuesto orientativo de proceso de 12 GiB.
El límite CUDA no incluye memoria del driver ni otros programas; el presupuesto
RSS recorta la caché, no es un límite duro del sistema operativo. El original
usa 512 parejas capa/experto LRU globales; 32 y 16 precargan sus máscaras y deben
registrar cero cargas de expertos dentro de cada respuesta.

No hay calentamiento independiente ni repeticiones temporales: las latencias
son exploratorias y están afectadas por la longitud EOS y la actividad del
equipo. La categoría viene del dataset; no se evalúa todavía el clasificador.

## Ejecutar y reanudar

Desde la raíz del proyecto, con el entorno habitual:

```bash
python -m asi.experiments.recovery_validation prepare --output results/recovery_validation_nueva
python -u -m asi.experiments.recovery_validation run --output results/recovery_validation_nueva
python -m asi.experiments.recovery_validation report --output results/recovery_validation_nueva
```

`prepare` congela preguntas, rutas, hashes, límites y variantes. `run` valida las
fuentes, ejecuta un proceso por variante y omite únicamente ejecuciones completas
asociadas a ese plan. Una ejecución parcial se conserva y detiene la reanudación
para inspeccionarla; no se borra ni se considera terminada automáticamente.
Los logs permiten seguir el avance y cada respuesta se guarda atómicamente.

`summary.md/json`, `comparison.png/pdf` y `outputs.html` recogen resultados y
textos completos. `blind_review.json` ofrece tarjetas sin nombre de variante;
`blind_key.json` permite desvelarlo después. Regenerar el informe no sobrescribe
las puntuaciones humanas. Escala propuesta 0/1/2: incorrecto/parcial/correcto,
incoherente/parcial/coherente y no/parcial/sí para seguimiento de instrucciones.

La respuesta numérica exacta puede puntuarse automáticamente. En programación,
compilar sintácticamente no demuestra resolver el problema; en biología,
mencionar la palabra esperada tampoco demuestra corrección. No se ejecuta código
generado en el equipo. No se combinan estos indicadores en una falsa puntuación
global de inteligencia. Repetición y EOS tampoco sustituyen corrección semántica.

## Siguientes controles de memoria y contexto

Tras revisar este piloto, repetir con el mismo archivo de preguntas y presupuesto
de GPU más bajo, reduciendo la caché original a 256 parejas para compararla con
16 residentes por capa. El proceso permite `--gpu-memory-gib` y
`--native-cache-experts`; no se debe presentar una mera configuración como medida.
Repetir también con caché RAM de 0,25 GiB frente a 2 GiB. Las salidas greedy de una
misma variante deberían ser iguales si solo cambia la política exacta de caché;
verificar tokens antes de interpretar diferencias de rendimiento.

Las transiciones actuales miden volver a categorías, no resolver un prompt
multilabel ni conservar el contexto de una conversación. Esas pruebas necesitan
un protocolo independiente, con evaluación de descomposición y pérdida de
información entre tareas. Todavía no se afirma haberlas validado.

## Escenarios preparados el 08/10

La ejecución principal está en `results/recovery_validation_20261008`. El segundo
plan ya está congelado en `results/recovery_validation_memory_20261008`: mismas
nueve preguntas, original y los dos controles de 16, límite CUDA de 5,5 GiB,
caché original de 256 parejas y caché RAM de 0,25 GiB. Se ha iniciado después de
validar el primero. Solo considerarlo terminado cuando sus tres informes estén
completos y exista la comparación de tokens. Para continuar, sin competir por GPU:

```bash
python -u -m asi.experiments.recovery_validation run --output results/recovery_validation_memory_20261008
python -m asi.experiments.recovery_validation compare-memory --first results/recovery_validation_20261008 --second results/recovery_validation_memory_20261008 --output results/recovery_memory_comparison_20261008.json
```

El comparador exige las mismas entradas, máscaras, adaptaciones y tokens de
salida; rechaza atribuir diferencias solo a memoria si cambia el texto. Cambian
GPU y RAM conjuntamente en este escenario: no permite aislar el efecto causal
de cada una por separado. Para hacerlo, preparar otro plan cambiando un único
límite y reutilizando `--prompt-ids`.

## Reanudar la preparación interrumpida del profesor

El 09/10 se añadió `--resume-teacher-stage` a `recovery_200 run`. Reutiliza el
manifest existente con `--resume-teacher`: verifica hashes e identidad antes de
preparar los IDs restantes. Guarda cada intento en `prepare_resume_NNN` y
`teacher_resume_NNN.log`; no sobrescribe el informe ni el log interrumpidos.
Esta opción no reanuda entrenamientos parciales ni recupera el optimizador.

```bash
python -u -m asi.experiments.recovery_200 run --output results/recovery_200_screened_20261008 --resume-teacher-stage
```
# Análisis del piloto completado

Véase [RECOVERY_PILOT_ANALYSIS.md](RECOVERY_PILOT_ANALYSIS.md) para los resultados
del piloto de 200 actualizaciones, revisión de respuestas, recursos y t-SNE.
La recuperación del formato no implica recuperación del razonamiento.

