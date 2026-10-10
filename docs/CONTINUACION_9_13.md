# Continuación de los puntos 9–13 — actualizado 07/10/2026

Esta numeración corresponde a la lista de trabajo del chat, no a los pasos
1.1–3.4 del documento original del TFG. No se consideran cerrados todos los
puntos por haber preparado su código.

## Resultados actuales: la cola de pilotos terminó

Informe reproducible: [resultados y gráficas](../results/continue_9_13_20261006/summary.md).
Se reconstruye con `python results/continue_9_13_20261006/report.py`, sin ejecutar
modelos. Los JSON originales y sus hashes quedan referenciados en ese informe.

| Punto | Ejecutado | Conclusión y límite |
|---|---|---|
| 9 | Seguimiento de tres semillas y control de normalización, seis ejecuciones cada uno | Normalizar no elimina la concentración. Delta NLL medio −0,0182, sin mejora consistente; el control sin intervención también varió. No establece capacidad lingüística mínima |
| 10 | 36 referencias verificables, 18 reutilizadas y 18 nuevas | Piloto completo; corpus general incompleto, 38/4.093 referencias nuevas |
| 11 | Router, LoRA y ambos: 24 actualizaciones por ajuste | CE test 2,275 sin ajustar → 1,694 / 1,779 / 1,558. Una semilla, seis ejemplos test; no demuestra recuperación general |
| 12 | 16 directos, alternancia y gradual | CE con 16: 2,165 / 2,421 / 2,572. El gradual no supera al directo en este piloto; persiste incoherencia |
| 13 | Inferencia aislada del ajuste mixto, 16/32 residentes y cachés RAM 0,25/2 GiB | Picos de VRAM 3,99/6,99 GiB y cero cargas de expertos dentro de las respuestas observadas; el coste de preparación sigue existiendo |

OLMoE activa **ocho expertos por token** en estas pruebas. Residentes 16/32 no
significa que se ejecuten todos. El ajuste mixto se seleccionó para perfilar por
validación, no por test. La revisión exploratoria detecta código incorrecto y
repeticiones; no se completa automáticamente el profesor entero solo por bajar CE.

La revisión adicional con hasta 256 tokens terminó en el segundo intento:
programación mejora formato y sintaxis y emite EOS, pero filtra cadenas numéricas
en lugar de quitar sus dígitos; biología acierta «energía» y luego repite hasta el
límite; matemáticas termina con `Final answer: <number>` en vez del resultado 660.
Son tres respuestas, no una evaluación representativa. El primer intento se
conserva: terminó con el código nativo 0xC0000005, sin causa identificada.

### Calibración de perfiles OLMoE

El nuevo comando valida que la calidad proceda del conjunto dev del entrenamiento
y que las mediciones sean de inferencia aislada del mismo adaptador seleccionado:

```bash
python -m asi recovery-hardware \
  --training-report results/recovery_quick_20261006/curriculum_mixed/report.json \
  --reports results/recovery_quick_20261006/profile_ram_0_25/report.json \
            results/recovery_quick_20261006/profile_ram_2/report.json \
  --gpu-gib 4 6 8 12 --ram-cache-gib 0.25 2 \
  --max-dev-kl 0.1 0.5 1 2 --reserve-mib 512 \
  --output results/recovery_hardware_nuevo
```

Resultado ejecutado: `results/recovery_hardware_20261007/summary.md`.
Peor KL dev medio por categoría: 0,726 con 32 y 2,956 con 16. Los umbrales son
ilustrativos, no garantías de corrección. La selección conserva identidad y hash
del checkpoint y devuelve `estimated_candidate` o `blocked`; no modifica el modelo
ni ejecuta automáticamente el candidato. Memoria de otras GPUs y contextos sigue
siendo una estimación. La caché RAM indicada no es un límite total del proceso.

`recovery evaluate --eval-mask-sizes 16` permite probar únicamente una máscara
que ya exista en el checkpoint. Mantiene el banco completo de adaptadores y su
identidad para cargar estrictamente los pesos, pero calcula capacidad y ejecuta
solo los expertos del perfil pedido. No crea una nueva clasificación de expertos.

Verificación real terminada: `results/recovery_quick_20261006/profile_16_only_20261007/report.json`.
Conserva 16 residentes por capa, pico CUDA 3,99 GiB y cero cargas durante las tres
respuestas posteriores al ajuste. Los tokens coinciden con los del perfil de 16
anterior. La prueba A→B→A conserva la pérdida exactamente en este caso. Se ejecutó
en la RTX 3060 de 12 GB: no equivale a haber probado físicamente una GPU de 4 GB.

El gestor aún necesita validación de decisiones extremo a extremo, más transiciones
y preguntas multilabel reales. El estudio largo de capacidad y el profesor completo
siguen pendientes: se cierra el bloque piloto, no esas investigaciones mayores.

## Historial de preparación del 06/10

## Cambio de estrategia: piloto corto antes de completar el profesor

Por decisión del usuario, se pospone completar las 4.093 referencias hasta revisar
si un piloto aporta una mejora interesante. La ejecución larga terminó con error
del proceso hijo tras 20 referencias guardadas; el log no conserva una causa
específica. No se borraron ni reinterpretaron los 424 archivos antiguos.

El piloto nuevo está en `results/recovery_quick_20261006`:

- 36 IDs congelados: 8 train, 2 dev y 2 test por cada una de tres categorías.
- Reutiliza 18 referencias nuevas y prepara únicamente las otras 18.
- Mantiene el corpus y la calibración originales. La calibración solo usó train
  del corpus padre, que es más amplio que el train del piloto.
- 24 actualizaciones, semilla 1337, 32 residentes y top-8; router, LoRA y ambos.
- Control original nativo y generación de hasta 32 tokens en un ejemplo test
  por categoría. Es un filtro preliminar pequeño, no una evaluación de respuestas
  largas ni evidencia estadística de recuperación.

Los IDs favorecen referencias disponibles, luego el orden del corpus; es una
muestra de conveniencia equilibrada. No se han elegido por los resultados del
modelo. La identidad del subconjunto queda registrada además de la del corpus.

`recovery --example-ids ids.json` permite este piloto sin falsear que el profesor
completo esté terminado. Exige cobertura de categorías y splits; verifica todos
los shards solicitados, sus hashes y su identidad embebida. Un profesor parcial
sin subconjunto explícito sigue siendo rechazado. La preparación registra
`subset_completed: true`, mientras `completed` del corpus entero permanece falso.

Se ha iniciado la preparación y una cola que espera ese subconjunto antes de
ejecutar el control nativo y los tres ajustes. Los archivos `prepare.log`,
`pipeline.log` y los logs por modo registran progreso. La cola espera como máximo
90 minutos a la preparación y se detiene si falla un proceso; no regenera todo el
corpus ni reinicia automáticamente entrenamientos interrumpidos.

Preparación terminada: 18 referencias nuevas, 589,97 s dentro de la etapa
(sin incluir toda la carga inicial). Las 36 solicitadas están disponibles y
`subset_completed` es verdadero; el profesor general conserva 38 referencias y
sigue incompleto. El control nativo ha comenzado. Las puntuaciones de recuperación
todavía no están disponibles en esta actualización.

Para ejecutar una cola preparada cuando no haya otra activa:

```bash
python -u -m asi recovery-study pilot-run \
  --output results/recovery_quick_20261006 --wait-seconds 5400
```

No acepta salidas de etapas ya existentes. La comparación y revisión ciega se
guardan al terminar; no confundir la creación de `jobs.json` con resultados.
El protocolo fija como señal provisional una mejora de al menos 5% en el peor
grupo de la pérdida de validación CE+KL (`dev_best_worst_group_ratio <= 0.95`).
No es un umbral universal de mejora sustancial ni autoriza a afirmar recuperación
sin revisar las respuestas. Un piloto negativo con solo 24 pasos tampoco descarta
el método. El test se informa, no se usa para elegir hiperparámetros.

Para el punto 12 se conserva `curriculum_jobs.json`: tres calendarios de 24 pasos,
con los mismos IDs y orden de ejemplos, y 8 pasos iniciales de 32 en el gradual.
No se han lanzado: primero se revisará el piloto de 32. Estos son ensayos pequeños
separados del plan largo descrito más abajo.

Para el punto 13, los nuevos informes de generación añaden tiempo de preparación
y picos CUDA asignado/reservado. Tras `train`, pueden incluir optimizador y
gradientes aún residentes: para calibrar inferencia aislada hay que ejecutar
`recovery evaluate` por separado. No sumar memoria reservada y asignada.

### Control adicional del punto 9

`capacity-pilot --normalize-selected` normaliza a `route_scale` la suma de pesos
top-k del router aprendido. Para unos mismos logits conserva los IDs elegidos y
mantiene gradientes; el entrenamiento posterior puede cambiar las elecciones.
El mecanismo habitual y los checkpoints existentes no se modifican. Plan e
informes registran la intervención. Está implementado y probado; todavía no hay
resultados del nuevo control largo.

Para repetir el seguimiento de 128 pasos con esta única intervención:

```bash
for seed in 1337 2027 31415; do
  python -u -m asi capacity-pilot \
    --output "results/capacity_normalized/seed_${seed}" \
    --seed "$seed" --data-seed 1337 --steps 128 \
    --eval-interval 32 --val-batches 8 --normalize-selected \
    --variants capacity_h120_k2_uniform_pool capacity_h120_k2_learned \
    --device cuda || exit 1
done
```

Ejecutarlo cuando la GPU esté libre. No confundir este control con el estudio
del tamaño de los expertos: aísla la normalización de sus contribuciones.

## Plan largo conservado, pospuesto durante el piloto

| Punto | Estado comprobado | Qué falta |
|---|---|---|
| 9. Capacidad y granularidad | Diez variantes piloto y seguimiento de seis ejecuciones terminados | Preentrenamientos largos y control de normalización; no hay umbral mínimo demostrado |
| 10. Profesor OLMoE verificable | Regeneración iniciada en carpeta nueva, con identidad y guardado atómico | Completar y verificar las 4.093 referencias |
| 11. Recuperación con 32 residentes | Plan congelado: router, LoRA y ambos, tres semillas | Ejecutar tras el profesor, control nativo, resumen y revisión ciega |
| 12. Reducción a 16 | Plan de tres calendarios y comparación pareada implementados | Ejecutar tras el punto 11 y revisar calidad de respuestas |
| 13. Gestor según hardware/calidad | Barrido de presupuestos conectado al planificador de pools existente | Mediciones OLMoE, RAM y transiciones; no hay todavía selector OLMoE calibrado |

## Profesor y recuperación

La ejecución iniciada usa el plan existente, sin modificar el corpus, y conserva
los 424 archivos históricos cuya procedencia no podía verificarse.

```bash
python -m asi recovery-study status \
  --plan results/recovery_study_v3_rebuild_20261004/plan.json
```

El contador de `teacher.entries` es progreso del manifiesto, no una auditoría
de sus tensores. El log local es
`results/recovery_study_v3_rebuild_20261004/teacher_execution.log`.
No ejecutar simultáneamente otro proceso sobre esa misma carpeta.

Si se interrumpe, el comando siguiente reutiliza el manifiesto verificable y
preserva el informe parcial. Usarlo cuando el proceso anterior haya terminado:

```bash
python -u -m asi recovery-study run \
  --plan results/recovery_study_v3_rebuild_20261004/plan.json --stage teacher
```

Cuando complete, verificar las referencias antes de interpretar resultados:

```bash
python -m asi teacher-audit \
  --teacher-dir results/recovery_study_v3_rebuild_20261004/teacher \
  --examples results/recovery_corpus_v3/corpus.json \
  --model-dir public_models/olmoe \
  --output results/teacher_verified_20261006.json
```

Después, ejecutar el control nativo y los nueve ajustes. Cada ajuste usa 3.745
actualizaciones. Estos comandos no se han lanzado automáticamente mientras se
regenera el profesor:

```bash
plan=results/recovery_study_v3_rebuild_20261004/plan.json
python -u -m asi recovery-study run --plan "$plan" --stage native || exit 1
for seed in 1337 2027 31415; do
  for mode in router expert-lora router-expert-lora; do
    python -u -m asi recovery-study run --plan "$plan" --stage "${mode}_${seed}" || exit 1
  done
done
```

Los ajustes no reanudan el optimizador a mitad de un entrenamiento. No sobrescribir
ni borrar una ejecución parcial para aparentar continuidad. Véase RECOVERY_V3.md
para agregación, conservación de archivos y revisión humana.

## Reducción progresiva: control de los datos

Plan nuevo: `results/recovery_curriculum_20261006/plan.json`. Reutiliza las rutas
del profesor y de los ajustes anteriores, sin copiarlos ni modificarlos.
Las tres políticas usan router + LoRA, rango 8, la misma unión de parámetros
adaptables y evalúan las máscaras 32 y 16:

- `direct16`: entrenar siempre con 16 residentes.
- `mixed`: alternar 32/16 desde el principio en ciclos equilibrados aleatorios.
- `gradual`: 3.745 actualizaciones con 32 y luego 7.490 alternando 32/16.

Todas tienen 11.235 pasos. `--paired-example-order` separa la aleatoriedad de los
ejemplos de la del calendario: las tres ven exactamente los mismos ejemplos en
el mismo orden para cada semilla. Se conserva la política antigua por defecto
para reproducir los experimentos históricos.

Cada ejecución parte del OLMoE original; no hereda el adaptador ganador del punto
11. Esto evita elegir un punto de partida usando el test. El gradual termina
alternando, no exclusivamente con 16. La diferencia de exposición a cada máscara
es la intervención estudiada. Tener la misma unión de parámetros adaptables no
implica que todos reciban gradientes en el control de 16.

El runner impide empezar estos trabajos si no han terminado los ajustes del punto
11. Después de revisar esos resultados:

```bash
plan=results/recovery_curriculum_20261006/plan.json
for seed in 1337 2027 31415; do
  for policy in direct16 mixed gradual; do
    python -u -m asi recovery-study run --plan "$plan" \
      --stage "curriculum_${policy}_${seed}" || exit 1
  done
done
python -m asi recovery-study curriculum-summary \
  --reports results/recovery_curriculum_20261006/curriculum_*/report.json \
  --output results/recovery_curriculum_summary
```

La comparación exige las tres políticas por semilla, mismos ejemplos y orden,
identidad del profesor, presupuesto, máscaras evaluadas y pérdidas iniciales.
La selección de checkpoint sigue usando validación. NLL no sustituye corrección,
coherencia ni revisión ciega de las respuestas completas.

## Presupuestos del gestor

`adaptive budgets` reutiliza el planificador actual de pools. Recibe el informe
de `adaptive run` (layout de memoria) y el de `adaptive evaluate` (calidad medida).
Rechaza identidades diferentes y no descarta etiquetas para hacer caber el modelo.

```bash
python -m asi adaptive budgets \
  --runtime-report results/mi_modelo_runtime.json \
  --quality-report results/mi_modelo_quality.json \
  --gpu-gib 4 6 8 12 16 --reserve-mib 512 \
  --max-nll-delta 0.1 --output results/mi_modelo_budget_grid.json
```

Son rutas de ejemplo: deben obtenerse con la calibración del modelo concreto.
El margen 0,1 es ilustrativo, no una tolerancia de calidad acordada o demostrada.
El resultado es una estimación, marca `executed: false` y conserva hashes de las
fuentes. No mide otra GPU ni latencias y no calibra RAM/disco. Los perfiles de pools
uniformes del modelo ASI no pueden reutilizarse como evidencia de OLMoE.

Para cerrar el punto 13 habrá que medir por separado residencia, RSS, transferencias,
prefill/decode y cambios de tema; reservar datos de validación para decidir y datos
test para informar. El estudio multilabel con ventanas de un solo dominio no
demuestra todavía calidad en preguntas que mezclan tareas.

Los puntos 14 en adelante (restricciones suaves, precarga/compresión, decisor
estructurado y otra familia de modelos) siguen fuera de este bloque.
