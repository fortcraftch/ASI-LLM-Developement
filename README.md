# ASI: expertos MoE y memoria limitada

Investigamos cómo ejecutar LLM mayores que la VRAM disponible manteniendo pocos
expertos en GPU y reduciendo las transferencias durante cada respuesta. Medimos
el ahorro de memoria junto con la degradación de calidad. ASI es el nombre del
prototipo, no una afirmación de superinteligencia.

Hay dos líneas: entrenar expertos con pools definidos y clasificar expertos de
modelos ya entrenados. Las recetas y los checkpoints de ambas líneas son distintos.

## Dónde empezar

**Línea de trabajo vigente:** [hoja de ruta principal del TFG](docs/ROADMAP_PRINCIPAL.md).
Recoge las siete fases acordadas, dependencias, criterios de cierre y progreso.

| Necesidad | Documento de referencia |
|---|---|
| Comparar normalizaciones con 32 expertos residentes y revisar sus resultados | [Escala de routing](docs/ROUTING_SCALE_PROTOCOL.md) |
| Preparar la comparación equivalente libre/categorizado con pesos y datos auditados | [MoE pareados de 124M](docs/MATCHED_MOE_124M.md) |
| Entrenar el modelo 3 en homelab: router global y ocho LLM independientes | [Banco de especialistas](docs/ROUTER_BANK_TRAINING.md) |
| Probar recalibración, núcleo compartido y primeras capas libres | [Intervenciones de máscaras](docs/MASK_INTERVENTIONS.md) |
| Conversar con DeepSeek libre y comparar memoria/velocidad con ASI y GPT-2 | [Generación local comparable](docs/LOCAL_124M_GENERATION.md) |
| Revisar el borrador académico y sus evidencias | [Paper en inglés](paper/README.md) |
| Separar el efecto de restringir expertos del finetuning y medir cambios de tema | [Validación controlada de recuperación](docs/RECOVERY_VALIDATION.md) |
| Investigar si la concentración de expertos persiste | [Seguimiento con tres semillas](docs/CAPACITY_FOLLOWUP.md) |
| Avanzar el paso 9 con comparaciones controladas de expertos | [Piloto de capacidad y granularidad](docs/CAPACITY_PILOT.md) |
| Ejecutar todos los benchmarks rápido, con un solo pool activo | [Evaluación por lotes y residencia completa](docs/BENCHMARKS_RAPIDOS.md) |
| Avance de los pasos 1–4 y preparación del paso 9 | [Evaluación optimizada, SFT ampliado y recetas](docs/AVANCE_2026-10-05.md) |
| Retomar el checkpoint entrenado y comparar con GPT-2 | [Evaluación 37595](docs/EVALUACION_37595.md) |
| Ajustar el modelo categorizado con respuestas OASST y compararlo antes/después | [Piloto SFT](docs/EVALUACION_37595.md#fine-tuning-supervisado-del-modelo-categorizado) |
| Revisar las ideas del tutor y la posición del TFG | [Estado del arte, actualizado 08/10/2026](docs/ESTADO_ARTE_2026-10-04.md) |
| Entender el proyecto y los términos | [Recapitulación](docs/RECAPITULACION.md) |
| Incorporar el estado actual al documento académico | [Actualización del TFG, 29/09/2026](docs/TFG_estado_actual_2026-09-29.md) |
| Saber qué está hecho, qué falta y qué archivo corresponde a cada paso | [Registro del TFG](docs/plan.json), consultable con `python -m asi plan` |
| Diseñar e interpretar experimentos | [Protocolo](docs/PLAN.md) |
| Ejecutar comandos y migrar scripts antiguos | [Uso](docs/USAGE.md) |
| Entrenar los modelos de 124M | [Entrenamiento](docs/TRAINING_124M.md) |
| Separar tamaño de expertos, capacidad activa y restricciones de routing | [Estudio de capacidad](docs/CAPACITY_STUDY.md) |
| Decidir residencia, unir pools y usar caché INT8 GPU | [Runtime adaptativo](docs/ADAPTIVE.md) |
| Compartir categorías entre modelos y clasificador | [Taxonomía](docs/TAXONOMY.md) |
| Entrenar el clasificador y planificar subtareas | [Ejemplos y tareas](docs/CLASSIFIER_DATA_TASKS.md) |
| Comparar los tres modelos entrenados | [Batería controlada](docs/COMPARISON.md) |
| Probar OLMoE público frente a AirLLM | [Piloto público](docs/PUBLIC_MOE.md) |
| Investigar la pérdida de coherencia al fijar expertos | [Estudio de routing](docs/ROUTING_STUDY.md) |
| Recuperar lenguaje con routers y LoRA de expertos, bajo varias selecciones | [Fine-tuning de recuperación](docs/RECOVERY.md) |
| Comparar los tres ajustes con datos públicos, profesor y varias semillas | [Estudio de recuperación ampliado](docs/RECOVERY_V3.md) |
| Consultar resultados ya obtenidos | [Hallazgos](docs/FINDINGS.md) |
| Entender las diferencias con AirLLM | [Trabajo relacionado](docs/AIRLLM.md) |

El registro distingue implementación, pruebas sintéticas y evidencia experimental.
No se ha demostrado todavía que fijar expertos conserve la calidad de los modelos
públicos. Las gráficas existentes se generan con `python -m asi graphs`.

```bash
python -m asi --help
python -m asi plan 2.4
python -m unittest discover -s tests -v
```

## Organización

- `asi/`: código importable; datos, modelos, runtime, análisis y experimentos.
- `configs/`: recetas y manifiestos. Los del experimento 124M están congelados.
- `data/classifier_seed_v1/`: ejemplos sintéticos del clasificador y subtareas.
- `tests/`: pruebas de mecanismos; no equivalen a resultados de calidad.
- `docs/`: guías, protocolo y documentos originales del TFG.
- `legacy/`: tres utilidades históricas, fuera de la CLI y del protocolo actual.
- `results/`: informes y checkpoints locales, excluidos de Git.
- `public_models/`: descargas públicas verificadas, excluidas de Git.
- `fineweb_edu_specialized_pipeline/specialized_fineweb/`: dataset local.

`requirements.txt` contiene las dependencias base. `requirements-public-moe.txt`
incluye esa base y los extras de AirLLM. Ambos se instalan en el mismo entorno
(localmente `F:\.venv`), sin crear una segunda venv. OLMoE usa respaldo en disco
y una caché RAM ajustable; no exige alojar todos los expertos en RAM.

Evaluación del DeepSeek original con routing libre: [protocolo, resultados y comando completo](docs/DEEPSEEK_EVAL.md).

Estado y comandos de los [puntos 9–13](docs/CONTINUACION_9_13.md): capacidad, profesor OLMoE, recuperación, calendario 32→16 y presupuestos del gestor.
## Seguimiento del piloto de recuperación

El análisis del piloto completado y los mapas están en
[RECOVERY_PILOT_ANALYSIS](docs/RECOVERY_PILOT_ANALYSIS.md).
El siguiente diagnóstico de selección de expertos con tokens idénticos se
documenta en [PAIRED_ROUTING](docs/PAIRED_ROUTING.md).

