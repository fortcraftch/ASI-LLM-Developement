# ASI: expertos MoE y memoria limitada

Investigamos cómo ejecutar LLM mayores que la VRAM disponible manteniendo pocos
expertos en GPU y reduciendo las transferencias durante cada respuesta. Medimos
el ahorro de memoria junto con la degradación de calidad. ASI es el nombre del
prototipo, no una afirmación de superinteligencia.

Hay dos líneas: entrenar expertos con pools definidos y clasificar expertos de
modelos ya entrenados. Las recetas y los checkpoints de ambas líneas son distintos.

## Dónde empezar

| Necesidad | Documento de referencia |
|---|---|
| Entender el proyecto y los términos | [Recapitulación](docs/RECAPITULACION.md) |
| Saber qué está hecho, qué falta y qué archivo corresponde a cada paso | [Registro del TFG](docs/plan.json), consultable con `python -m asi plan` |
| Diseñar e interpretar experimentos | [Protocolo](docs/PLAN.md) |
| Ejecutar comandos y migrar scripts antiguos | [Uso](docs/USAGE.md) |
| Entrenar los modelos de 124M | [Entrenamiento](docs/TRAINING_124M.md) |
| Decidir residencia, unir pools y usar caché INT8 GPU | [Runtime adaptativo](docs/ADAPTIVE.md) |
| Compartir categorías entre modelos y clasificador | [Taxonomía](docs/TAXONOMY.md) |
| Entrenar el clasificador y planificar subtareas | [Ejemplos y tareas](docs/CLASSIFIER_DATA_TASKS.md) |
| Comparar los tres modelos entrenados | [Batería controlada](docs/COMPARISON.md) |
| Probar OLMoE público frente a AirLLM | [Piloto público](docs/PUBLIC_MOE.md) |
| Investigar la pérdida de coherencia al fijar expertos | [Estudio de routing](docs/ROUTING_STUDY.md) |
| Recuperar lenguaje con routers y LoRA de expertos, bajo varias selecciones | [Fine-tuning de recuperación](docs/RECOVERY.md) |
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
