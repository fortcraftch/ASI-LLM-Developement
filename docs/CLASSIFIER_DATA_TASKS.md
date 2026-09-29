# Ejemplos del clasificador y tareas separables

## Corpus inicial original

`data/classifier_seed_v1` contiene **528 ejemplos originales sintéticos**: 456 de entrenamiento, 36 de validación y 36 de test reservado. No proceden de scraping ni tienen revisión humana. El generador reproducible es `asi/data/classifier_examples.py`.

Incluye anclas de nombre/alcance para las 207 etiquetas del catálogo, escenarios de los 30 dominios en español e inglés, consultas multietiqueta relacionadas, tareas dependientes/independientes, ambigüedades léxicas y entradas sin contexto suficiente (`unknown`). Por ejemplo, una pitón y un programa Python no deben clasificarse por una coincidencia de palabras; comparar código e implementarlo puede ser una única categoría con dos acciones.

Las anclas usan nombres y descripciones del catálogo para enseñar la asociación inicial etiqueta-concepto. **No son ejemplos representativos de conversación**. Las etiquetas finas tienen principalmente anclas de entrenamiento y no validación propia: `report.json` enumera las etiquetas sin validación directa. Este corpus sirve para comprobar entrenamiento, anotación y formato, no para demostrar un clasificador general fiable.

La validación y el test usan escenarios distintos de las plantillas de entrenamiento. Se comprueban duplicados del texto y de los últimos 512 tokens, familias de plantillas y IDs de categorías. La separación documental, los casi duplicados semánticos y la corrección de todas las anotaciones todavía requieren revisión. Las etiquetas amplias de validación no son anotaciones finas exhaustivas; las métricas globales sobre todos los nodos se deben interpretar como diagnóstico, no como precisión de categorías finas.

Archivos:

- `train_val.jsonl`: entrada compatible con el entrenador actual.
- `heldout_test.jsonl`: reservado; no lo usa el entrenador ni se debe mezclar con validación para ajustar umbrales.
- `report.json`: distribución, cobertura y limitaciones.
- `tasks/*.json`: cuatro casos estructurados para estudiar separación y residencia.

Para regenerar en un directorio nuevo:

```bash
python -m asi data classifier-examples --output data/otro_seed
```

Cuando exista el checkpoint adecuado:

```bash
python -m asi adaptive train-classifier --checkpoint results/domain_124m/model_37595.pt --pool-manifest configs/experiment_124m_v1/expert_pools.json --taxonomy configs/knowledge_v1.json --examples data/classifier_seed_v1/train_val.jsonl --output results/semantic_seed_head
```

No se ha ejecutado ese entrenamiento real. Para pasar de semilla a clasificador útil: ampliar con preguntas reales por categoría, anotaciones multietiqueta completas, suficientes ejemplos desconocidos/difíciles y test independiente. La metadata `task_relation` no entrena un detector de tareas: la cabeza actual predice temas únicamente.

## Temas no son tareas

| Input | Plan | Cambios de pools después de la carga inicial |
|---|---|---:|
| Explicar las matemáticas de una red neuronal | Una explicación conjunta; matemáticas + IA | 0, pero deben caber ambos pools |
| Comparar dos implementaciones Python y programar la elegida | Comparación → implementación; ambas de tecnología | 0 |
| Deducir una derivada y después implementarla | Matemáticas → tecnología, pasando el resultado | 1 |
| Deducir → programar → demostrar una propiedad del programa concreto | Matemáticas → tecnología → matemáticas | 2; rechazado con límite 1 |

La clasificación concreta de una comparación de algoritmos depende del contenido. No se asigna matemáticas automáticamente por aparecer «eficiencia»: el análisis algorítmico también pertenece a informática. La separación evita tener que alojar ambos pools a la vez **solo cuando la dependencia entre tareas se puede expresar mediante un resultado intermedio**. No se divide una relación conceptual por el mero hecho de tener dos etiquetas.

Un cambio de pools puede implicar transferir varios expertos de varias capas. «Un cambio» no significa una única copia de tensor. Además, la primera carga desde RAM sigue existiendo. El resultado medido es ausencia de transferencias durante cada generación y cambios limitados entre tareas; no garantiza menor tiempo total porque se añaden prefills y contexto intermedio.

## Planificador de tareas

Cada tarea declara `id`, `prompt`, `categories` y `depends_on`. `relation` es `coupled`, `independent` o `dependent`. Se valida el grafo y, para hasta 12 tareas, se busca un orden que respete dependencias y minimice cambios del conjunto de pools requerido. Solo se reordenan tareas permitidas por el grafo. Si necesita más cambios que el límite, no ejecuta.

```bash
python -m asi tasks --spec data/classifier_seed_v1/tasks/derive_then_code.json --profile configs/model_categories_124m_v1.json --max-switches 1 --max-hot-pools 1 --output results/derive_plan.json
python -m asi tasks --spec data/classifier_seed_v1/tasks/compare_python.json --profile configs/model_categories_124m_v1.json --output results/python_plan.json
python -m asi tasks --spec data/classifier_seed_v1/tasks/neural_math.json --profile configs/model_categories_124m_v1.json --max-hot-pools 2 --output results/neural_plan.json
```

El límite se calcula sobre cambios de conjuntos requeridos. El planificador no optimiza todavía una caché que retenga varios grupos entre fases ni compara los costes reales de ejecutar unido frente a separado.

## Propuestas automáticas conservadoras

```bash
python -m asi tasks --prompt 'Qué código Python es más eficiente entre A y B y el que lo sea prográmalo' --output results/task_proposal.json
```

El prototipo reconoce por reglas una comparación seguida de implementación en español, ignorando delimitadores dentro de bloques de código cerrados. Conserva el input original. Si no reconoce una frontera clara, mantiene una sola tarea. La propuesta deja las categorías vacías: deben asignarse antes de planificar. **No hay todavía un descomponedor semántico entrenado ni una extracción automática fiable para cualquier input**. Esto permite experimentar con planes explícitos sin atribuir a un clasificador de temas la capacidad de detectar dependencias.

## Ejecución y resultados intermedios

Con un perfil vinculado al checkpoint, se puede añadir `--execute --checkpoint ... --pool-manifest ...` al comando con `--spec`. El ejecutor verifica identidad y estima memoria para cada grupo. Esta primera ruta usa expertos densos; no incorpora aún la selección por calidad calibrada ni el tier INT8 del runtime adaptativo.

Antes de cada tarea se preparan sus pools. El prompt incluye la solicitud original y los resultados de sus dependencias. Cada tarea empieza un nuevo prefill: no se reutiliza una caché KV calculada con otros expertos. Se rechaza un prompt que, junto al presupuesto de respuesta, supere el contexto; no se truncan silenciosamente resultados intermedios.

La salida final concatena las respuestas en el orden original de tareas, sin llamar de nuevo a la LLM para sintetizarlas. Así no se introduce otro cambio oculto. No se ejecuta el código generado ni se verifica automáticamente que sea correcto. La propagación de un resultado intermedio equivocado sigue siendo un riesgo de calidad que debe evaluarse.

Las pruebas comprueban cobertura del corpus, aislamiento de splits, distinción entre temas y acciones, dependencias, límites de cambios, conservación de respuestas intermedias y ausencia de cambios de caché durante una tarea. Las pruebas con respuestas simuladas verifican coordinación, no capacidad matemática/programación. El siguiente estudio debe comparar respuesta conjunta frente a tareas con los mismos pesos, calidad evaluada, tiempo total, tokens adicionales, pico de memoria y bytes transferidos.
