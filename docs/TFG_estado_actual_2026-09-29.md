**Estado actual del desarrollo y resultados — 29/09/2026**

Esta sección actualiza el estado del proyecto sin sustituir los objetivos anteriores. Se distingue entre implementación disponible, prueba técnica y resultado experimental. ASI es el nombre del prototipo: no se ha demostrado una superinteligencia ni conocimiento exclusivo dentro de cada experto. El objetivo es reducir la memoria necesaria y los cambios de expertos durante una respuesta, midiendo la degradación que esto pueda producir.

**Pasos para el Desarrollo**

*Paso 1*

1. **Entrenamiento categorizado (1.1).** Están implementados la preparación de datos, los manifiestos de categorías y el entrenamiento restringido a pools. Hay recetas de 8/16/32/64 expertos bajo un presupuesto de 124 millones de parámetros. La primera receta categorizada tiene 16 expertos, ocho pools, dos expertos por pool y dos activos; contiene 122.325.504 parámetros. Los tests comprueban restricciones de rutas y gradientes, pero esto no demuestra especialización semántica exclusiva. El entrenamiento de homelab y su evaluación final deben documentarse con sus propios informes. Se añadió reinicio opcional en un proceso nuevo tras errores CUDA, guardado atómico y rotación de checkpoints.
2. **Comparación de calidad (1.2).** Está preparada la batería para comparar modelo base, base categorizado posteriormente y categorizado durante el entrenamiento. Falta completar la comparación controlada con los tres modelos entrenados; una pérdida de entrenamiento aislada no sustituye esa evaluación.
3. **Memoria y transferencias (1.3).** Se registran expertos residentes, VRAM, RAM, respaldo en disco, cargas y tiempos. La inferencia puede utilizar disco con RAM limitada. Se distinguen memoria de pesos, memoria total, lecturas lógicas y tráfico físico; no son medidas intercambiables.
4. **Selector hit/miss (1.4).** Existen políticas de carga a demanda, popularidad, tema y predicción de uso. La implementación permite estudiar anticipación y cachés, pero el predictor todavía no ha demostrado una ventaja general frente a la popularidad.
5. **Estabilidad durante la respuesta (1.5).** Con dos plazas por capa y routing libre persistían cambios frecuentes. Con una selección fija se observaron respuestas sin cargas internas de expertos, a costa de degradación. Cambiar de categoría puede requerir nuevas transferencias. Las pruebas de retorno A→B→A no demuestran estabilidad de conversaciones largas que conserven su contexto previo.

*Paso 2*

1. **Categorización posterior (2.1).** Se implementaron calibración por uso, etiquetado de expertos y carga desde disco. Se probaron el checkpoint local de ocho expertos por capa y el modelo público OLMoE-1B-7B-0924-Instruct. El prototipo local no es el DeepSeek-V3 oficial completo. Clasificar expertos no modifica sus pesos ni garantiza que su conocimiento corresponda únicamente a una etiqueta.
2. **Ejecución original frente a restringida (2.2).** OLMoE tiene 64 expertos por capa y activa ocho por token. Se comprobó que limitar residentes puede reducir transferencias, pero las selecciones pequeñas dañan mucho la coherencia. Existe un adaptador y protocolo de comparación con AirLLM; no se demuestra todavía una aceleración general con calidad equivalente. Se añadió fine-tuning del router y de las proyecciones de salida de expertos mediante LoRA, con evaluación reservada.
3. **N sobre N, N sobre 2N y selecciones mayores (2.3).** Se implementaron selección fija sin E y routing restringido dentro de un grupo residente. Hay pilotos con distintos números de residentes y activos. Conservar 32 residentes permite elegir ocho distintos por token sin mover sus pesos; no significa activar los 32. El barrido completo y su comparación con cuantización siguen pendientes.
4. **Programa dinámico (2.4).** Existen taxonomía jerárquica, perfiles de categorías por modelo, planificación por memoria y filtro por calidad medida. También hay unión de pools y planificación de subtareas explícitas. Falta una reorganización automática general de categorías según hardware, y validar la decisión conjunta de calidad, memoria y latencia con modelos finales.

*Paso 3*

1. **Escalado (3.1).** Pendiente y condicionado a demostrar utilidad en las arquitecturas actuales. No se afirma compatibilidad con cualquier MoE ni con el destino denominado V4.1 flash.
2. **Internet y herramientas (3.2).** Pendiente de integración y evaluación dentro del modelo.
3. **Expertos comprimidos en GPU (3.3).** Existe un prototipo de caché INT8 residente y promoción a pesos flotantes al cambiar de contexto, probado con modelos pequeños. Falta evaluación real de degradación y velocidad. No equivale a ejecutar todo el modelo con kernels INT8.
4. **Modelos independientes (3.4).** Están implementadas recetas y selección entre ocho especialistas densos con presupuesto conjunto inferior a 124M. Falta entrenarlos y compararlos con los modelos MoE.

**Componentes de la Arquitectura**

- **Categorizador dinámico y carga dinámica:** calibración, perfiles y cachés disponibles; adaptación automática general de la taxonomía, pendiente. Los expertos calientes pueden estar en VRAM, otros en RAM y los restantes en disco. No se exige que todo el modelo quepa en RAM.
- **Categorizador del input:** cabeza multietiqueta con embeddings compartidos y congelados, taxonomía común y corpus semilla. Falta entrenamiento definitivo y evaluación representativa. El planificador de tareas explícitas no equivale a comprender y dividir automáticamente cualquier petición compleja.
- **Selector Hit/Miss:** políticas y trazas disponibles; falta demostrar predicción útil en conversaciones reales. Cero swapping durante una respuesta no implica cero transferencias entre respuestas.
- **Gestor V:** residencia RAM/VRAM/disco e INT8 experimental disponibles. La compresión no demuestra por sí sola mejor calidad ni mayor variabilidad útil. La integración y medición a escala real siguen pendientes.

**Arquitecturas de los modelos**

- **Modelo base:** referencia con routing original. Se conserva como control de calidad; con memoria limitada también puede hacer swapping.
- **Modelo base categorizado:** checkpoint ya entrenado al que se asignan pools posteriormente. Implementado y probado; restringirlo puede deteriorar gravemente el lenguaje.
- **Modelo categorizado:** aprende desde el inicio con datos y expertos asignados a categorías. El entrenamiento está implementado; su evaluación final comparativa sigue pendiente.
- **Modelo Router:** banco de modelos independientes por categoría. Arquitectura y ejecución disponibles; entrenamiento y comparación pendientes. Este nombre no debe confundirse con el router interno de una capa MoE.

El clasificador externo tiene coste propio, que debe medirse por separado del presupuesto de parámetros de la LLM.

**Hipótesis, Investigación y Conclusiones**

El piloto de recuperación v2 utilizó 36 ejemplos train, 12 dev y 12 test. Comparó ajustar únicamente el router con ajustar el router y correcciones LoRA de los expertos. LoRA modifica el peso efectivo de su proyección de salida; no es un ajuste limitado al selector ni un reentrenamiento completo de los 7B parámetros.

- Con 32 residentes y ocho activos, la pérdida test fue 4,95 sin ajuste, 3,38 ajustando el router y 2,57 con router más LoRA.
- Con 16 residentes fue 7,37, 5,89 y 4,36 respectivamente. Sin embargo, LoRA repitió más que el control de router en generación libre con 16 residentes.
- Se observaron cero cargas de expertos durante las respuestas de esos ensayos. El resultado es recuperación parcial: una menor pérdida no garantiza corrección, coherencia ni recuperación completa del conocimiento.

El estudio ampliado v3 tiene un corpus de 3.745 ejemplos train, 168 dev y 180 test, procedentes de GSM8K, MBPP y SciQ. Reagrupa temas mediante reglas léxicas y elimina duplicados detectables; no son los splits oficiales de esos benchmarks y no se puede excluir contaminación del preentrenamiento. SciQ tiene licencia no comercial y el filtrado de biología requiere revisión.

Se calibraron selecciones con 24 ejemplos train y se verificaron inicialmente cuatro referencias del profesor original. Las referencias son probabilidades guardadas por ejemplo, no copias del modelo. El plan compara router, LoRA de expertos y ambos, con tres semillas y 32 residentes. Incluye EOS, lectura del profesor desde disco, control nativo, métricas de tareas y revisión ciega de respuestas. La estimación de almacenamiento del profesor es de unos 66 GiB. Posteriormente se podrá estudiar un calendario de 32 a 32/16 residentes.

Pasaron 110 tests y dos actualizaciones técnicas reales: distilación de 169 tokens y una prueba de 384 tokens, con picos GPU de 7,59 y 8,53 GiB en la RTX 3060. Sus pesos se descartaron. No son resultados de los nueve entrenamientos ampliados ni una validación completa de calidad. Al revisar el estado local, el manifiesto del profesor no era JSON legible por contener bytes nulos; se conservan los archivos y no se acredita una preparación completa hasta verificar o recuperar ese manifiesto.

La hipótesis sigue abierta: restringir expertos puede romper una cooperación aprendida. Se ha demostrado que algunas restricciones evitan cargas internas de expertos y que ciertos ajustes recuperan parte de la predicción. Todavía no se demuestra que modelos públicos enormes funcionen con calidad comparable en hardware pequeño sin swapping por token.

**Correspondencia con el proyecto**

- Paso 1: `asi/data/`, `asi/experiments/train.py`, `train_restart.py`, `audit.py`, `cache_study.py` y `decode_study.py`.
- Paso 2: `posthoc.py`, `fixed_study.py`, `public_moe.py`, `routing_study.py`, `recovery.py`, `recovery_protocol.py`, `recovery_study.py` y `asi/data/recovery_corpus.py`.
- Componentes dinámicos y paso 3: `asi/taxonomy.py`, `asi/runtime/`, `asi/quantization.py` y las recetas de `configs/experiment_124m_v1/`.
- Trazabilidad detallada: `docs/plan.json`. Resultados y limitaciones: `docs/FINDINGS.md`, `docs/RECOVERY.md` y `docs/RECOVERY_V3.md`. Los pasos de cada experimento deben citar su informe, hardware y configuración.
