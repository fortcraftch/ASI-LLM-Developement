# Estado del arte y posición del TFG — actualizado el 8 de octubre de 2026

Documento iniciado el 04/10; se conserva el nombre del archivo para mantener los enlaces. La actualización del 05/10 incorpora Jev/TypeSafe y el piloto SFT. Las propuestas experimentales no equivalen a implementaciones ni reproducciones.

La revisión del paper añade un precedente importante: **DEMix** (Gururangan
et al., NAACL 2022) entrena expertos feed-forward por dominios y estudia su mezcla,
adición y retirada. Por tanto, entrenar expertos temáticos no es una novedad por
sí misma. El TFG estudia el compromiso entre calidad, memoria y transferencias
al mantener candidatos residentes durante una respuesta completa, comparando
restricción durante entrenamiento y posterior. [Artículo y publicación oficial](https://aclanthology.org/2022.naacl-main.407/).
El [borrador del paper](../paper/README.md) reúne las evidencias más recientes,
incluidos los resultados negativos de generación y la recuperación parcial.

## Pregunta de investigación

¿Cuánta calidad se pierde al mantener un conjunto pequeño de expertos MoE durante toda una respuesta, y cuánto puede recuperarse mediante entrenamiento por categorías o adaptación posterior, bajo presupuestos explícitos de VRAM, RAM y transferencias?

El objetivo es ejecutar modelos mayores que la VRAM disponible. La velocidad y la calidad son restricciones y medidas del compromiso, no una promesa de mejorar el modelo original. Guardar pesos en RAM no significa ejecutar operaciones en RAM: el cálculo se hace en CPU o GPU. El disco permite alojar expertos fríos, pero su capacidad por sí sola no garantiza una latencia utilizable.

## Trabajos remitidos por el tutor

### ReMoE: adaptar el router para reutilizar expertos

ReMoE ajusta únicamente el router para favorecer reutilización entre tokens próximos, sin modificar los expertos ni exigir una nueva política de caché. La publicación informa mejoras de reutilización y tiempos en sistemas con offloading. Su unidad de estabilidad es temporal; no exige una pareja temática fija durante toda la respuesta. La ficha es de ICML 2026. El repositorio consultado publica instrucciones y un modelo GGUF; no debe confundirse con una reproducción local ya realizada. [Artículo](https://proceedings.mlr.press/v306/zhu26ab.html), [repositorio de autores](https://github.com/BUAA-OSCAR/ReMoE).

**Experimento propuesto para ASI:** comparar tres restricciones: ninguna, preferencia suave por expertos residentes y máscara fija por respuesta. Mantener idénticos ejemplos y presupuesto de adaptación. Medir calidad, cambios por token y bytes; no maximizar únicamente el hit rate, porque un router que colapsa en pocos expertos también puede tener muchos aciertos de caché. Es una propuesta inspirada en ReMoE, no una implementación del artículo.

### FloE: reducir el coste de mover cada experto

FloE combina sparsificación contextual de las proyecciones gate/down y cuantización de muy pocos bits de la proyección up. Predictores de expertos y de canales, kernels específicos y transferencias compactas permiten reducir tráfico y solapar movimiento y cálculo. No se limita a seleccionar un subconjunto fijo de expertos. Sus aceleraciones dependen de los modelos, baselines y hardware del artículo; no se pueden trasladar directamente a nuestra RTX 3060. [Artículo y arquitectura](https://arxiv.org/html/2505.05950v2).

**Experimento propuesto:** primero establecer un control INT8 con idéntico routing y medir degradación y bytes. La poda por canales y los kernels especializados vendrían después: combinar todas las técnicas desde el principio impediría atribuir las mejoras. Nuestro almacenamiento INT8 no demuestra por sí mismo las ventajas del sistema FloE.

### HybriMoE: decidir dónde calcular, además de dónde guardar

HybriMoE distribuye trabajo entre CPU y GPU dentro de una capa, conserva expertos según puntuaciones y anticipa cargas por su impacto sobre la planificación. Usa mediciones iniciales de cómputo y transferencias; un experto no residente puede calcularse en CPU si es conveniente. Está implementado sobre kTransformers y figura como aceptado en DAC 2025. [Artículo](https://arxiv.org/html/2504.05897v1).

**Experimento propuesto:** microbenchmark por hardware para decidir entre transferir un experto o calcularlo en CPU. Separar prefill y decode: tienen cargas distintas. Esta alternativa preservaría rutas que una máscara rígida descarta, aunque puede aumentar latencia. La caché RAM de ASI no equivale a tener implementado este planificador híbrido.

### Mira: caché GPU y precarga predictiva

Mira propone predictores ligeros por capa que anticipan expertos dos capas adelante, una región HOT y otra STAGE en GPU, telemetría para reajustar residencia y un formato INT8 con cálculo/descuantización fusionados. HOT y STAGE son ambas regiones de GPU; no corresponden directamente a nuestra distinción RAM/VRAM. La versión consultada es el preprint del 29 de septiembre de 2026, sin revisión por pares confirmada aquí. [Artículo](https://arxiv.org/html/2609.38090v1).

**Experimento propuesto:** reservar una fracción del presupuesto GPU para precarga, contabilizando expertos precargados que finalmente no se usan. Comparar frente a LRU y al predictor histórico del proyecto. Solo hablar de solapamiento efectivo tras medirlo; `non_blocking` o una predicción correcta no prueban que se haya ocultado el tiempo de transferencia.

### CLM: selección mediante similitud, sin generar una etiqueta palabra a palabra

El [enlace de LinkedIn del tutor](https://lnkd.in/p/eNCYWtRq) describe Contrastive Language Models. La documentación del paquete de los autores expone codificadores de estado y acción, entrenamiento contrastivo InfoNCE y reutilización de embeddings de candidatos. Su referencia usa un backbone Qwen3-8B y cabezas entrenables: no es un clasificador pequeño por el mero hecho de no generar texto. [Documentación primaria del paquete](https://pypi.org/project/contrastive-lm/).

**Experimento propuesto:** representar cada pool mediante una descripción y ejemplos, precalcular sus vectores y comparar la entrada con ellos usando un encoder pequeño o representaciones compartidas. Para multilabel hacen falta objetivos y umbrales adecuados: una elección softmax exclusiva no resuelve varias etiquetas. Medir macro/micro-F1, calibración, rechazo de entradas desconocidas y coste separado del clasificador. No se ha implementado CLM en ASI.

### Jev / TypeSafe: decisiones estructuradas para seleccionar categorías y acciones

**Procedencia y verificación (05/10/2026).** El tutor remitió esta [publicación de LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7510782536386748416/). No se pudo recuperar su contenido, por lo que no se atribuyen a su autor afirmaciones concretas ni se confirma qué versión describe. A partir de la referencia explícita del estudiante a «Jev/TypeSafe», se consultaron fuentes oficiales. No se utilizaron como documentación oficial los sitios de terceros con nombres similares.

**Qué ofrece.** Jev recibe un estado y preguntas con respuestas de tipo predefinido. Expone Choice (elección entre opciones), Score (evaluación sobre una escala) y Noul (probabilidad de una afirmación binaria). Devuelve valores y distribuciones utilizables por código, sin redactar respuestas de chat. Varias preguntas pueden evaluarse contra el mismo estado. Es pertinente para clasificación y decisiones del runtime, no un sustituto directo del generador de texto. [Introducción oficial](https://docs.typesafe.ai/introduction), [concepto System One](https://docs.typesafe.ai/concepts/system-one).

**Arquitectura y acceso: límites de lo verificable.** El anuncio del 15/09/2026 presenta una arquitectura, un muestreador paralelo y un método denominado Reinforcement Learning for Calibrated Decisions (RLCD). Esto no especifica suficientemente las capas, parámetros o entrenamiento para reproducirlo. Las fuentes consultadas documentan acceso mediante API; no se han verificado pesos descargables ni si su arquitectura interna es densa o MoE. Por tanto, no se propone «particionar Jev en expertos» como una operación ya viable. [Anuncio de los autores](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API oficial](https://api.typesafe.ai/redoc).

**Resultados que no debemos extrapolar.** Las aceleraciones anunciadas corresponden a workflows y comparadores del proveedor; su anuncio reconoce límites del protocolo, referencias basadas en otros modelos y posibles sesgos. No equivalen a aceleración local en una RTX 3060, ni comparan conversación abierta con conversación abierta. Garantizar la estructura de salida tampoco garantiza que la opción elegida sea correcta. [Discusión experimental del proveedor](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

**Confianza no significa acierto garantizado.** TypeSafe distingue probabilidades y un indicador de confianza derivado de ellas. Para Choice, con n opciones, define `(p_max - 1/n) / (1 - 1/n)`. Ese valor no es directamente una probabilidad de acierto. La calibración se comprueba sobre conjuntos de predicciones y no certifica una respuesta individual. Hay que medirla sobre nuestras categorías, especialmente ante entradas desconocidas o ambiguas. [Confianza](https://docs.typesafe.ai/confidence), [límites de System One](https://docs.typesafe.ai/concepts/system-one).

**Línea experimental propuesta para ASI (sin implementar).** Separar el modelo que decide qué recursos utilizar del modelo que redacta la respuesta:

1. Comparar el clasificador actual con un decisor local de salidas estructuradas, entrenado sobre el manifiesto común y datos propios o con licencia adecuada. La interfaz estaría inspirada en Jev; no sería una reproducción de sus pesos ni de RLCD.
2. Predecir varias categorías con salidas binarias por categoría y umbrales calibrados. Una única Choice exclusiva no implementa multilabel. La coincidencia entre etiquetas no basta para concluir que las tareas del prompt son separables.
3. Dejar que el código aplique las restricciones duras de memoria. El decisor puede puntuar pools o detectar incertidumbre; el gestor debe validar VRAM/RAM y decidir si ampliar candidatos, usar respaldo o pedir aclaración. Una predicción no garantiza el presupuesto.
4. Medir micro/macro-F1, cobertura frente a error al abstenerse, Brier/ECE con definición del binning, latencia p50/p95, memoria y coste del decisor. Después medir calidad final, cargas y bytes del generador con los mismos prompts. En un control por API, separar red y servicio del cómputo local y fijar la versión.
5. Solo si el decisor local justifica mayor capacidad, comparar su versión densa con una versión MoE especializada por categorías, manteniendo datos y presupuesto controlados. Esta sería una rama sobre **modelos de decisión categorizados**, distinta de convertir un LLM generativo denso a MoE.

**Relación con CLM.** Ambas referencias motivan decisiones sin generar una etiqueta token a token. CLM utiliza representaciones contrastivas de estados y candidatos; Jev publica una interfaz de decisiones tipadas y entrenamiento RLCD. No se han verificado como arquitecturas equivalentes. Estado local: revisión documental y propuesta; sin integración API, entrenamiento de un decisor nuevo ni mediciones de Jev.

## Qué aporta y qué no demuestra nuestro prototipo

La gestión de expertos, offloading, precarga y adaptación de routers tienen antecedentes directos. No sería defendible presentar esas ideas aisladas como originales. La posible contribución del TFG es una evaluación controlada de **residencia estable por respuesta condicionada por categorías**, comparando especialización durante el preentrenamiento y clasificación posterior, y exponiendo conjuntamente calidad, memoria y movimiento de datos.

El proyecto dispone de un modelo propio categorizado de 122.325.504 parámetros, un adaptador de OLMoE público con disco/RAM/GPU, instrumentación de cachés y rutas, recuperación con router y LoRA de expertos, protocolos de comparación y planificación multilabel. Son prototipos con diferentes niveles de validación, no un cargador universal ni reproducciones de los trabajos anteriores. El historial y sus limitaciones están en [la actualización anterior](TFG_estado_actual_2026-09-29.md); los nuevos resultados se documentan en [la evaluación del checkpoint](EVALUACION_37595.md).

La evaluación local del 04/10 compara tres configuraciones del mismo checkpoint y el GPT-2 propio, en seis tareas de 100 casos cada una. La mezcla completa mejora descriptivamente LAMBADA (14% → 23%), pero reduce ARC-Easy (42% → 36%) frente a la pareja por categoría. Las continuaciones conservan errores y repeticiones incluso indicando el tema manualmente. Con dos residentes por capa se miden 358,35 MiB de pesos GPU y 9,21 tokens/s; con 16 activos, 466,63 MiB y 5,90 tokens/s. Son mediciones del prototipo bmm en RTX 3060, con cero cargas durante las respuestas y sin equivalencia con los sistemas optimizados de los artículos.

Actualización medida el 05/10: el SFT OASST de 200 actualizaciones reduce la pérdida de validación de 3,109 a 2,811. Con dos expertos por categoría, los mismos 100 casos por tarea arrojan HellaSwag 38% → 38%, LAMBADA 14% → 19%, PIQA 66% → 69%, ARC-Easy 42% → 40%, BoolQ 57% → 64% y aritmética 0% → 0%. Persisten errores y repeticiones en conversación: no se ha demostrado recuperación general. Se mantienen memoria residente y ausencia de cargas durante las respuestas medidas. El [estudio de capacidad](CAPACITY_STUDY.md) prepara controles de tamaño y granularidad, pero no ha entrenado el barrido ni establecido un umbral mínimo por experto. La optimización del evaluador MoE sigue pendiente; menor residencia no implica automáticamente menor latencia.

El siguiente control `base_16_learned` está preparado, no entrenado. El estudio ampliado de recuperación tampoco está terminado: existen 424 referencias estructuralmente válidas de 4.093, pero un manifiesto dañado impide verificar estrictamente su procedencia. Se preservan los archivos y se prepara regeneración verificable. Separar siempre «implementado», «medido» y «pendiente» al trasladar estos resultados al TFG.

## Prioridad experimental

1. Cerrar la evaluación del checkpoint categorizado con selección por entrada y controles de routing, y entrenar el base de igual presupuesto.
2. Clasificar ese mismo base después de entrenarlo: evita necesitar un tercer preentrenamiento y permite una comparación causal de máscaras sobre pesos idénticos.
3. Completar referencias verificables del profesor, control nativo y las tres adaptaciones con 32 residentes. La auditoría de archivos interrumpidos no es un resultado de recuperación.
4. Evaluar como línea complementaria un decisor estructurado inspirado en Jev frente al clasificador actual, sin confundirlo con una reproducción de Jev ni con el generador de respuestas.
5. Contrastar residencia rígida con una restricción suave inspirada en ReMoE. Solo después añadir predicción/precarga y compresión, mediante ablaciones separadas.

Para cada ensayo: identidades de datos/pesos, semillas, presupuesto de entrenamiento, protocolo de prompts, NLL y tareas, revisión ciega de respuestas, TTFT y tiempo por token, VRAM asignada/pico, RSS, bytes por nivel y fallos de caché. No comparar aceleraciones entre máquinas como si fueran causales ni equiparar ausencia de swapping con corrección.

## Ayuda concreta que solicitar a la universidad

- **NLP y evaluación:** revisar agrupaciones temáticas, contaminación de benchmarks y diseño de evaluación ciega; establecer qué degradación es aceptable.
- **Aprendizaje eficiente:** decidir cómo adaptar rutas sin colapso de expertos y cómo distinguir pérdida de cooperación de pérdida de conocimiento.
- **Sistemas GPU:** revisar trazas de memoria, medir PCIe/disco y estudiar cuándo precarga o cálculo CPU compensa su coste.
- **Diseño experimental:** dimensionar muestras, intervalos e hipótesis; separar efectos de precisión numérica, selección y backend.

Resumen para presentar: «Estudio la viabilidad de mantener expertos MoE residentes durante una respuesta para ejecutar modelos con poca VRAM. Comparo entrenamiento categorizado con especialización posterior y adaptación, midiendo el compromiso entre calidad, latencia y tráfico de memoria. Busco revisión metodológica y apoyo para validar las conclusiones, no doy por demostrada la conservación de capacidades».
