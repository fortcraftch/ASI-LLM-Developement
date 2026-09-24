# Mapa del TFG y protocolo

La numeración **1.1–3.3** corresponde a cada elemento de las listas de los pasos
1, 2 y 3 de [TFG_original.txt](TFG_original.txt). El documento propone hipótesis;
el código y los pilotos permiten comprobar solo una parte de ellas.

El registro completo está en [plan.json](plan.json). Consultarlo con
`python -m asi plan` o `python -m asi plan 1.4` no carga modelos.

## Estado por paso

Las rutas abreviadas parten de `asi/`. «Disponible» describe herramientas, no
hipótesis demostradas.

| Paso | Código principal | Disponible | Pendiente |
|---|---|---|---|
| **1.1** Entrenar por categorías | `data/*`, `models/domain.py`, `experiments/train.py`, `analysis/experts.py` | Pools, entrenamiento restringido, aislamiento de rutas y gradientes | Entrenamiento suficiente y evaluación/ablaciones que demuestren especialización |
| **1.2** Calidad frente al original | `experiments/audit.py`, `models/*` | Auditoría restringida/libre/ablación de un checkpoint | Control entrenado con iguales datos, parámetros y presupuesto; varias semillas |
| **1.3** GPU/RAM/SSD | `runtime/cache.py`, `runtime/storage.py`, `models/original.py` | Inventario, GPU limitada por capa, shards de disco y caché RAM limitada; arranque sin cargar expertos | RSS total, I/O físico y generalización a modelos grandes; compresión pendiente |
| **1.4** Predicción de caché | `runtime/routing.py`, `runtime/cache.py`, `experiments/cache_study.py` | Predictor estadístico entrenable, actualización online opcional y comparación con LRU/popularidad/semántica | Más datos independientes, ajuste en desarrollo y demostrar beneficio neto |
| **1.5** Conversaciones | `runtime/generation.py`, `experiments/cache_study.py`, `experiments/decode_study.py` | Sesiones diagnósticas, generación incremental real, repetición, swapping por token y límite de dos expertos/capa | Conversaciones reales largas, corpus mayor y varias semillas |
| **2.1** Categorizar pesos existentes | `models/original.py`, `analysis/experts.py`, `experiments/posthoc.py` | Calibración por capa, E, etiquetas candidatas y hashes | Estabilidad y especificidad; carga escalable de modelos oficiales grandes |
| **2.2** Comparar con router E | `experiments/posthoc.py`, `runtime/cache.py`, `runtime/generation.py` | Baseline residente, prefetch exacto y restricción; piloto real | Benchmark repetido, calidad externa y coste integral |
| **2.3** N activos de N/2N/XN | `experiments/posthoc.py`, `experiments/train.py`, `models/domain.py`, `quantization.py` | Tamaños configurables y curvas de cobertura K | Entrenamientos comparables, toda la pool residente y cuantización ejecutable |
| **3.1** Escalar arquitectura | Sin implementación específica | Protocolo reutilizable | Verificar destino y demostrar beneficio antes de escalar |
| **3.2** Herramientas/internet | Sin implementación | — | Integración, permisos y evaluación |
| **3.3** Compresión en GPU | `quantization.py` como preparación | Exportador offline INT8 | Tier comprimido en GPU, descompresión/kernels y evaluación |

## Responsabilidad de cada archivo o grupo

| Archivo o grupo | Responsabilidad y pasos |
|---|---|
| `asi/data/prepare.py` | Clasifica documentos, aplica reglas de subdominio, divide train/val y escribe shards. **1.1**, corpus para **2.1**. |
| `asi/data/pools.py`, `configs/expert_pools.json` | Agrupan categorías sin copiar tokens y rechazan asignaciones inválidas. **1.1, 2.1**. |
| `asi/data/inspect.py` | Revisa distribución y cobertura. Control de **1.1, 2.1**. |
| `asi/models/domain.py`, `configs/domain_experts.json` | Arquitectura con pools. El JSON documenta una referencia; el entrenamiento usa argumentos. **1.1, 1.2, 2.3**. |
| `asi/models/original.py` | Carga estricta de arquitectura externa, hashes y adaptación de generación. **2.1, 2.2**. |
| `asi/experiments/train.py` | Muestreo, lectura continua de shards pequeños, entrenamiento, validación y checkpoints. **1.1, 2.3**. |
| `asi/experiments/audit.py` | Compara routing restringido/libre/ablación, exporta pérdida, memoria y trazas. **1.1–1.3, 1.5**. |
| `asi/experiments/posthoc.py` | Calibración, evaluación separada y comparación de políticas. **2.1–2.3**. |
| `asi/experiments/cache_study.py` | Aprende uso por contexto en train, congela en test y compara cinco políticas en GPU con igual presupuesto entre cachés. **1.4, 1.5, 2.2**. |
| `asi/experiments/decode_study.py` | Compara las cinco políticas generando respuestas, separa prefill/decode y registra cada token bajo límite por capa. **1.3, 1.5, 2.2**. |
| `asi/analysis/experts.py` | `RoutingTrace` y `ExpertCalibrator`: identidad capa/experto, selecciones, pesos, E·h y asociaciones entre capas. **1.1, 1.5, 2.1, 2.2**. |
| `asi/runtime/routing.py` | Clasificación multietiqueta, continuidad heurística y `ExpertUsagePredictor`, que aprende tasas por etiqueta actual/anterior. **1.4, 1.5**. |
| `asi/runtime/cache.py` | Caché por pools y caché original; backing RAM, residencia GPU, precarga, desalojos e inventario. **1.3–1.5, 2.2**. |
| `asi/runtime/storage.py` | Exporta shards sin comprimir y gestiona LRU de tensores de expertos en RAM, con hashes y lecturas de disco. Junto con el cargador meta en `models/original.py`, permite arrancar sin cargar todos los expertos. **1.3, 2.1, 2.2**. |
| `asi/runtime/generation.py` | Dos modalidades de conversación, contexto opcional y registros. **1.5, 2.2**. |
| `asi/quantization.py` | Exporta pesos INT8 y escalas; no ejecuta expertos comprimidos. Preparación de **1.3, 2.3, 3.3**. |
| `asi/__main__.py`, `asi/__init__.py`, `asi/plan.py` | CLI, rutas y mapa: infraestructura transversal. Los demás `__init__.py` delimitan paquetes. |
| `examples/prompts.jsonl` | Diagnóstico, mezclas y cambios de tema. **1.5, 2.2**; no es un benchmark de calidad. |
| `examples/sessions.jsonl`, `tests/test_cache_study.py` | Sesiones diagnósticas redactadas con split explícito y pruebas de causalidad temporal, serialización, aislamiento train/test y precarga exacta CPU/GPU. **1.4, 1.5, 2.2**. |
| `tests/test_expert_runtime.py` | Aislamiento, equivalencia de cómputo, caché CUDA, calibración y shards. **1.1, 1.3, 2.1, 2.2**. |
| `tests/test_posthoc.py` | Muestreo, cobertura y procedencia. **2.1–2.3**. |
| `tests/test_incremental_storage.py` | KV incremental, límites por capa, métricas por token, carga meta, límites RAM, integridad y equivalencia de ejecución desde disco. **1.3, 1.5, 2.2**. |
| `tests/test_project_structure.py` | CLI, importaciones sin ejecución y rutas del mapa: infraestructura. |
| `legacy/fineweb.py`, `legacy/hellaswag.py`, `legacy/plot_results.py` | Utilidades históricas de descarga, evaluación y gráficas. Fuera del protocolo actual; no prueban un paso por sí solas. |
| `requirements.txt`, `.gitignore`, `README.md`, `docs/*` | Dependencias, separación de artefactos y documentación transversal. |

## Conceptos e interpretación

Un experto se identifica por **(capa, ID)**. El experto 2 de una capa y el de otra
son pesos distintos. Un experto calibrado puede asociarse a varias etiquetas.

**E** son los pesos del router. La selección depende también del estado oculto h,
activación, sesgos y reglas top-K. Comparar vectores E aislados no basta para
asignar una profesión: se observan E·h y selecciones con entradas etiquetadas,
contrastando con popularidad global.

Los expertos no se llaman entre sí como funciones independientes: cada capa
selecciona expertos y transmite su representación a capas posteriores. Una
arista de una traza es una asociación, no una prueba causal de comunicación
indebida. Para estudiar dependencias hacen falta ablaciones y medir sus efectos.

En PyTorch, `device=cpu` significa almacenamiento en **RAM**. Estas cachés
transfieren los pesos a GPU para ejecutar sus FFN. RAM fijada puede facilitar
transferencias; el beneficio debe medirse. Backbone, expertos compartidos y
buffers también ocupan memoria. El modo de shards añade carga disco/RAM/GPU;
su límite RAM corresponde a tensores de expertos, no al RSS total ni a la caché
de páginas del sistema operativo. El registro de bytes leídos es lógico y no
demuestra cuántos bytes alcanzaron físicamente el SSD.

El criterio principal es **cuántos pasos de decode no necesitan cargar expertos**,
acompañado por bytes transferidos y capacidad GPU/RAM. Un 90% de aciertos por
experto puede seguir implicando cargas en todos los tokens si falla alguna capa.
La latencia es un coste que se debe documentar, no una promesa de superar al modelo
residente. Dos expertos significa dos **enrutados** por capa; los compartidos,
backbone y KV se contabilizan aparte.

## Orden y criterios del estudio

1. **Controles y corpus (1.1, 2.1).** Versionar manifiestos, tokenizer, tokens,
   hashes y arquitectura. Reservar calibración, desarrollo y test; revisar
   duplicados. Ventanas no solapadas no garantizan independencia del entrenamiento.
2. **Categorizar el checkpoint (2.1).** Equilibrar tokens por dominio, comparar
   rankings con popularidad global y repetir muestreo. Revisar estabilidad de
   etiquetas y señalar explícitamente dominios sin validación.
3. **Equivalencia y coste (2.2, 1.3).** Comparar residencia completa y prefetch
   con mismas entradas y dtype; verificar rutas, NLL y logits con tolerancia
   declarada. Probar restricción como una modificación de calidad distinta.
4. **Predicción y sesiones (1.4, 1.5).** Secuencias estables, transiciones habituales
   y cambios bruscos. Comparar LRU, popularidad, clasificación y predictor aprendido
   con igual capacidad. Una política que conozca el futuro es solo una cota ideal.
5. **Entrenamiento controlado (1.1, 1.2, 2.3).** Igual arquitectura base, datos,
   tokens, tokenizer, optimizador y semillas emparejadas. Informar parámetros
   totales/activos y cómputo. Variar tamaño de pool por separado y declarar cambios
   de presupuesto; cambiar el router de un checkpoint no sustituye ese control.
6. **Compresión y escalado (2.3, 3.3; luego 3.1–3.2).** Comparar compresión con
   coste RAM/GPU antes de escalar. Verificar disponibilidad y arquitectura del
   destino denominado V4.1 flash en el documento: no es una dependencia actual.

Fijar antes de cada ensayo los márgenes aceptables de calidad y latencia según
el uso previsto. Un miss no invalida el sistema por sí solo: importa su frecuencia
y coste, y el objetivo conjunto de memoria, calidad y latencia.

## Cómo publicar un resultado comprensible

Conservar comando, configuración, semillas, versiones, hardware, hashes, entradas
exactas, exclusiones, salidas y resumen humano. Usar una carpeta nueva por ensayo.
Separar etiquetas conocidas del corpus de predicciones del clasificador.

| Métrica | Interpretación y cautela |
|---|---|
| NLL y delta | Menor es mejor sobre esas entradas; requiere igual tokenización/protocolo. No equivale a calidad de tareas. |
| Cobertura top-K | Fracción de selecciones nativas dentro de candidatos; no mide precisión de respuestas. |
| Preferencia por dominio | Contrastar con popularidad global, tokens observados y variabilidad entre muestras. |
| Hit/miss | En caché nativa cuenta expertos únicos por capa/forward, no tokens. Acompañar de precargas y bytes. |
| Tiempo | Separar clasificación, precarga, prefill/decode cuando sea posible. Calentar, sincronizar CUDA, repetir, alternar orden; publicar p50/p95 y dispersión. |
| Memoria | Separar expertos, backbone/shared, KV/buffers, clasificador, temporales, allocator, RAM total y disco. No sumar reservada y asignada como cantidades disjuntas. |
| Calidad de tarea | Evaluación externa por dominio y entradas mixtas, criterios previos y ejemplos de fallos. |

Cada conclusión debe presentar pregunta, control, intervención, tamaño de muestra,
resultado, limitaciones y decisión siguiente. Los tests verifican mecanismos;
las hipótesis requieren experimentos. Véase [FINDINGS.md](FINDINGS.md).
