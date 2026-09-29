# Primer modelo público: OLMoE y comparación con AirLLM

Estado: adaptadores y piloto implementados. Verificación con OLMoE diminuto,
incluida ejecución real de AirLLM en RTX 3060. También se ha ejecutado calibración
del checkpoint público 7B con caché RAM de 256 MiB. Esta calibración mínima de
un ejemplo por categoría verifica el flujo; no demuestra especialización o calidad.
La generación fija real y sus limitaciones se describen en [FINDINGS.md](FINDINGS.md).
Los contrastes de residencia, activación y mezcla están en [ROUTING_STUDY.md](ROUTING_STUDY.md).

## Modelo elegido y presupuesto

[OLMoE-1B-7B-0924-Instruct de AllenAI](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct),
revisión `7f1c97f440f06ce36705e4f2b843edb5925f4498`. Licencia Apache 2.0,
modelo orientado al inglés, 16 capas MoE, 64 expertos por capa, 8 activados por
token. Se congela la revisión y se verifican SHA-256 de los archivos descargados.

| Concepto | Estimación, no pico medido |
|---|---:|
| Parámetros totales del checkpoint | 6.919.161.856 |
| Todos los pesos BF16 | 12,89 GiB (13,84 GB) |
| Pesos de todos los expertos | 12 GiB |
| Pesos comunes | 0,89 GiB |
| 8 expertos por cada una de las 16 capas | 1,5 GiB |
| Pesos comunes + selección fija | 2,39 GiB |
| Caché RAM de expertos | Ajustable a la memoria disponible; admite 0 bytes retenidos |
| Disco presupuestado para originales, AirLLM y margen | 38,67 GiB |

Además se necesitan KV, activaciones, buffers de transferencia y memoria del
framework. Se reservan inicialmente 2 GiB de VRAM para ellos. Si no cabe la
selección solicitada, el programa rechaza la ejecución: no reduce K en silencio
ni dice haber encontrado un equilibrio calidad/velocidad óptimo.

El checkpoint completo reside en disco. La RAM funciona como caché LRU adaptable:
aprovecha memoria disponible y expulsa expertos cuando hay presión de memoria.
La GPU conserva la selección activa. No se requiere RAM suficiente para todos
los pesos; se puede ejecutar en la 3060 local con una caché pequeña. Aun así,
backbone, KV y temporales deben caber en sus dispositivos respectivos.
Homelab mantiene el presupuesto de **32 GiB RAM y 650 GB de disco**.

Un modelo de 200–250 **mil millones** de parámetros ocupa aproximadamente
400–500 GB en BF16 o 100–125 GB en INT4 ideal, antes de otros costes. No cabe
completo en una caché de 32 GiB RAM, pero sí puede tener respaldo en disco.
Eso no garantiza que el backbone quepa ni que la latencia sea aceptable.
Primero probamos el mecanismo con OLMoE.

Los límites son estimaciones previas y comprobaciones del RSS del proceso por
fase/token; **no son cuotas duras del sistema operativo**. El RSS no incluye toda
la caché de archivos del SO. Para un límite físico estricto en Linux debe usarse
un contenedor/cgroup con límite de memoria y sin swap. No ejecutar junto al
entrenamiento si compiten por GPU/RAM. El adaptador evalúa; no actualiza la LLM.

## Qué hace el piloto

1. Comprueba hardware y abre el checkpoint en disco. Construye los expertos en
   `meta` (sin materializar sus pesos), carga el backbone en GPU y lee cada
   experto individualmente cuando hay un miss de GPU y RAM.
2. Ejecuta el router original sobre 24 ejemplos sintéticos de entrenamiento:
   8 de programación, 8 de matemáticas y 8 de biología. Cuenta qué expertos
   selecciona por capa y acumula sus pesos de mezcla.
3. Selecciona K expertos por categoría/capa. Las categorías pueden compartir
   expertos; no se presupone que sean especialistas exclusivos. La asociación
   significa «el router los usó con estos ejemplos», no «dominan esta materia».
4. Genera un perfil con los mismos identificadores de la taxonomía compartida.
   Entrena una cabeza ligera con los embeddings congelados del propio OLMoE y
   las **tres categorías realmente calibradas**. Su coste se registra aparte.
5. Evalúa tres prompts de validación separados del entrenamiento en cuatro
   ejecuciones independientes. Por defecto genera 16 tokens, tres repeticiones.

| Modo | Significado |
|---|---|
| `native` | Router original por token; caché GPU de 128 expertos capa/ID en total; puede transferir durante la respuesta. |
| `fixed --routing oracle` | Se proporciona la categoría correcta del ejemplo y se cargan 8 expertos por capa antes de responder. Aísla el efecto de restringir expertos. |
| `fixed --routing classifier` | La cabeza predice una categoría; permite observar también sus errores y su tiempo. |
| `airllm` | AirLLM carga capas completas; sin compresión y sin prefetch en esta primera comparación. |

En `fixed` se utilizan todos los K expertos elegidos, con mezcla constante
calibrada; no se calcula E. Se conserva la masa media de los pesos originales,
porque OLMoE no normaliza normalmente su top-8 a suma 1. Se comprueba que no
se cargue ningún experto dentro de la generación. Al cambiar de categoría se
cargan solo los expertos que faltan; repetir categoría no fuerza una recarga.

Esta primera cabeza usa **argmax de tres clases**, sin descomposición ni
multilabel general. No sustituye el clasificador semántico y planificador de
tareas ya existentes. El corpus pequeño permite comprobar la integración;
harán falta más categorías, ejemplos representativos y evaluación independiente.

## Entorno y comandos

Usar el mismo entorno para entrenamiento e inferencia. En este equipo es
`F:\.venv`. En otro equipo puede ser la `.venv` habitual del proyecto:

```bash
# Activar el entorno existente; no crear otro para AirLLM.
source .venv/bin/activate
python -m pip install --use-pep517 -r requirements-public-moe.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

El entorno debe tener PyTorch CUDA compatible con la GPU;
`cuda.is_available()` debe ser `True`. La unificación local conserva torch
2.11.0+cu128 y Transformers 5.16.1. Los extras incluyen los requisitos base.
Instalar dependencias cuando no haya un entrenamiento usando ese entorno.

```bash
python -m asi public-moe inspect --output results/olmoe_hardware.json
python -m asi public-moe download --output results/olmoe_download.json
python -m asi public-moe calibrate --experts 8 --output results/olmoe_calibration.json
python -m asi public-moe benchmark --backend native --output results/olmoe_native.json
python -m asi public-moe benchmark --backend fixed --routing oracle --output results/olmoe_fixed_oracle.json
python -m asi public-moe benchmark --backend fixed --routing classifier --output results/olmoe_fixed_classifier.json
python -m asi public-moe benchmark --backend airllm --output results/olmoe_airllm.json
python -m asi public-moe compare --reports results/olmoe_native.json results/olmoe_fixed_oracle.json results/olmoe_fixed_classifier.json results/olmoe_airllm.json --output results/olmoe_comparison.json
```

Cada comando escribe un informe nuevo; no sobrescribe resultados existentes.
Valores por defecto: RAM 32 GiB, disco 650 GB decimales, reserva GPU 2 GiB,
modelo en `public_models/olmoe`, calibración en `results/public_olmoe_calibration`.
Puede cambiarse `--model-dir` y `--bundle`, manteniéndolos iguales entre pasos.
`--ram-cache-gib 0.25` limita la caché de expertos a 256 MiB;
`--ram-cache-gib 0` desactiva su retención, pero sigue necesitando memoria temporal
para leer un experto. Sin esta opción se aprovecha la RAM disponible dentro del
presupuesto de proceso, reservando 512 MiB de margen. La política actual es LRU,
no un predictor semántico entrenado. `cache_final.disk_ram` y las instantáneas
por respuesta registran hits, misses, expulsiones y bytes lógicos leídos.
Para otro K hay que crear otro bundle e informes y usar ese K en la evaluación.
Esta coincidencia de K se exige para `fixed`. En `native`, `--experts` controla
la capacidad global de la caché GPU (valor × número de capas), sin cambiar el
top-8 del router original; permite controles con más caché sin recalibrar.
Los informes bloqueados llevan `executed=false` y el motivo; no son benchmarks.

El primer modo AirLLM también prepara archivos por capa. El adaptador convierte
sus shards derivados al formato de expertos empaquetados de Transformers una
sola vez, durante la carga inicial. No cambia los pesos originales ni añade
reempacado por token. Es **AirLLM con puente de compatibilidad OLMoE**, no una
instalación sin adaptar. Revisión AirLLM:
`ae70eefdec83e868f9825613653e34d3c442045c`.

## Cómo interpretar los resultados

Los JSON contienen tiempo inicial de carga, tiempo de clasificación y preparación,
tiempo al primer token, tiempo de generación, tokens/s de decodificación,
VRAM asignada máxima durante generación, memoria reservada, RSS muestreado,
expertos residentes por capa, hits/misses, cargas, expulsiones y bytes RAM→GPU
para ASI. Los contadores de caché son acumulados: usar la diferencia antes/después
para aislar la generación. La preparación se mide separadamente.

`compare` verifica modelo, bundle, versiones, GPU, prompts tokenizados y longitud;
separa primera pasada y posteriores. Incluye preparación y clasificación en el
tiempo de respuesta agregado. El arranque/particionado queda separado. El caché
de archivos del SO no se vacía: primera pasada no garantiza disco frío.

La coincidencia de tokens con `native` indica cuánto cambia la salida; **no mide
su corrección**. En este piloto no hay todavía NLL de continuaciones de referencia,
test de tareas ni juicio semántico. No concluir que una aceleración conserva calidad.
El bucle Python de expertos de ASI y los kernels de AirLLM también difieren: los
tiempos comparan estas implementaciones completas, no aíslan exclusivamente el swap.
Los bytes transferidos por AirLLM y su I/O de disco no están instrumentados aquí.
La generación ignora EOS para medir igual número de tokens; no es el modo de chat final.

## Archivos y correspondencia con el TFG

| Archivo | Aportación |
|---|---|
| `asi/models/hf_olmoe.py` | Pasos 2.1/2.2: calibración postentrenamiento, selección, caché y residencia de expertos. |
| `asi/models/olmoe_storage.py` | Paso 1.3/2.4: respaldo safetensors en disco, inicio meta y caché RAM adaptable. |
| `asi/experiments/public_moe.py` | Paso 2.4: piloto completo hardware → perfil de categorías → cabeza de input → inferencia. Comparación inicial relacionada con 1.2. |
| `asi/models/airllm_olmoe.py` | Comparador público con carga de capas y adaptación de formato. |
| `tests/test_public_moe.py` | Equivalencia original, caché limitada, contexto fijo, identidad de archivos y presupuesto. |
| `tests/smoke_airllm_olmoe.py` | Integración CUDA real de AirLLM con un OLMoE diminuto aleatorio, sin descargas. |

Repetir la comprobación opcional de AirLLM con:
`python tests/smoke_airllm_olmoe.py`. Escribe `results/public_moe_smoke/airllm.json`.
Los pasos generales de selección automática por calidad/hardware, cobertura universal,
validación de especialización y comparación a gran escala permanecen abiertos.
