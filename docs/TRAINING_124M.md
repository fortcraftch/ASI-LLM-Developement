# Primera LLM categorizada: presupuesto de 124M

Diseño basado en `TFG_actualizado_2026-09-27.txt` y en el `manifest.json` de la raíz. Las recetas están preparadas; no se ha entrenado ninguno de estos modelos ni medido su velocidad en la RTX 5060 Ti de 16 GB.

## Datos y categorías

El manifiesto declara 9.672.101 documentos, 9.953.989.344 tokens, 137 categorías y 17 dominios amplios. Hay 9.855.399.480 tokens de entrenamiento y 98.589.864 de validación. Tokenizador GPT-2, shards uint16. No es el pequeño dataset piloto utilizado anteriormente.

Agrupamos las 137 categorías, sin pérdidas ni duplicados, en ocho pools. Un pool es un grupo temático que recibe sus propios expertos en cada capa MoE. La taxonomía es una hipótesis experimental, no una clasificación óptima demostrada.

| Pool | Dominios originales agrupados |
|---|---|
| technology | Informática, IA e ingeniería |
| math_physics | Matemáticas y ciencias físicas |
| biology | Ciencias de la vida |
| health | Medicina y salud |
| environment | Medioambiente y energía |
| society_economy | Sociedad, derecho, economía y finanzas personales |
| history_arts | Historia/geografía, artes y filosofía |
| education_language | Educación y lengua/escritura |

Con 124M, separar 137 pools fragmentaría demasiado los expertos. Además, los dominios tienen tamaños muy diferentes: IA tiene unos 30M tokens y salud unos 1.481M. La agrupación conserva todos los datos y permite comparar ocho grupos con igual número de expertos.

## Configuración recomendada

Usar `configs/experiment_124m_v1/domain_16_uniform_pool.json`:

| Elemento | Valor |
|---|---:|
| Parámetros totales, incluidos expertos inactivos | **122.325.504** |
| Capas | 12: una densa y 11 MoE |
| Dimensión del modelo | 512 |
| Pools | 8 |
| Expertos enrutados por capa MoE | 16 |
| Expertos por pool y capa | 2 |
| Expertos enrutados activos por token y capa | 2 |
| Expertos compartidos por capa MoE | 2, siempre activos |
| Dimensión interna de cada experto enrutado | 120 |
| Contexto inicial | 1.024 tokens |

Hay 176 instancias de expertos enrutados (16 × 11). Una respuesta utiliza 22 (2 × 11), además del backbone y los expertos compartidos. Por tanto, «dos expertos en GPU» significa **dos enrutados por capa**, no dos módulos en todo el modelo. El entrenamiento mantiene todos los pesos en GPU; la residencia limitada es una política de inferencia.

`uniform_pool` significa que la categoría escoge la pareja y ambos expertos participan con el mismo peso. **E no se calcula**, ni en entrenamiento ni en inferencia. Sus parámetros siguen presentes por compatibilidad y están incluidos en el recuento, pero no reciben gradientes. Durante una respuesta la pareja permanece fija. Cambiar de pool entre entradas puede requerir transferencias. La red compartida sigue conectando información de distintos dominios: esto no garantiza aislamiento semántico absoluto.

Los expertos enrutados suman 32.440.320 parámetros; el resto suma 89.885.184. Este modelo pequeño sirve para estudiar el mecanismo: no cabe esperar reducciones gigantes de memoria total porque buena parte del modelo es compartida. Caché KV, activaciones, clasificador y estados del optimizador también ocupan memoria.

## Entrenamiento en la 5060 Ti de 16 GB

Punto de partida conservador: batch 2, contexto 1.024 y 262.144 tokens por actualización. Esto requiere **128 microbatches de acumulación**. AdamW, LR máxima 0,0003, mínima 0,00003, calentamiento de 1.000 actualizaciones, weight decay 0,1 y clipping de gradiente a 1. Son ajustes iniciales, no hiperparámetros optimizados.

El presupuesto es de 37.596 actualizaciones, aproximadamente 9.856 millones de tokens vistos. Se muestrean ventanas aleatorias con reemplazo: **no equivale a recorrer exactamente una vez todo el dataset**. El muestreo de pools es proporcional a la raíz cuadrada de sus tokens, un compromiso entre igualar los temas y seguir su volumen natural. Cada actualización usa un pool; las ventanas internas mezclan sus shards en lugar de recorrer categorías alfabéticamente.

Se validan ventanas estables cada 500 actualizaciones: 16 batches por pool, unos 32.768 tokens por pool. El `val_loss` global es la media de los pools con igual peso, no la media ponderada por tamaño del corpus. Consultar también `val_by_pool`. Esta validación parcial sirve para seguimiento, no sustituye la evaluación final extensa.

Copiar al otro equipo el código, las recetas y **todos los shards** del dataset correspondiente. Su directorio debe contener el `manifest.json` original y las carpetas de categorías con sus `.npy`. El entrenador verifica la identidad del manifiesto y de los pools; no comprueba el hash de cada shard. No sustituirlo por el piloto local.

Desde la raíz del repositorio, en PowerShell:

```powershell
$data = 'D:/datasets/specialized_fineweb_10bt' # Cambiar por la ruta real
$recipes = 'configs/experiment_124m_v1'
python -m asi train --recipe "$recipes/domain_16_uniform_pool.json" --pool-manifest "$recipes/expert_pools.json" --dry-run
```

El dry-run comprueba configuración y parámetros sin cargar pesos; **no valida VRAM ni lee los shards**. Probar después un entrenamiento corto en una carpeta independiente:

Si una copia antigua en Linux muestra `Pool manifest differs from the frozen recipe`, actualizar `asi/experiments/train.py` y el nuevo `asi/data/identity.py`: la comprobación original confundía los saltos de línea CRLF/LF con cambios del manifiesto. La versión corregida admite ambas representaciones y sigue rechazando cambios de datos u orden de pools. No regenerar la receta ni sustituir sus hashes para ocultar una discrepancia real.

```powershell
python -m asi train --recipe "$recipes/domain_16_uniform_pool.json" --pool-manifest "$recipes/expert_pools.json" --data-root $data --device cuda --max-steps 10 --val-steps-per-pool 1 --val-interval 5 --save-interval 5 --log-dir results/smoke_124m
```

Es una prueba mecánica, no de calidad. Si funciona, empezar el entrenamiento completo desde cero:

```powershell
python -m asi train --recipe "$recipes/domain_16_uniform_pool.json" --pool-manifest "$recipes/expert_pools.json" --data-root $data --device cuda --log-dir results/domain_124m
```

Si la memoria medida lo permite, antes de empezar la ejecución definitiva se puede probar `--batch-size 4`: mantiene el batch global con 64 microbatches. No hay una medición todavía que garantice ese ajuste. No cambiar batch, semilla, muestreo ni calendario al reanudar:

```powershell
python -m asi train --recipe "$recipes/domain_16_uniform_pool.json" --pool-manifest "$recipes/expert_pools.json" --data-root $data --device cuda --log-dir results/domain_124m --resume results/domain_124m/model_00500.pt
```

El checkpoint guarda optimizador, posición/aleatoriedad de datos y RNG. `run_metadata.json` registra la configuración efectiva. La reanudación exige conservar esos ajustes; no convertir el smoke de diez pasos en el entrenamiento definitivo. Las mediciones de tokens/segundo del log incluyen validación y guardado cuando ocurren en ese paso.

## Qué comparar

1. **Base**: `base_16_learned.json`. E puede elegir dos de los 16 expertos. Mismo tamaño, semilla, datos y calendario que el categorizado. Las categorías organizan los batches, pero no restringen el modelo.
2. **Base clasificado después**: reutilizar el checkpoint anterior y medir afinidades por categoría. No requiere otro preentrenamiento. La clasificación observada no demuestra una especialización exclusiva.
3. **Categorizado durante el entrenamiento**: `domain_16_uniform_pool.json`. Dos expertos fijos por tema. Como control adicional, `domain_16_learned.json` conserva E para ponderar los dos expertos del pool.
4. **Router de modelos independientes**: ocho recetas `router_*.json`, cada una entrena un modelo denso completo en un tema. Cada uno tiene 15.496.576 parámetros; los ocho suman **123.972.608**. Aquí el límite de 124M se aplica al conjunto, no a cada especialista.

Los modelos independientes reciben un presupuesto de tokens por tema aproximado al esperado con el muestreo sqrt de los MoE, redondeado por batch. Su batch global es 32.768 y tienen otro backbone y otro calendario por especialista; no son una comparación de arquitectura manteniendo idéntica optimización. Documentar estas diferencias al interpretar resultados.

Para calibrar el nuevo base local, una vez entrenado:

```powershell
python -m asi posthoc calibrate --adapter local --checkpoint results/base_124m/model_37595.pt --data-root $data --pool-manifest "$recipes/expert_pools.json" --experts-per-label 2 --output results/posthoc_base_124m
```

El adaptador local permite calibración y evaluación de afinidades. La ruta antigua `chat existing` y la suite congelada de comparación siguen destinadas al adaptador original: falta integrar allí estos checkpoints locales y esta nueva taxonomía antes de la comparación completa. No reutilizar sin revisar el bundle antiguo de pruebas.

Para inferencia categorizada con una sola pareja residente por capa:

```powershell
python -m asi chat domain --checkpoint results/domain_124m/model_37595.pt --pool-manifest "$recipes/expert_pools.json" --max-pools 1 --max-hot-pools 1
```

Para la familia Router, entrenar las ocho recetas por separado, rellenar `router_bank.json` con sus checkpoints y ejecutar `python -m asi chat router --bank configs/experiment_124m_v1/router_bank.json --pool-manifest configs/experiment_124m_v1/expert_pools.json`. Selecciona un modelo completo y lo reutiliza mientras no cambie el tema. Las rutas del banco son relativas a ese JSON. Sus valores iniciales son nulos porque aún no existen checkpoints.

## Variantes y programa dinámico

El comando `python -m asi design --output configs/nuevo_experimento` genera recetas a partir de taxonomía, manifiesto y presupuesto; cuenta parámetros sobre el grafo real sin materializar pesos. No sobrescribe directorios existentes.

| Expertos por capa | Por pool | Activos | Ancho de experto | Parámetros totales |
|---:|---:|---:|---:|---:|
| 8 | 1 | 1 | 224 | 123.632.128 |
| 16 | 2 | 2 | 120 | 122.325.504 |
| 32 | 4 | 2 | 64 | 122.685.952 |
| 64 | 8 | 2 | 32 | 121.784.832 |

Las variantes de 32/64 conservan E dentro del pool: para evitar transferencias por token se debe alojar el pool completo (4/8 expertos por capa), aunque solo dos computen por token. La de ocho activa uno; no es una comparación top-2. Cambiar el número de expertos manteniendo el límite obliga a cambiar su anchura y también afecta a los módulos compartidos: las diferencias no se pueden atribuir solo al número de expertos.

La primera entrega del paso 2.4 permite definir taxonomía, contar parámetros y adaptar el clasificador externo a los dominios del manifiesto. El paso 3.4 tiene entrenamiento y selección de modelos independientes. La ampliación opcional descrita en [ADAPTIVE.md](ADAPTIVE.md) implementa una primera decisión por memoria/calidad calibrada, unión de pools, entrenamiento de una cabeza clasificadora y caché INT8 en GPU. **Quedan pendientes sus entrenamientos definitivos y evaluación real**; las recetas de esta guía no cambian. El presupuesto de parámetros es de la LLM; el clasificador externo tiene su propio coste y debe medirse por separado.

No hay todavía resultados de calidad ni de velocidad de estas nuevas arquitecturas. Las pruebas unitarias verifican comportamiento del código; no sustituyen los entrenamientos y evaluaciones del TFG.

## Diagnóstico de timeout CUDA en MoE

Un error 702 observado en `loss.item()` puede proceder de un kernel anterior, porque esa lectura sincroniza CUDA. Terminar el proceso que falló y lanzar uno nuevo. En Linux, establecer `export ASI_MOE_BACKEND=bmm` antes del comando habitual permite aislar la ruta `grouped_mm`. `auto` conserva la selección anterior; `grouped` fuerza ese backend y falla si no está disponible. Los errores del backend se propagan sin intentar continuar con otro kernel en un contexto posiblemente inválido.

Para localizar la operación, usar temporalmente `export CUDA_LAUNCH_BLOCKING=1` y retirar después con `unset CUDA_LAUNCH_BLOCKING`; altera el rendimiento medido. La elección de backend no cambia pools ni parámetros y permite reanudar con los mismos argumentos de entrenamiento. Puede cambiar el redondeo numérico. Se registra en `run_metadata.json`. No se ha reproducido el timeout de la 5060 Ti localmente: si persiste con BMM, recoger la traza síncrona, versión de PyTorch/CUDA y `nvidia-smi` antes de atribuirlo al MoE o al controlador.
