# Evaluar calidad con todos los pesos residentes

El objetivo de este modo es terminar los benchmarks cuanto antes. No mide el
ahorro de VRAM ni la eficacia del swapping. Por defecto, `checkpoint-eval` mantiene
todos los expertos residentes, pero `classified` activa únicamente el pool elegido
para cada pregunta (dos expertos por capa en `model_37595.pt`). El backbone y los
expertos compartidos mantienen su funcionamiento habitual.

## Batería completa en homelab

Desde la raíz del repositorio, con el entorno activado y el proyecto GPT-2 vecino:

```bash
python -u -m asi checkpoint-eval \
  --checkpoint model_37595.pt \
  --gpt2-evaluator ../Creating-GPT-2-From-0/eval.py \
  --pool-manifest configs/experiment_124m_v1/expert_pools.json \
  --tasks all --limit 0 --modes classified \
  --expert-residency all --device cuda --classifier-device cuda \
  --eval-batch-size 8 --batch-tokens 2048 --classifier-batch-size 8 \
  --skip-probes \
  --output "results/eval_rapida_$(date +%Y%m%d_%H%M%S).json"
```

Cambiar únicamente `--checkpoint` para evaluar un SFT. `--limit 0` recorre todos los
ejemplos de cada split; aritmética conserva los 500 casos del evaluador. Las tareas
son las 19 registradas en el evaluador vecino. Se mantienen sus criterios de
corrección, que no siempre coinciden con los protocolos oficiales de cada benchmark.
Los datasets deben estar disponibles o poder descargarse. Un error de acceso queda
registrado y el informe no se marca completo.

El valor predeterminado de `--modes` ahora es **solo `classified`**. Las ablaciones
`all_uniform` y `all_router` siguen disponibles explícitamente; activan expertos
de otra manera y no son necesarias para esta evaluación del categorizado.

## Qué se acelera

- Clasificación por lotes, por defecto en el dispositivo del modelo, sin pasar las
  opciones ni las respuestas al clasificador. Las rutas se reutilizan entre tareas.
- Puntuación de alternativas y ejemplos agrupados por pool y longitud. Se aplica
  padding a la derecha con máscara de pérdida; no se mezclan pools en un forward.
- Generación agrupada por pool y longitud de entrada, con caché KV y proyección al
  vocabulario solo del último token. Al agotar contexto se reproduce la ventana
  deslizante del evaluador original. EOS y límites siguen las reglas de cada tarea.
- LAMBADA conserva tanto la pérdida de la palabra objetivo como la comparación
  exacta de sus tokens generados. No se sustituye generación por teacher forcing.

En las otras tareas, una primera pasada recoge las solicitudes de puntuación y
generación del evaluador; luego se calculan por lotes y se devuelven al mismo código
de corrección. No se usa ninguna predicción provisional como resultado final.
La primera pasada puede tardar sin mostrar la barra de evaluación; la barra final
corresponde a corregir respuestas ya calculadas, **no mide velocidad del modelo**.
El campo `seconds` incluye planificación, clasificación, cálculo y corrección.

## Memoria, reproducibilidad y medición

`--eval-batch-size` limita ejemplos por lote; `--batch-tokens` limita tokens con padding
(una petición individual más larga se procesa sola). Si falta memoria, reducir a
`--eval-batch-size 2 --batch-tokens 1024` y/o `--classifier-device cpu`. No hay cambio
automático de precisión ni omisión de preguntas al producirse un OOM.
Todos los pesos del modelo deben caber para `--expert-residency all`.

Para estudiar residencia limitada usar `--expert-residency selected`. Para repetir
el orden de ejecución y cálculo anterior, añadir también `--reference-eval`
(`--classifier-device cpu` es su valor implícito). El orden agrupado reduce por sí
mismo cambios de pool; no usarlo para medir swapping de una conversación real.

La evaluación usa pesos FP32 y cálculo BF16 en CUDA, como antes. Cambiar el tamaño
de las operaciones puede cambiar ligeramente sus resultados numéricos e incluso
alguna elección cercana a un empate. Los informes identifican el motor y sus
ajustes; mantenerlos iguales en las comparaciones científicas entre checkpoints.
Las pruebas FP32 contrastan pérdidas, generación, ventanas de contexto y las 19
funciones de corrección con el evaluador original.

El JSON separa la carga inicial (`startup_cache`), las transferencias durante cada
tarea (`cache_delta`), el tiempo del clasificador (`classifier_seconds`) y el
inventario final. La memoria del clasificador se registra aparte; el pico global
de CUDA incluye también ese clasificador cuando reside allí.
`--skip-probes` omite las continuaciones cualitativas adicionales; no omite las
preguntas generativas de LAMBADA, GSM8K, TriviaQA, WebQuestions o aritmética.

## Comprobación local, 05/10/2026

RTX 3060, checkpoint `results/domain_sft_oasst_expanded/best.pt`, 100 ejemplos por
tarea. Comparación con `--reference-eval --expert-residency selected`:

| Tarea | Referencia | Optimizada |
|---|---:|---:|
| HellaSwag | 49,82 s | 8,24 s |
| LAMBADA | 39,51 s | 7,06 s |
| PIQA | 31,41 s | 2,59 s |
| ARC-Easy | 48,59 s | 3,84 s |
| BoolQ | 47,47 s | 3,98 s |
| Aritmética | 79,67 s | 6,04 s |
| Total de tareas | **296,45 s** | **31,75 s** |

Es una aceleración observada de 9,34 veces en esta muestra; no una previsión exacta
para datasets completos. Excluye cargar inicialmente modelo/clasificador. Otra
ejecución optimizada anterior tardó 24,81 s: existe variación entre sesiones.
Las 601 rutas coinciden con la referencia. Residen 16 expertos por capa, pero solo
se activa el pool seleccionado. Transferencias durante las tareas: cero; la carga
inicial sí existe. Pico CUDA global: aproximadamente 2.052 MiB, incluido clasificador.

HellaSwag, LAMBADA, ARC-Easy y aritmética conservan sus porcentajes de acierto.
PIQA cambia del 70 al 71 % y BoolQ del 60 al 61 %; no atribuirlo a mejora del modelo.
Los pesos son idénticos y el cambio de operaciones BF16 no es bit a bit equivalente.
La perplejidad también puede variar ligeramente. Se conserva el modo de referencia
para contrastarlo. No se entrenó ni recalibró ningún componente con los benchmarks.

131 pruebas automatizadas superadas, incluidas pruebas del protocolo de las 19
tareas con datos sintéticos, comparación FP32, caché KV, ventana deslizante y
residencia completa con activación restringida. Las mediciones con datasets reales
de este apartado abarcan seis tareas; no se ha ejecutado aquí toda la batería completa.

Fuentes locales: `results/benchmark_reference_all_100.json`,
`results/benchmark_optimized_final_100.json` y
`results/benchmark_optimization_summary.json`.
