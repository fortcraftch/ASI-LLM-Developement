# EvaluaciÃ³n del DeepSeek con routing libre

## Referencia final: paso 19072 (10/10/2026)

Checkpoint recibido: `../Creating-DeepSeek-V3-From-0/model_19072.pt`, paso 19072,
pÃ©rdida de validaciÃ³n guardada 3,07658. Se ha puesto en cola la repeticiÃ³n del
protocolo del parcial: seis tareas, 100 ejemplos, batch 8, presupuesto de 2048
tokens y auditorÃ­a de 32 tokens. Espera a que termine el experimento OLMoE para
evitar interferencias de GPU; despuÃ©s repite la generaciÃ³n de 64 tokens con
tres repeticiones y las referencias ASI/GPT-2.

Estado y resultados: `results/deepseek_19072_reference_20261010/`.
`status.json` distingue espera, ejecuciÃ³n, fallo y finalizaciÃ³n; `commands.json`
conserva los comandos y `comparison.json` registrarÃ¡ diferencias con el parcial
y comprobaciones de identidad del protocolo/datasets. La cola no constituye
todavÃ­a un resultado medido. No sustituye una evaluaciÃ³n completa de 19 tareas.

RepeticiÃ³n manual de los benchmarks en un archivo nuevo:

```bash
python -u -m asi deepseek-eval \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_19072.pt \
  --limit 100 --eval-batch-size 8 --batch-tokens 2048 \
  --audit-routing --audit-tokens 32 \
  --output results/deepseek_19072_eval_100_manual.json
```

## Protocolo general

### CachÃ© escribible para la ejecuciÃ³n local

El primer intento del paso 19072 fallÃ³ por permisos al crear locks en la cachÃ©
de usuario de Hugging Face. Se conservÃ³ el informe fallido y se copiaron los
cinco datasets ya disponibles a `results/hf_datasets_cache`. Se verificÃ³ que
sus huellas coinciden con las del parcial. La repeticiÃ³n y sus siguientes
pruebas se guardan en `results/deepseek_19072_reference_20261010/retry_cache/`;
consultar allÃ­ `status.json`, no el estado histÃ³rico del intento inicial.

El lanzador de esa repeticiÃ³n establece `HF_DATASETS_CACHE` a la ruta absoluta
de la copia local y activa `HF_DATASETS_OFFLINE=1` y `HF_HUB_OFFLINE=1` antes de
importar datasets. No modifica permisos globales ni descarga versiones nuevas.
En otros equipos, el modo offline requiere que los datasets estÃ©n ya preparados.

El comando `deepseek-eval` carga estrictamente el checkpoint y su arquitectura
original, sin convertirlo al modelo categorizado. Reutiliza los benchmarks y
criterios de puntuaciÃ³n de GPT-2 y el motor de evaluaciÃ³n por lotes de ASI.
Todos los expertos permanecen en VRAM; el router original decide cuÃ¡les usar.

Para conversar con este mismo checkpoint y comparar memoria/velocidad con ASI
y GPT-2, ver [generaciÃ³n de modelos locales](LOCAL_124M_GENERATION.md).

Desde la raÃ­z de ASI, en Bash, para ejecutar los 19 benchmarks completos y una
sonda de routing:

```bash
python -u -m asi deepseek-eval \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_07874.pt \
  --tasks all --limit 0 --device cuda \
  --eval-batch-size 8 --batch-tokens 2048 \
  --audit-routing --audit-tokens 128 \
  --output "results/deepseek_eval_$(date +%Y%m%d_%H%M%S).json"
```

`--limit 0` utiliza los datasets completos; aritmÃ©tica genera 500 ejercicios.
Los datasets no descargados necesitan conexiÃ³n. Cambiar `--checkpoint` permite
evaluar el modelo final. Para comparar las seis pruebas histÃ³ricas, omitir
`--tasks all`; usar `--limit 100` reproduce el tamaÃ±o del piloto.
`--architecture` y `--gpt2-evaluator` permiten cambiar las rutas de los proyectos
hermanos. Ante falta de VRAM, reducir el batch y el presupuesto de tokens.

El JSON se guarda despuÃ©s de cada benchmark, registra errores de datasets y
solo marca `completed: true` si terminaron todas las pruebas. No sobrescribe un
resultado existente. Conserva hashes de checkpoint, arquitectura y evaluador,
huellas de datasets, puntuaciones, tiempos y memoria CUDA.

## Piloto ejecutado: 6 de octubre de 2026

Checkpoint **parcial**, paso 7874, 123.63 M parÃ¡metros. RTX 3060, pesos FP32 y
autocast BF16. Resultado: `results/deepseek_07874_eval_100.json`.
100 ejemplos por prueba; porcentajes de acierto:

| Prueba | GPT-2 base | ASI categorizado, paso 37595 | DeepSeek libre, paso 7874 |
|---|---:|---:|---:|
| HellaSwag, normalizado | 36 | 38 | 34 |
| LAMBADA | 23 | 14 | 20 |
| PIQA | 70 | 66 | 63 |
| ARC-Easy | 40 | 42 | 38 |
| BoolQ | 52 | 57 | 69 |
| AritmÃ©tica | 0 | 0 | 0 |

Fuentes histÃ³ricas: `results/checkpoint_37595_eval_verified/report.json`
(modo `classified`) y `gpt2_matched.json` en el mismo directorio.
DeepSeek y ASI tienen idÃ©ntico hash del evaluador y huellas de datasets.
El resultado histÃ³rico GPT-2 no conserva esas huellas, por lo que su procedencia
no puede verificarse al mismo nivel. Los presupuestos de entrenamiento son
distintos y la muestra es pequeÃ±a: esto no demuestra superioridad de una
arquitectura ni mide directamente la coherencia de una conversaciÃ³n.

Las seis pruebas DeepSeek sumaron 111.60 segundos, incluyendo la preparaciÃ³n de
cada tarea, con pico de memoria asignada CUDA de 1.43 GiB. No comparar ese tiempo
como velocidad pura de arquitectura con ejecuciones histÃ³ricas que usaban otro
batch o intercambio de expertos. La generaciÃ³n nativa recalcula el contexto:
no se ha aÃ±adido una cachÃ© KV incremental a su interfaz.

## CuÃ¡ntos expertos utiliza

La sonda genera 32 tokens por prompt (programaciÃ³n, matemÃ¡ticas y biologÃ­a).
Hay 11 capas MoE, **8 expertos enrutables residentes por capa y 2 activados por
posiciÃ³n**, ademÃ¡s de la rama compartida equivalente a 2 expertos.

| Prompt | Expertos distintos por capa durante la generaciÃ³n | Cambios de selecciÃ³n por capa, sobre 31 transiciones |
|---|---:|---:|
| ProgramaciÃ³n | 4â€“8 | 18â€“29 |
| MatemÃ¡ticas | 4â€“7 | 21â€“31 |
| BiologÃ­a | 5â€“8 | 14â€“31 |

Dos activos no significa dos fijos durante toda la respuesta. El router cambia
de pareja dentro de una misma categorÃ­a. **Estos cambios no son swapping**:
todos los pesos siguen en GPU. El JSON separa las asignaciones del prompt de
las posiciones que predicen nuevos tokens, sin contar otra vez los prefijos
recalculados. Incluye selecciones y pesos por capa y paso.

`--audit-prompts archivo.json` acepta una lista de prompts propios. Esta sonda
es exploratoria, no una estimaciÃ³n representativa de todo el dataset.
Las pruebas unitarias verifican que observar el router no modifica sus logits
y que el conteo no duplica los prefijos; tambiÃ©n verifican la generaciÃ³n por
lotes frente al cÃ¡lculo secuencial con el contexto completo.

## Repetición completada (11/10/2026)

Los seis benchmarks y la generación de cinco variantes terminaron correctamente
en `results/deepseek_19072_reference_20261010/retry_cache/`. Todas las comprobaciones
de comparabilidad con el parcial son verdaderas. Aciertos finales sobre 100:
HellaSwag normalizado 34, LAMBADA 24, PIQA 64, ARC-Easy 40, BoolQ 72 y aritmética 0.
La sonda final usa 4–8 expertos distintos por capa y 17–30 cambios de selección;
todos permanecen residentes. La tabla histórica del parcial se conserva arriba.

## Suite completa iniciada el 12/10/2026

```bash
python -m asi deepseek-eval --checkpoint ../Creating-DeepSeek-V3-From-0/model_19072.pt --output results/full_suite_20261012/deepseek_19072.json --tasks all --limit 0 --eval-batch-size 4 --batch-tokens 512
```

El resultado solo es utilizable al terminar con `completed: true`, 19 tareas
sin errores y huellas/recuentos compatibles. ASI y banco ya completaron los
66.832 ejemplos de los splits locales. No comparar tiempos como velocidad
causal de arquitectura: difieren residencia, cachés y cargas del escritorio.

El motor compartido informa ahora cada 30 segundos, al terminar un lote,
cuántas solicitudes de scoring o generación se han completado. No sincroniza
CUDA para informar ni estima tiempo restante a partir de longitudes desiguales.
Esto se aplica a nuevos procesos; la ejecución completa iniciada antes conserva
su versión anterior, archivada con hash en `results/full_suite_20261012/`.
Ocho pruebas de equivalencia del motor siguen pasando.
