# Evaluación del DeepSeek con routing libre

## Referencia final: paso 19072 (10/10/2026)

Checkpoint recibido: `../Creating-DeepSeek-V3-From-0/model_19072.pt`, paso 19072,
pérdida de validación guardada 3,07658. Se ha puesto en cola la repetición del
protocolo del parcial: seis tareas, 100 ejemplos, batch 8, presupuesto de 2048
tokens y auditoría de 32 tokens. Espera a que termine el experimento OLMoE para
evitar interferencias de GPU; después repite la generación de 64 tokens con
tres repeticiones y las referencias ASI/GPT-2.

Estado y resultados: `results/deepseek_19072_reference_20261010/`.
`status.json` distingue espera, ejecución, fallo y finalización; `commands.json`
conserva los comandos y `comparison.json` registrará diferencias con el parcial
y comprobaciones de identidad del protocolo/datasets. La cola no constituye
todavía un resultado medido. No sustituye una evaluación completa de 19 tareas.

Repetición manual de los benchmarks en un archivo nuevo:

```bash
python -u -m asi deepseek-eval \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_19072.pt \
  --limit 100 --eval-batch-size 8 --batch-tokens 2048 \
  --audit-routing --audit-tokens 32 \
  --output results/deepseek_19072_eval_100_manual.json
```

## Protocolo general

El comando `deepseek-eval` carga estrictamente el checkpoint y su arquitectura
original, sin convertirlo al modelo categorizado. Reutiliza los benchmarks y
criterios de puntuación de GPT-2 y el motor de evaluación por lotes de ASI.
Todos los expertos permanecen en VRAM; el router original decide cuáles usar.

Para conversar con este mismo checkpoint y comparar memoria/velocidad con ASI
y GPT-2, ver [generación de modelos locales](LOCAL_124M_GENERATION.md).

Desde la raíz de ASI, en Bash, para ejecutar los 19 benchmarks completos y una
sonda de routing:

```bash
python -u -m asi deepseek-eval \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_07874.pt \
  --tasks all --limit 0 --device cuda \
  --eval-batch-size 8 --batch-tokens 2048 \
  --audit-routing --audit-tokens 128 \
  --output "results/deepseek_eval_$(date +%Y%m%d_%H%M%S).json"
```

`--limit 0` utiliza los datasets completos; aritmética genera 500 ejercicios.
Los datasets no descargados necesitan conexión. Cambiar `--checkpoint` permite
evaluar el modelo final. Para comparar las seis pruebas históricas, omitir
`--tasks all`; usar `--limit 100` reproduce el tamaño del piloto.
`--architecture` y `--gpt2-evaluator` permiten cambiar las rutas de los proyectos
hermanos. Ante falta de VRAM, reducir el batch y el presupuesto de tokens.

El JSON se guarda después de cada benchmark, registra errores de datasets y
solo marca `completed: true` si terminaron todas las pruebas. No sobrescribe un
resultado existente. Conserva hashes de checkpoint, arquitectura y evaluador,
huellas de datasets, puntuaciones, tiempos y memoria CUDA.

## Piloto ejecutado: 6 de octubre de 2026

Checkpoint **parcial**, paso 7874, 123.63 M parámetros. RTX 3060, pesos FP32 y
autocast BF16. Resultado: `results/deepseek_07874_eval_100.json`.
100 ejemplos por prueba; porcentajes de acierto:

| Prueba | GPT-2 base | ASI categorizado, paso 37595 | DeepSeek libre, paso 7874 |
|---|---:|---:|---:|
| HellaSwag, normalizado | 36 | 38 | 34 |
| LAMBADA | 23 | 14 | 20 |
| PIQA | 70 | 66 | 63 |
| ARC-Easy | 40 | 42 | 38 |
| BoolQ | 52 | 57 | 69 |
| Aritmética | 0 | 0 | 0 |

Fuentes históricas: `results/checkpoint_37595_eval_verified/report.json`
(modo `classified`) y `gpt2_matched.json` en el mismo directorio.
DeepSeek y ASI tienen idéntico hash del evaluador y huellas de datasets.
El resultado histórico GPT-2 no conserva esas huellas, por lo que su procedencia
no puede verificarse al mismo nivel. Los presupuestos de entrenamiento son
distintos y la muestra es pequeña: esto no demuestra superioridad de una
arquitectura ni mide directamente la coherencia de una conversación.

Las seis pruebas DeepSeek sumaron 111.60 segundos, incluyendo la preparación de
cada tarea, con pico de memoria asignada CUDA de 1.43 GiB. No comparar ese tiempo
como velocidad pura de arquitectura con ejecuciones históricas que usaban otro
batch o intercambio de expertos. La generación nativa recalcula el contexto:
no se ha añadido una caché KV incremental a su interfaz.

## Cuántos expertos utiliza

La sonda genera 32 tokens por prompt (programación, matemáticas y biología).
Hay 11 capas MoE, **8 expertos enrutables residentes por capa y 2 activados por
posición**, además de la rama compartida equivalente a 2 expertos.

| Prompt | Expertos distintos por capa durante la generación | Cambios de selección por capa, sobre 31 transiciones |
|---|---:|---:|
| Programación | 4–8 | 18–29 |
| Matemáticas | 4–7 | 21–31 |
| Biología | 5–8 | 14–31 |

Dos activos no significa dos fijos durante toda la respuesta. El router cambia
de pareja dentro de una misma categoría. **Estos cambios no son swapping**:
todos los pesos siguen en GPU. El JSON separa las asignaciones del prompt de
las posiciones que predicen nuevos tokens, sin contar otra vez los prefijos
recalculados. Incluye selecciones y pesos por capa y paso.

`--audit-prompts archivo.json` acepta una lista de prompts propios. Esta sonda
es exploratoria, no una estimación representativa de todo el dataset.
Las pruebas unitarias verifican que observar el router no modifica sus logits
y que el conteo no duplica los prefijos; también verifican la generación por
lotes frente al cálculo secuencial con el contexto completo.
