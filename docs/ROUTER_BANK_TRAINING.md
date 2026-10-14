# Modelo 3: router global y especialistas independientes

Preparación del 10/10/2026. Se reutilizan las ocho recetas `router_*.json` de
`configs/experiment_124m_v1`; no se modifica el diseño congelado para los otros
modelos. Cada especialista es una LLM **densa completa**, con embeddings,
atención y cuatro capas propias. No comparte backbone ni intercambia expertos
entre capas. El router global elige un modelo por prompt/categoría y lo mantiene
durante toda la respuesta.

**15.496.576 parámetros por especialista; 123.972.608 en el conjunto.** El
clasificador externo tiene un coste separado. Esta preparación entrena las LLM,
no un nuevo clasificador; la conversación usa `DomainSessionRouter` existente.
La comparación posterior debe separar selección de categoría conocida y
categoría predicha, para no atribuir errores del clasificador a los especialistas.

## Entrenar en homelab (5060 Ti, 16 GB)

Desde la raíz del repositorio, con el entorno Python activado y el dataset
completo utilizado para diseñar las recetas:

```bash
bash configs/experiment_124m_v1/train_router_homelab.sh /ruta/al/dataset --dry-run
bash configs/experiment_124m_v1/train_router_homelab.sh /ruta/al/dataset
```

`--dry-run` comprueba arquitecturas, presupuesto y pools; no verifica los shards
ni ejecuta un paso real en la GPU remota. Al entrenar se exige el manifiesto
original del dataset y el entrenador verifica sus datos de train/validación.

El lanzador entrena los ocho modelos **secuencialmente**. Batch físico 8,
contexto 1024 y batch efectivo de 32.768 tokens: cuatro microbatches por
actualización. Guarda cada 25 actualizaciones y conserva los tres checkpoints
más recientes **de cada especialista**. Reutiliza el supervisor existente para
reiniciar un proceso limpio tras errores CUDA, hasta cinco reintentos.

Repetir el segundo comando reanuda el último checkpoint de cada modelo, omite
los terminados y continúa con el siguiente. Se guarda un plan y se rechazan
cambios de receta, batch, semilla o política de guardado en la misma salida.
Un checkpoint corrupto o de otra receta no se ignora silenciosamente.

El banco `results/router_124m_seed1337/router_bank.json` se actualiza después
de verificar el checkpoint final de cada especialista. Sus rutas son relativas
al banco, para poder transferir toda la carpeta a otro equipo.

## Exposición de entrenamiento

| Especialista | Actualizaciones | Tokens presentados |
|---|---:|---:|
| technology | 34.029 | 1.115.062.272 |
| math_physics | 27.399 | 897.810.432 |
| biology | 40.543 | 1.328.513.024 |
| health | 42.241 | 1.384.153.088 |
| environment | 24.239 | 794.263.552 |
| society_economy | 47.958 | 1.571.487.744 |
| history_arts | 52.339 | 1.715.044.352 |
| education_language | 32.024 | 1.049.362.432 |

Las recetas reparten la exposición prevista de la receta ASI por la raíz
cuadrada del tamaño de cada pool. No son ocho entrenamientos con la exposición
completa de ASI cada uno. Los tokens presentados pueden incluir repeticiones;
no equivalen a tokens únicos. La equivalencia es con el calendario previsto de
ASI, no automáticamente con la exposición real de DeepSeek `model_19072`.

## Usar el banco terminado

```bash
python -m asi chat router \
  --bank results/router_124m_seed1337/router_bank.json \
  --pool-manifest configs/experiment_124m_v1/expert_pools.json --device cuda
```

Este diseño puede ahorrar memoria residente, pero cada respuesta dispone de
solo 15,5M parámetros de LLM. No presupone que alcance la calidad de un modelo
de 124M activo por completo. Su entrenamiento y evaluación constituyen la
tercera rama del experimento. Los ocho checkpoints recibidos el 12/10/2026
superan la auditoría de integridad, receta, pasos, carga estricta y pesos finitos
(`results/router_124m_audit_20261012/report.json`). Esto no demuestra calidad.

## Evaluación con el protocolo compartido

Primera ejecución completa de los splits usados por el evaluador local,
12/10/2026, residencia de un modelo, lotes 4/512:

| Tarea | Preguntas | Acierto |
|---|---:|---:|
| HellaSwag (normalizado) | 10.042 | 25,55% |
| LAMBADA | 5.153 | 4,91% |
| PIQA | 1.838 | 56,58% |
| ARC-Easy | 570 | 38,95% |
| BoolQ | 3.270 | 39,45% |
| Aritmética sintética | 500 | 0,00% |

Fuente: `results/router_124m_audit_20261012/eval_full_six_selected_retry.json`,
`completed=true`. Pico CUDA asignado: 1061,38 MiB, incluido clasificador y
activaciones. ARC-Easy utiliza el split de 570 preguntas del evaluador, no
su test de 2376. Estas seis tareas no son la suite de 19; tampoco deben
compararse directamente con porcentajes de 100 preguntas de otros modelos.
La selección por categoría predicha y el agrupamiento de peticiones siguen
siendo limitaciones del protocolo. No es una prueba de velocidad repetida.

```bash
python -m asi bank-audit --bank results/router_124m_seed1337/router_bank.json \
  --output results/router_bank_audit_new/report.json
python -m asi bank-eval --bank results/router_124m_seed1337/router_bank.json \
  --audit results/router_bank_audit_new/report.json \
  --tasks hellaswag,lambada,piqa,arc_easy,boolq,arithmetic --limit 100 \
  --output results/router_bank_audit_new/eval_100.json
```

`--tasks all --limit 0` solicita la suite completa del evaluador de GPT-2;
no confundir un subconjunto de 100 preguntas con esa ejecución. Para usar
exclusivamente el caché compartido, configurar `HF_DATASETS_CACHE` con la ruta
absoluta de `results/hf_datasets_cache` y `HF_HUB_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`; solo funcionará con datasets previamente descargados.

El evaluador mantiene los ocho modelos en GPU para acelerar las pruebas,
ejecuta uno por enunciado y agrupa peticiones por especialista. Reutiliza
exactamente el motor de benchmarks de ASI/DeepSeek. Registra hashes de pesos,
evaluador, auditoría y manifiesto, huellas de datasets, categorías seleccionadas,
memoria del clasificador por separado y tiempos que incluyen clasificación y
carga de datos. No es una medición de swapping ni de ahorro de VRAM del banco.

`--residency selected` conserva los pesos de los otros siete modelos en RAM
y mueve a GPU únicamente el seleccionado (el clasificador tiene coste aparte).
Las activaciones del mismo modelo son hits sin nueva transferencia. El informe
separa preparación de evaluación y registra cargas, expulsiones y bytes lógicos
de pesos en ambas direcciones; no incluye buffers en esos contadores ni mide
el tráfico físico PCIe. Los lotes están agrupados por categoría: sus cambios
no representan el patrón de una conversación real. Un control de igualdad de
puntuaciones frente a `--residency all` debe preceder a conclusiones de memoria.

Control ejecutado sobre PIQA (100 preguntas): con lotes iguales 4/512, las 200
NLL de continuaciones y las 100 decisiones son exactamente iguales entre
`all` y `selected`. Con 8/2048 cambian 188 NLL y dos decisiones, con categorías
idénticas; el delta absoluto máximo de NLL media es 0,0320. Esto localiza la
diferencia observada de 56% a 54% en el batching, no en la residencia. BF16
estaba habilitado, pero falta aislar el kernel y comprobar FP32 para explicar
su origen numérico. Fuente: `piqa_trace_comparison.json` en la auditoría.

Control posterior FP32, TF32 desactivado: los dos tamaños de lote obtienen
54/100, sin cambios de decisión y con delta NLL medio máximo de 1,91e-6.
BF16 daba delta 0,0320 y dos cambios; mismo routing y checkpoints. Esto respalda
sensibilidad a la precisión, aunque no identifica el kernel responsable ni
garantiza invariancia en todas las tareas. `--precision fp32` permite reproducir
este control también en generación; el valor predeterminado sigue siendo bf16
para conservar el protocolo histórico. Evidencia: `piqa_precision_comparison.json`.

Para analizar diferencias individuales, `--trace-dir results/nueva_traza` guarda
por tarea los tokens de entrada, NLL de cada continuación y tokens generados.
La ruta debe ser nueva; las huellas de esos archivos quedan en el informe.
Esto permite detectar diferencias que dos porcentajes agregados iguales podrían
ocultar. El registro se realiza después de puntuar y no participa en el routing.

`--declared-routes annotations.json` permite un control con categorías externas:
objeto JSON que asocia SHA256 del texto UTF-8 de cada contexto a un pool.
Se rechazan entradas sin anotación, sin recurrir al clasificador. Las anotaciones
deben revisarse antes de mirar resultados; no se asigna automáticamente una
categoría «correcta» por el nombre del benchmark ni por su respuesta.
