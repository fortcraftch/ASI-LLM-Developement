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
tercera rama del experimento; todavía no hay resultados de calidad de este banco.
