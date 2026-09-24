# Comandos y migración

Ejecutar desde la raíz con el entorno activado. En esta máquina se ha validado
`F:\.venv\Scripts\python.exe`. Para otro entorno, instalar `requirements.txt`
y verificar CUDA en PyTorch. Clasificador y tokenizer pueden descargarse la
primera vez. Consultar opciones con `--help` en cualquier comando.

## Datos y entrenamiento — 1.1

El dataset permanece en `fineweb_edu_specialized_pipeline/specialized_fineweb`;
es la ruta por defecto para preparación, inspección y entrenamiento.

```powershell
python -m asi data inspect
python -m asi data prepare --max-docs 10000 --output results/dataset_trial
python -m asi data pools --manifest results/dataset_trial/manifest.json --pools configs/expert_pools.json --output results/dataset_trial/expert_pools.json
python -m asi train --total-batch-size 16384 --batch-size 2 --seq-len 512 --max-steps 10 --val-interval 5 --save-interval 5 --log-dir results/domain_trial
```

Adaptar la configuración de pools si un dataset pequeño no contiene todas sus
categorías. El manifiesto no copia tokens. Los shards son arrays NumPy uint16
de tokens GPT-2. `prepare --resume` continúa y `--reset` elimina shards previos
de su salida. Las puntuaciones de subdominio son heurísticas, no probabilidades.

El entrenamiento configura la arquitectura mediante argumentos; no lee
automáticamente `configs/domain_experts.json`. Guarda configuración efectiva y
orden de pools en checkpoints. `train --resume` reanuda. Diez pasos solo verifican
el flujo técnico, no especialización.

## Auditoría y conversación por pools — 1.1–1.5

```powershell
python -m asi audit --checkpoint results/domain_trial/model_00009.pt --pool-manifest fineweb_edu_specialized_pipeline/specialized_fineweb/expert_pools.json --pin-memory --export-router-vectors --output results/domain_trial_audit
python -m asi chat domain --checkpoint results/domain_trial/model_00009.pt --pin-memory --session-context --trace-dir results/domain_trial_chat
```

`audit --smoke` utiliza pesos aleatorios pequeños. `--no-classifier` usa palabras
clave: son diagnósticos, no sustituyen el sistema real. Elegir modos con
`--modes restricted unrestricted ablated`. `--max-hot-pools` limita residencia;
tener pesos retenidos no los autoriza en la máscara de routing.

## Modelo ya entrenado — 2.1–2.3

Se requiere la arquitectura externa
`F:/LLM/Creating-DeepSeek-V3-From-0/train_deepseek_v3.py` con sus dependencias.
Si cambia de ruta, usar `--architecture`. El cargador espera `DeepSeekV3Config`
y `DeepSeekV3`, carga estrictamente y registra hashes; no es un importador
universal de checkpoints oficiales.

```powershell
python -m asi posthoc calibrate --checkpoint results/expert_audit_training/model_04750.pt --data-root fineweb_edu_specialized_pipeline/specialized_fineweb --pool-manifest fineweb_edu_specialized_pipeline/specialized_fineweb/expert_pools.json --pin-memory --output results/posthoc_new_run
python -m asi chat existing --checkpoint results/expert_audit_training/model_04750.pt --expert-labels results/posthoc_04750/expert_labels.json --pin-memory --session-context --output results/chat_new_run.json
```

Una entrada: añadir `--prompt "Explain how a Python generator works."`.
Una lista: `--prompts examples/prompts.jsonl`. Se rechazan etiquetas con hashes
distintos del checkpoint/arquitectura actual: recalibrar después de cambiarlos.

`--policy prefetch` conserva routing y carga desde RAM lo no previsto.
`--policy restrict` cambia el modelo y puede degradar calidad.
`--max-hot-experts` limita instancias residentes globales, no expertos por token;
debe admitir la unión demandada por una capa en un forward. El generador por defecto
reprocesa contexto; `--incremental` activa el adaptador KV. `--experts-per-layer 2`
lo activa automáticamente y usa prefill serial para respetar el límite también
al leer el prompt. `--session-context`
conserva historia; sin él no se comparte texto aunque la caché persista.

La calibración usa por defecto 32 ventanas train de 128 tokens por dominio y
hasta ocho ventanas val. Las etiquetas del corpus son conocidas; el flujo de
prompts mide además predicciones y coste del clasificador.

| Artefacto | Uso |
|---|---|
| `SUMMARY.md`, `report.json` | Resumen y evidencia completa, ventanas, métricas y procedencia |
| `expert_labels.json` | Candidatos editables; `reviewed: false` indica propuesta |
| `expert_domain_scores.csv`, `calibration.json` | Tasas, pesos y E·h para revisar etiquetas |
| `router_vectors/` | Matrices E por capa |
| `coverage_curves.json`, `global_popularity_coverage.json` | Cobertura temática frente a popularidad |
| `native_routes.json` | Selecciones nativas para contrastar candidatos |

## Exportación y pruebas

### Estudio de caché aprendido — 1.4, 1.5 y 2.2

```powershell
python -m asi cache-study --checkpoint results/expert_audit_training/model_04750.pt --expert-labels results/posthoc_04750/expert_labels.json --sessions examples/sessions.jsonl --max-hot-experts 52 --prefetch-experts 32 --repeats 3 --pin-memory --output results/cache_study_new_run
```

La salida debe ser nueva. El comando entrena el predictor en sesiones `train`,
lo congela y evalúa únicamente sesiones `test`, con orden de políticas aleatorio
reproducible. Rechaza IDs repetidos y prompts idénticos normalizados entre splits;
eso no detecta paráfrasis ni exposición durante el entrenamiento del LLM.

Se comparan residencia completa (control), LRU puro, popularidad aprendida,
listas semánticas del mapeo y predictor aprendido. Las cuatro cachés tienen igual
capacidad; las tres políticas de precarga tienen igual presupuesto máximo. Las
listas semánticas proceden de la calibración externa indicada, mientras popularidad
y predictor usan exactamente las mismas sesiones train. Registrar esta diferencia.

El predictor aprende presencia de expertos por turno condicionada a etiquetas
actuales y del turno anterior; combina ambas tasas suavizadas hacia popularidad
global. No cambia pesos ni el router E. No es una red neuronal ni un predictor de
la próxima respuesta. `rank([], etiquetas_anteriores)` permite predecir demanda
sin conocer el nuevo tema; el estudio usa además las etiquetas del prompt actual.

`examples/sessions.jsonl` contiene sesiones **redactadas para diagnóstico**, no
conversaciones reales recogidas de usuarios. Cada línea lleva `id`, `split`,
`scenario` y una lista `prompts`. El contexto concatena prompts anteriores y se
trunca a `--seq-len`; se evalúa un forward de contexto por turno, sin generar
respuestas. Calentamiento fuera de medición y caché vacía al comenzar cada sesión.
El clasificador se ejecuta también en controles para igualar su coste; no es
un benchmark mínimo del modelo sin clasificador. Las comprobaciones y escritura
de informes están fuera de los tiempos. CPU solo representa residencia lógica.

Salidas: `predictor.json`, `training_observations.json`, `sessions.jsonl`,
`turns.jsonl`, `report.json` y `SUMMARY.md`. Incluyen hash de pesos/arquitectura,
split, configuración, latencias por turno, precargas, bytes H2D, hit rate y
comparación de todas las rutas, argmax y 64 logits por token. Los percentiles
combinan turnos y repeticiones; estas últimas no son nuevas sesiones independientes.

Uso del predictor en conversación con el modelo original:

```powershell
python -m asi chat existing --checkpoint results/expert_audit_training/model_04750.pt --expert-labels results/posthoc_04750/expert_labels.json --cache-strategy learned --usage-predictor results/cache_study_new_run/predictor.json --pin-memory --session-context --output results/chat_learned.json
```

Para actualizarlo después de cada turno, añadir `--learn-online --predictor-output
results/chat_predictor_updated.json`. Se exige una salida distinta del predictor
original. Omitirlo mantiene el predictor congelado. Los datos del chat generativo
difieren de las sesiones de diagnóstico; evaluar ese cambio antes de extrapolar.
`--cache-strategy lru` desactiva precarga y prioridad temática; `popularity` usa
el predictor sin sus variables contextuales. `restrict` solo permite estrategia
semántica y no aprendizaje online, porque cambiaría la demanda usada como objetivo.

### Dos expertos por capa y swapping por token — 1.3, 1.5, 2.2

```powershell
python -m asi decode-study --checkpoint results/expert_audit_training/model_04750.pt --expert-labels results/posthoc_04750/expert_labels.json --usage-predictor results/cache_study_04750_v1/predictor.json --experts-per-layer 2 --max-new-tokens 16 --seq-len 64 --repeats 3 --pin-memory --output results/decode_two_experts_new_run
```

El estudio mantiene el router nativo: cuando su pareja cambia, carga los expertos
nuevos. No fija dos expertos para toda la respuesta ni fuerza al router a elegir
los que ya estén residentes. El control carga todos; cada caché admite como máximo
dos enrutados por capa (22 en las 11 capas MoE de este checkpoint). El prefill
serial impide que la unión de selecciones de varios tokens exceda el presupuesto.
Los expertos compartidos y demás componentes siguen en GPU.

Se comprueba primero el adaptador contra el forward original en FP32 sobre un
prefijo y todo el vocabulario. Después se comparan salidas greedy, rutas y logits
muestreados de cada caché contra un baseline con idéntico decode incremental.
No se exige igualdad bit a bit entre operaciones batched y seriales distintas.

`turns.jsonl` registra cada paso: fase, token, expertos, hits/misses, transferencias
y residencia por capa. El informe separa prefill, precargas y decode, e incluye
fracción de pasos sin cargas. El primer token generado sale del último prefill;
16 tokens de salida normalmente requieren 15 forwards de decode. La caché de
expertos persiste entre turnos; el KV se reconstruye al comenzar cada turno sobre
la historia retenida, que incluye respuestas del baseline. No hay aprendizaje en
test. Los casos siguen siendo diagnósticos redactados, no conversaciones externas.

Chat con el mismo límite:

```powershell
python -m asi chat existing --checkpoint results/expert_audit_training/model_04750.pt --expert-labels results/posthoc_04750/expert_labels.json --experts-per-layer 2 --cache-strategy lru --session-context --max-new-tokens 32 --output results/chat_two_experts.json
```

### Carga desde disco con RAM limitada — 1.3 y 2.1

```powershell
python -m asi export-store --checkpoint results/expert_audit_training/model_04750.pt --output results/expert_store_new
python -m asi chat existing --expert-store results/expert_store_new --expert-labels results/posthoc_04750/expert_labels.json --experts-per-layer 2 --ram-expert-mib 4 --cache-strategy lru --max-new-tokens 16 --prompt "Explain how a Python generator works." --output results/chat_disk.json
```

El exportador crea un backbone y un archivo sin comprimir por experto, con hashes
y configuración. Usa el checkpoint mapeado; es un paso previo, sin prometer coste
constante de RSS durante conversión. El cargador de inferencia construye la
arquitectura en `meta`, carga solo el backbone y crea marcadores vacíos para los
expertos. No necesita abrir el checkpoint original después de exportar, pero sí
su archivo de arquitectura. El modelo debe conectarse a la caché antes del forward.

La caché de disco conserva tensores en RAM hasta `--ram-expert-mib`, con desalojo
LRU antes de leer otro experto. Cada experto debe caber individualmente. Las
copias frías de ejecución son marcadores vacíos y no retienen pesos fuera de ese
presupuesto. Este modo requiere CUDA y usa RAM pageable (la opción de pinning de
la caché completa no se aplica al almacén). Se verifican hashes en la primera
carga de cada shard; ese tráfico se registra aparte.

Los informes incluyen hits/misses de RAM, lecturas lógicas de archivos, bytes
H2D y máximo de tensores RAM retenidos. No confundir ese límite con RAM total:
backbone, buffers, tokenizer, clasificador, deserialización y caché de páginas
del sistema también consumen memoria. Tampoco se pueden equiparar lecturas
lógicas con I/O físico del SSD. No hay compresión ni ejecución distribuida.

### Exportación INT8 offline

```powershell
python -m asi quantize --checkpoint results/expert_audit_training/model_00009.pt --output results/expert_int8_trial
python -m unittest discover -s tests -v
```

La exportación INT8 guarda pesos y escalas; las cachés no ejecutan estos archivos
comprimidos. Los tests CUDA se omiten cuando no hay GPU.

## Scripts anteriores → comandos actuales

| Antes | Ahora |
|---|---|
| `segment_and_shard_fineweb.py` | `python -m asi data prepare` |
| `build_expert_pool_manifest.py` | `python -m asi data pools` |
| `inspect_specialized_fineweb.py` | `python -m asi data inspect` |
| `train_deepseek_v3_domain.py` | `python -m asi train` |
| `audit_experts.py` | `python -m asi audit` |
| `calibrate_existing.py` | `python -m asi posthoc calibrate` |
| `generate_deepseek_v3_domain.py` | `python -m asi chat domain` |
| `generate_existing.py` | `python -m asi chat existing` |
| `expert_quantization.py` | `python -m asi quantize` |

Las dos cachés se reúnen en `runtime/cache.py`, las trazas y calibración en
`analysis/experts.py` y ambos generadores en `runtime/generation.py`. Las guías
se consolidan en PLAN, USAGE y FINDINGS. No hay duplicados de compatibilidad.

Los resultados históricos conservan sus comandos originales como procedencia.
La copia anterior del código y guías está en `results/refactor_backup/source.zip`,
sin dataset ni checkpoints. No hay que descomprimirla para usar el proyecto.
