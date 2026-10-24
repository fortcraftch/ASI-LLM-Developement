# Recovery V2: mapa del repositorio

Inspección inicial: 2026-10-17. No se reemplaza el backend histórico.

| Archivo real | Función y reutilización prevista |
|---|---|
| `asi/models/hf_olmoe.py` | `OlmoeExpertCache`, caché de expertos; `CategoryCalibration`. Validar equivalencia antes de reutilizar. |
| `asi/models/olmoe_storage.py` | `OlmoeDiskStore`, `load_disk_backed`: almacenamiento y carga con presupuestos de memoria. |
| `asi/experiments/public_moe.py` | Calibración, selección de tokens y `decode`; referencia de generación histórica. |
| `asi/experiments/recovery.py` | Objetivos, evaluación de máscaras, estado de adaptación y entrenamiento del router. |
| `asi/experiments/recovery_protocol.py` | Escritura atómica, referencias de profesor, batching y orden de entrenamiento. |
| `asi/experiments/routing_scale.py` | Ejecución reanudable de controles de routing y decodificación. |
| `asi/analysis/mask_intervention_report.py` | Comparación de controles; exige cobertura e identidad compatibles. |
| `paper/build_evidence.py` | Generador histórico de tablas y figuras. No modificar durante F0–F6. |
| `paper/evidence/asi_evidence.json` | Evidencia JSON con fecha `as_of`, mapa `sources_sha256` y resultados derivados. Conservar. |
| `tests/` | Suite unittest existente; ejecutar antes y después de cambios en código. |

Nuevo trabajo aislado en `recovery_v2/`; resultados en `results/v2/F0/`.
`recovery_v2/data/exposure_inventory.py` inventaría conservadoramente ejemplos
identificables en JSON históricos; no certifica ausencia de exposición ni crea splits.
`recovery_v2/data/prepare_candidates.py` amplía la auditoría a formatos tabulares,
textos y metadatos; excluye conservadoramente todos los splits test locales y
prepara candidatos, sin crear ni leer el test sellado.
`recovery_v2/data/cache_overlap.py` contrasta esos candidatos con todos los textos
de las cachés Arrow/Parquet, incluyendo campos anidados, sin asumir que descargar
un dataset equivale a evaluarlo. Es una exclusión conservadora adicional.
El mismo runner admite `--source`, `--output` y `--include-history` para
auditar BigCodeBench contra el registro histórico además de las cachés.
`recovery_v2/data/evaluation_sources.py` extrae repositorio/configuración/split
de los manifiestos históricos y vincula cachés con esa evidencia de exposición.
`recovery_v2/data/consolidate_inventory.py` exige escaneo completo y separa
coincidencias con evaluación registrada de coincidencias de procedencia incierta.
`recovery_v2/data/duplicate_groups.py` agrupa transitivamente candidatos por
coincidencias exactas, plantillas numéricas y 13-gramas; grupos no son splits.
`recovery_v2/data/scan_checkpoint.py` ofrece checkpoints atómicos por archivo,
con identidad de candidatos y huellas verificadas; conectado a `cache_overlap`
después de terminar el primer escaneo. El informe anterior conserva su evidencia.
`recovery_v2/data/stage_review.py` aplica exclusiones conservadoras y prepara
filas con referencias y categoría provisional para revisión; no asigna splits.
`recovery_v2/eval/sandbox_probe.py` verifica aislamiento Docker con pruebas
controladas; requiere una imagen local fijada por digest y no ejecuta datasets.
`recovery_v2/eval/reference_runner.py` ejecuta referencias en un contenedor por
tarea solo tras verificar el informe de aislamiento de la misma imagen; guarda
cada resultado y permite reanudar por hash del paquete.
`recovery_v2/eval/container/` define una imagen científica parcial con base y
dependencias principales fijadas; guardar ID de imagen y freeze de transitivas.
La ejecución de decodificación anterior queda conservada, pendiente de completar;
no constituye un test independiente para V2. El árbol ya contiene modificaciones
anteriores: no restaurarlas ni atribuirlas a este protocolo.

## Revisión editorial autorizada (2026-10-18)
Excepción explícita de Marc a la pausa F7: paper/asi_paper.tex consume tablas y figuras de paper/build_evidence.py. asi/analysis/local_generation_report.py resume tiempos; los registros de asi/experiments/local_generation.py guardan seconds, first_token_seconds y decode_tokens_per_second. results/project_overview_v1/datos_graficas.json conserva las parejas y transferencias de trazas históricas. La revisión no altera resultados históricos ni los usa para seleccionar V2.

Revisión 18/10 adicional: ../Creating-GPT-2-From-0/results/gpt2Base.json contiene 18 tareas completas (66.332 casos) sin hashes históricos de checkpoint/evaluador/datasets. paper/build_editorial.py incorpora la copia archivada como referencia histórica diferenciada; no usa SFT ni completa aritmética con el piloto de 100 casos.

Revisión solicitada tras edición manual de Marc: no sobrescribir tablas de suite ni manuscrito remoto sin recuperar su versión. build_editorial.py ofrecerá --swapping-only para regenerar solo la figura del mecanismo; la tabla suite calculada se emitirá como propuesta separada .generated.tex.

Integración remota: fuente manual descargada en paper/sources/overleaf_manual_20261018; tabla editorial_suite.tex preservada byte a byte. Solo retirar bloque de tabla E, referencia huérfana de figura y actualizar pie de intercambio sobre esa fuente.

## Diagnóstico de mecanismo DeepSeek final (2026-10-21)
- `asi/models/original.py`: carga del checkpoint nativo y adaptador incremental.
- `asi/experiments/decode_study.py`: `run_decode`, rutas y contadores por paso; `check_incremental`, paridad FP32.
- `asi/runtime/cache.py`: `NativeExpertSessionCache`, caché exacta y política fija uniforme.
- `recovery_v2/residency_probe.py`: réplica diagnóstica aislada, calibración textual separada y trazas reanudables; no selecciona recuperación ni accede al test sellado.
- `recovery_v2/plot_residency_probe.py`: figura de cambios de pareja y H2D desde las trazas verificadas del diagnóstico final DeepSeek.

## Réplica OLMoE (2026-10-21)
- `asi/models/hf_olmoe.py`: `OlmoeExpertCache.callbacks` entrega elecciones por capa; `snapshot` mide H2D y residencia.
- `asi/models/olmoe_storage.py`: carga BF16 con respaldo en disco y caché RAM limitada.
- `asi/experiments/public_moe.py`: `decode`, `input_ids`, `read_lock`; generación nativa y verificación SHA256.
- `recovery_v2/olmoe_residency_probe.py`: réplica reanudable por prompt con máscaras históricas congeladas, sin adaptación ni acceso al test sellado.
- `recovery_v2/plot_olmoe_residency.py`: valida nueve generaciones completas y separa cambios activos de transferencias; no publica tandas parciales.
- `recovery_v2/plot_olmoe_base.py`: extrae únicamente la traza nativa completada; representa expertos entrantes por capa/token sin comparar variantes.
- `recovery_v2/timed_olmoe_cache.py`: subclase aislada que mide obtención de pesos y copias H2D con eventos CUDA; no modifica el backend histórico.
- `recovery_v2/plot_olmoe_transfers.py`: compara tokens/rutas con la traza sin perfilado y representa H2D y tiempos de carga separados del cómputo.

El diagnóstico OLMoE admite `native_cache_experts` para contrastar 512/256 plazas globales sin cambiar routing, pesos ni datos. Comparación descriptiva de un prompt; no establece ley de escalado.
- `recovery_v2/compare_olmoe_cache.py`: exige misma identidad, tokens y rutas en 512/256 plazas; compara tráfico y tiempos como dos observaciones descriptivas.

## Banco denso de capacidad activa equivalente (2026-10-22)
- asi/experiments/train_bank.py: audit_main valida recetas congeladas, pasos finales, finitud y SHA256 de ocho checkpoints.
- asi/experiments/bank_eval.py: suite compartida GPT-2, selección por input y residencia all/selected; debe usar el presupuesto de la auditoría validada.
- asi/runtime/ensemble.py: IndependentModelRouter aplica límite agregado; BankResidency gestiona modelos completos.
- asi/experiments/local_generation.py: PROBES y timed_generate definen el protocolo histórico de generación (TTFT separado de decode).
- recovery_v2/bank_suite.py: cola reanudable por tarea para el banco nuevo, resultados aislados en F0; no selecciona checkpoints por resultados.

- recovery_v2/bank_generation.py: generación por categoría con PROBES históricos, TTFT/decode separados, RSS/NVML muestreados y reanudación por categoría.
- recovery_v2/bank_paper_update.py: valida evidencia del banco nuevo y genera filas/figura desde snapshot Overleaf, preservando estructura y tablas ajenas.

## Auditoría de inferencia densa (2026-10-22)
- asi/models/domain.py: MLA separa rama training SDPA de inferencia absorb/einsum; GPT admite logits_to_keep.
- ../Creating-GPT-2-From-0/eval.py: GPT denso usa SDPA causal y lm_head de prefijo completo.
- recovery_v2/dense_inference_audit.py: ablación aislada de atención y proyección final; verifica logits antes de medir, sin cambiar pesos ni código de producción.
`n- recovery_v2/dense_inference_report.py: valida la repetición aislada, comprueba tokens/logits y genera ablación con bootstrap por prompt, sin sustituir memoria histórica.

## Inferencia nivelada de cuatro familias (2026-10-23)
- recovery_v2/four_model_inference_audit.py: auditoría reanudable por familia/prompt; reutiliza mode/logits de dense_inference_audit, mantiene ASI classified y DeepSeek libre. Verifica equivalencia antes de aceptar optimizaciones; mide RSS/NVML/CUDA y H2D de expertos.
- asi/experiments/checkpoint_eval.py configure_mode y asi/runtime/cache.py ExpertCacheManager: preparación de pareja ASI fuera de región medida.
- recovery_v2/asi_precision_diagnostic.py: diagnóstico del primer token divergente, MLA FP32 y control completamente FP32; no modifica pesos ni resultados anteriores.
- recovery_v2/gpt2_attention_control.py: control emparejado GPT2 SDPA automático frente a backend matemático sin fusión; no cambia pesos ni protocolo de generación.
- recovery_v2/gpt2_control_report.py: valida control MATH y añade marcadores a figura histórica sin tocar tablas.
- recovery_v2/sdpa_four_model_audit.py: tanda SDPA de las cuatro familias con baseline emparejado; completa mediciones y separa estado de ejecución de decisión sobre divergencias antes de publicar.
- recovery_v2/sdpa_summary.py: resumen validado SDPA, bootstrap por prompt, diferencias de respuestas y figura local pendiente de decisión.
- recovery_v2/sdpa_paper_update.py: deriva tabla y figura de ventajas/costes desde summary.json SDPA; conserva tablas manuales usando fuente fresca Overleaf.
- recovery_v2/benchmark_overview.py: diferencias descriptivas en puntos porcentuales derivadas de la tabla editorial vigente, sin modificarla ni combinar protocolos de GPT2.
- recovery_v2/four_model_quality.py: diagnóstico nuevo de generación emparejada, reutiliza load_variant/timed_generate y mode SDPA; memoria memory_monitor, caché ASI ExpertCacheManager, salidas EOS conservadas y rúbrica previa por prompt. No test sellado ni selección de recuperación.
- recovery_v2/four_model_quality_report.py: bootstrap por prompt de proxies estructurales y repetición, exporta todas las respuestas sin ejecutar código; evaluación semántica separada.
- Preparación DeepSeek16: ../Creating-DeepSeek-V3-From-0/train_deepseek_v3.py ya acepta --config, top-k y reanudación; train_restart.py conserva checkpoints atómicos y reinicios CUDA. configs/deepseek16_v1 y recovery_v2/deepseek16_launcher.py parametrizan este entrenador sin modificarlo.

- Correcci�n de alcance: configs/deepseek16_v1/train_homelab.sh reutiliza base_16_learned.json, id�ntica a domain_16_uniform_pool salvo family/routing_mode. activate_training_pool expone las ocho pools (16 expertos) para family=base. La propuesta de expertos224/154M y su launcher se retiraron; sin ejecuci�n de entrenamiento.
