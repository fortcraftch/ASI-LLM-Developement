# Hoja de ruta principal del TFG

Acuerdo con Marc del 11/10/2026. Esta es la línea de trabajo vigente hasta
completar sus fases; los documentos anteriores conservan valor histórico.
Objetivo: medir memoria y transferencias evitadas, latencia y coste de calidad.
No presuponer recuperación, superioridad de una arquitectura ni causalidad de mapas.

## Estado y orden

| Fase | Estado | Trabajo y criterio de cierre | Paper |
|---|---|---|---|
| 1. Evidencias y protocolo | EN CURSO | Consolidar siete máscaras, revisar referencias y respuestas; reservar datos nuevos y fijar criterios antes de nuevas selecciones | 0, 2, 4.2, 5.3–5.6, 6 |
| 2. Selección y escala | Piloto de escala completado; controles pendientes | Misma máscara/pesos/entradas: escala original, renormalización y masa nativa; controles aleatorios, compartidos y por capa con igual memoria; uso y ablaciones además de mapas | 3.1, 5.2, 5.3 |
| 3. Recuperación controlada | Pendiente | Original/restringido/recuperado; router/LoRA/ambos con datos y actualizaciones iguales; ampliar categorías, semillas y longitud; reducir residentes solo después de validar | 0, 4.2, 5.4, 5.5 |
| 4. Modelos locales comparables | Recetas pareadas preparadas; banco recibido y sus ocho checkpoints verificados el 12/10 | Pareja nueva con arquitectura e inicialización iguales, mismos datos/orden/tokens/optimizador/calendario; evaluar banco con selección conocida y predicha | 4.1, 5.1 |
| 5. Calidad y rendimiento amplios | Banco y ASI: 19 tareas completas; DeepSeek en ejecución | 19 tareas completas, instrucciones verificables, programación aislada y matemáticas; prefill/TTFT/decode/total/memorias/tráfico, calentamiento y orden alternado | 4.3, 5.1 |
| 6. Sistema dinámico completo | Pendiente | Conversaciones, multilabel, clasificador, selección por hardware/calidad, hit/miss, frío/caliente, RAM variable y anticipación; medir coste total | 3.2, 6 |
| 7. Paper final | Pendiente | Integrar evidencias, apéndices, resultados negativos, estado del arte, resumen y conclusiones; sincronizar/compilar/verificar Overleaf | Introducción, 2, 6, resumen y conclusiones |

Las fases 1–3 son la prioridad inmediata. La preparación de la pareja equivalente
y de las evaluaciones del banco puede avanzar en paralelo, sin interferir con
los entrenamientos. No se ha confirmado acceso remoto al proceso de homelab.

## Decisiones fijadas

- 12/10: banco evaluado en las 19 tareas locales, 66.832 preguntas entre dos
  bloques completados. Pesos, evaluador, manifiesto, lote 4/512, semilla y GPU
  coinciden; BF16 se registra con dos nombres equivalentes por evolución del
  esquema. El paper local incorpora tabla completa, figura de comparación sobre
  100 preguntas, control de memoria y precisión, y 31 fuentes con SHA256.
  Figura y tabla revisadas visualmente. Sincronización/compilación en Overleaf
  verificadas: 25 páginas, cero errores/advertencias; evidencia en
  `paper/evidence/overleaf_bank_sync_20261012.json`. Pendientes suite completa
  de las otras arquitecturas y control de categorías anotadas.

- 12/10: comprobador existente de GPT-2 confirmó acceso a todos los datasets
  de la suite. Iniciada evaluación completa de las 13 tareas restantes del
  banco, misma residencia selected y lotes 4/512 BF16, sin repetir las seis
  completadas. Salida `results/router_124m_audit_20261012/eval_full_remaining13.json`.
  No considerar terminada la suite hasta validar `completed`, tareas, tamaños,
  huellas y compatibilidad con el primer bloque.

- 12/10: control FP32 de PIQA completado con TF32 desactivado. Lotes 4/512 y
  8/2048: 54/100 ambos, cero cambios de decisión, delta NLL máximo 1,91e-6;
  frente a dos cambios y 0,0320 en BF16. Evidencia de sensibilidad a precisión,
  no efecto de residencia. Ocho tests del motor pasan; contexto de precisión
  aplicable también a generación. Descarga de datasets restantes iniciada con
  el comprobador existente de GPT-2, sin cargar LLM ni cambiar resultados.

- 12/10: banco completo en seis splits finalizado (21.373 preguntas): HellaSwag
  normalizado 25,55%, LAMBADA 4,91%, PIQA 56,58%, ARC-Easy 38,95%, BoolQ
  39,45%, aritmética 0%. Pico CUDA 1061,38 MiB. No es aún suite completa de 19.
  Control PIQA instrumentado: 200 continuaciones/100 preguntas, NLL y decisiones
  idénticos entre all/selected con igual lote. Cambiar 4/512 a 8/2048 cambia 188
  puntuaciones (delta NLL medio máximo 0,0320) y dos decisiones; categorías
  idénticas. Evidencia `piqa_trace_comparison.json`; no identifica aún el kernel
  ni demuestra equivalencia universal. Corpus v3: seis reparaciones de código,
  27 aserciones sin ejecutar; 240 filas, 23 revisiones y máximo 275 tokens.

- 12/10: borrador de recuperación v2 generado sin cambiar sus 240 textos.
  Invalida objetivos de evaluación heredados en las 17 correcciones y conserva
  los antiguos como procedencia; seis tests aprobados. No autoriza entrenamiento.
  El reintento del banco completó HellaSwag (10.042 casos, accuracy_norm
  25,5527%) y pasó a LAMBADA; las otras tareas siguen pendientes de finalización.

- 12/10: instrumentación opcional `bank-eval --trace-dir` incorporada al motor
  compartido: tokens de entrada, NLL por continuación y salida generada con
  huellas de archivos e implementación. Siete tests del motor aprobados. Aún
  falta ejecutar los controles CUDA con esta instrumentación; el reintento
  completo anterior sigue activo y no se reinicia para añadirla.

- 12/10, control de residencia del banco: ensayo completo inicial detenido por
  OOM en HellaSwag, informe preservado. Ensayos pareados de 100 casos con lotes
  4/512: mismas categorías y métricas agregadas entre ocho modelos residentes y
  uno; pesos LLM GPU 472,92 frente a 59,11 MiB, pico total 1432,39 frente a
  948,76 MiB. Tres tests de política aprobados y ejecución CUDA real completada.
  No hay comparación por predicción individual todavía. PIQA cambia de 54% con
  lotes 8/2048 a 56% con 4/512 en ambos modos: pendiente analizar sensibilidad
  numérica; no atribuirlo a residencia. Evidencia `residency_comparison.json` en
  la carpeta de auditoría. Reintento completo iniciado con residencia selected
  y 4/512, salida `eval_full_six_selected_retry.json`; verificar finalización.

- 12/10: banco recibido y ocho checkpoints finales verificados (123.972.608
  parámetros totales). Evaluación con categorías predichas completada sobre
  100 ejemplos de cada una de seis tareas: HellaSwag normalizado 28%, LAMBADA
  4%, PIQA 54%, ARC-Easy 36%, BoolQ 36%, aritmética 0%. Huellas de datasets,
  evaluador y tamaños coinciden con ASI y DeepSeek final; arquitecturas y
  exposición no son equivalentes. Evidencia y figura en
  `results/router_124m_audit_20261012/comparison_100.json` y `accuracy_100.png`.
  Iniciada ejecución completa de las seis tareas (`eval_full_six.json`);
  comprobar `completed` antes de usarla. Pendientes suite de 19, categorías
  anotadas y medición de residencia limitada. No concluir que el routing causa
  las diferencias ni trasladar los tiempos de este ensayo al paper como speedup.

- El DeepSeek final 19072 y el categorizado actual son referencias exploratorias;
  no constituyen una comparación causal de arquitectura equivalente.
- La pareja nueva propuesta usa **16 expertos, 2 activos, 8 pools de 2** en
  ambos modelos; libre frente a restricción por pool. Con 8 expertos y 8 pools
  disjuntos no caben dos expertos por pool. Congelar recetas antes de entrenar.
- Dentro de un pool de 2 con 2 activos, el router puede aprender pesos, pero no
  elegir otro subconjunto. Estudiar elección requiere candidatos > activos,
  como una ablación adicional y con su propio control de presupuesto.
- El banco independiente conserva ocho LLM densas de 15.496.576 parámetros,
  123.972.608 en total. Clasificador aparte. Evaluar categoría conocida y predicha.
- No abrir un router no entrenado y presentarlo como un baseline libre entrenado.
- Separar ausencia de pérdida al cambiar residencia **sin cambiar cómputo**, de
  degradación al restringir las elecciones frente al modelo original.
- La preparación sí puede transferir expertos; cero cargas durante una respuesta
  no implica cero tráfico de extremo a extremo.
- Un resultado negativo se conserva. Si la recuperación funciona, 5.6 pasa a
  diagnóstico/resolución o apéndice; no desaparece la evidencia.

## Protocolo de decisión para las fases 1–3

1. Los seis ítems de máscaras y los tests inspeccionados previamente son
   diagnósticos. No seleccionar nuevas máscaras/adaptaciones con ellos.
2. Construir registro de todos los IDs ya usados, separación por temas/plantillas,
   duplicados y referencias defectuosas antes de congelar nuevos dev/test.
3. Fijar semilla, prompts, EOS, longitudes, pesos, máscaras y presupuesto de GPU
   antes de ejecutar. Evitar producto cartesiano innecesario: filtrar en dev.
4. Registrar por separado: respuesta correcta, razonamiento, instrucciones,
   formato, repetición, EOS y pérdida de referencia. Coincidir con un número o
   palabra no demuestra corrección completa. No ejecutar código generado sin aislamiento.
5. Definir márgenes de degradación admisible y tamaño de muestra antes del test
   final. Umbrales de trabajo fijados el 11/10 en `RECOVERY_V3.md`; datos y test aún sin congelar.
6. Avanzar una recuperación solo si mejora calidad libre respecto a la restricción
   sin adaptar y mantiene el presupuesto/residencia. CE por sí sola no basta.
7. Detener/reestructurar una línea si mejora CE sin mejora de utilidad en dev;
   documentar el resultado y no expandir pesos o datos indefinidamente.
8. Guardar hashes, informes por intento, límites y cambios de protocolo. Actualizar
   el paper al cerrar experimentos, sin reemplazar métricas históricas silenciosamente.

## Registro de avance

- 11/10: hoja de ruta fijada como principal; objetivo persistente creado.
- Fase 1: siete máscaras completas (96 calibración, 42 respuestas); DeepSeek
  final y sincronización del paper completados previamente.
- Revisión inicial detecta referencia ambigua/defectuosa `mbpp_test_135` y
  coincidencias numéricas con razonamiento erróneo. Ver
  [revisión inicial](MASK_REVIEW_20261011.md). Falta revisión completa anotada,
  registro exhaustivo de contaminación y congelar nuevos conjuntos.

Próxima acción: completar la auditoría de referencias y evaluación semántica,
emitir análisis de sensibilidad excluyendo ítems defectuosos sin borrar el
resultado original, y preparar el protocolo pareado de escala de routing.

- 11/10, continuación: implementado `python -m asi recovery-data-audit`.
  Inventariados 97 JSON históricos sobre 4.093 ejemplos; incluye duplicados de
  prompt y grupos de plantillas. Los IDs explícitamente ausentes del profesor
  no se consideran ejecutados. Quedan provisionalmente 118 candidatos dev y
  128 test, **no congelados ni certificados independientes**.
- Auditoría no cerrada: `results/recovery_study_v3/teacher/manifest.json`
  contiene bytes nulos y no se puede analizar. El informe previo ya señalaba
  corrupción; falta contrastar los archivos y su alcance, sin borrar evidencia.
  El auditor termina con código 1 y registra el problema, en lugar de aprobarlo.
- Sensibilidad sin la referencia MBPP ambigua completada y registrada en
  `MASK_REVIEW_20261011.md`; no se usa para elegir la próxima máscara.

Próxima acción concreta: reconciliar el manifiesto corrupto con el inventario
del profesor, completar anotaciones y proponer nuevos splits por temas antes
de congelar la comparación de escala de routing.

- 11/10, reconciliación: verificados los SHA-256 de los 424 shards antiguos
  (6,59 GiB). Todos coinciden con el inventario del 04/10, sin archivos adicionales
  ni ausentes. El manifiesto conserva el mismo hash corrupto. Esto no acredita
  la identidad del profesor ni autoriza reutilizar sus probabilidades; siguen
  en cuarentena. Evidencia: `results/recovery_data_audit_20261011/teacher_reconciliation.json`.
- Fase 2 preparada parcialmente: `restricted_unit_mass` incorporada al runtime,
  separada de `restricted` y `restricted_native_mass`. Prueba de gate sobre
  estados idénticos verifica fórmulas, selecciones iguales y ausencia de cargas.
  [Protocolo](ROUTING_SCALE_PROTOCOL.md); inferencia real pendiente.

Siguiente trabajo: documentar el alcance de la cuarentena sin certificar el
manifiesto corrupto, completar revisión de referencias y congelar desarrollo
revisado para el experimento de escala. No usar los seis diagnósticos para selección.

- 11/10, escala: 15 pruebas del runtime aprobadas. Preparado `routing-scale`
  con cuatro brazos (original, restringido, masa uno, masa nativa), pesos
  congelados y máscara old32 idéntica. Nueve referencias de desarrollo revisadas
  antes de ejecutar; plan versionado en `configs/routing_scale_dev_v1.json`.
  Prueba adicional comprueba rechazo de fuentes modificadas, referencias
  cambiadas, IDs históricos y selección de test. No es un test confirmatorio.
- La cuarentena del profesor no bloquea este piloto sin profesor: no usa sus
  probabilidades, excluye todos sus IDs identificados y declara explícitamente
  la incertidumbre por el manifiesto corrupto. La auditoría histórica permanece
  incompleta; no se la convierte artificialmente en una aprobación.

Siguiente acción: ejecutar y revisar las 36 respuestas de escala; continuar la
anotación de las 42 respuestas históricas y diseñar el test confirmatorio, sin
elegir variantes con sus resultados. Mantener la incorporación del modelo 3
condicionada a que termine su entrenamiento.

- 11/10: anotadas las 42 respuestas históricas en
  `results/mask_intervention_20261010/analysis/qualitative_review_20261011.json`.
  Rúbrica separada y huellas por texto; revisión del asistente, no ciega, pendiente
  de adjudicación humana. No se ejecuta código generado ni se atribuye precisión
  funcional a mera sintaxis o coincidencia de palabras/números.
- Piloto de escala iniciado en `results/routing_scale_dev_20261011`.
  Se guarda por respuesta; consultar el `report.json` y los logs para conocer
  progreso real. No confundir «iniciado» con «completado».
- Detectados candidatos con referencias ambiguas y separación temática por
  palabras demasiado amplia. Antes del test confirmatorio, revisar semánticamente
  los temas además de excluir IDs/plantillas. Ver `MASK_REVIEW_20261011.md`.

- 11/10, escala completada: 36/36 respuestas, cuatro brazos × nueve preguntas.
  Verificados prompts idénticos y cero cargas durante las 27 respuestas
  restringidas. CE original 1,922; restringido 2,359; masa uno 7,767; masa nativa
  2,422. Gráficas y salidas en el informe de escala. Masa uno no pasa a ser
  política por defecto; falta revisar semántica completa antes de cerrar fase 2.
- Fase 4: preparadas tres recetas equivalentes de 122.325.504 parámetros
  (libre aprendido, pool aprendido, pool uniforme). Normalización seleccionada
  explícita desde cero, mismo presupuesto y datos. Smoke CPU verifica pesos
  iniciales y lotes idénticos, y reanudación equivalente. `--pairing-audit` y
  `training-pairing-report` verifican evidencia real; no se ha iniciado ningún
  entrenamiento de 124M ni se ha interrumpido homelab. Ver `MATCHED_MOE_124M.md`.

Próximas acciones: revisar semánticamente las 36 respuestas, integrar el resultado
negativo de escala en el paper con sus límites y cerrar criterios/datos para la
siguiente recuperación. El test confirmatorio, entrenamientos pareados y modelo
3 siguen pendientes; no declararlos completados por estas pruebas de preparación.

- 11/10: revisión inicial de las 36 salidas completada y guardada con SHA por
  texto. Todos los textos de masa uno resultan fragmentarios en la revisión del
  asistente. Se separan código plausible, ejemplos incorrectos, errores de
  explicación y formato; falta adjudicación humana, no se calcula precisión
  semántica validada con estas anotaciones.
- Paper local actualizado con tabla, figura y límites del piloto de escala.
  Evidencias regeneradas desde 20 informes; figura inspeccionada. **Pendiente
  sincronizar, compilar y exportar desde Overleaf**; el PDF anterior no contiene
  esta subsección. No hay nuevos entrenamientos en ejecución por este avance.

Siguiente acción: sincronizar y verificar esta actualización del paper, después
cerrar el conjunto y los criterios de la siguiente recuperación. Mantener el
alcance completo de las siete fases: estos avances no las dan por terminadas.

- 11/10: piloto de escala sincronizado con Overleaf, compilado sin errores ni
  advertencias y exportado (22 páginas). Tabla y figura de páginas 18–19 revisadas.
  Conservado el PDF previo; huellas en `paper/evidence/overleaf_scale_sync_20261011.json`.
  Siguiente prioridad: protocolo y datos revisados para recuperación controlada.

- 11/10: siguiente protocolo incorporado a `RECOVERY_V3.md`, sin duplicar una
  guía nueva. Tres ajustes CE con el mismo control 32/8; utilidad completa como
  métrica principal, selección solo en desarrollo, comparación pareada y margen
  práctico explícito. Objetivos 90 dev/300 test pendientes de revisión y congelación.
  Ampliada la cuarentena a las dos referencias ambiguas ya identificadas; se
  conserva el inventario histórico con su hash, sin reescribirlo.

- Auditoría posterior al piloto guardada separadamente en
  `results/recovery_data_audit_post_scale_20261011`: 101 JSON inventariados,
  107 candidatos dev y 128 test, ninguno certificado. Biología solo aporta
  26 dev, por debajo del objetivo de 30; se necesitan nuevas referencias.
  El manifiesto corrupto mantiene `audit_complete=false`; cuatro pruebas del
  auditor aprobadas. Próxima acción: revisar las referencias de entrenamiento y
  preparar preguntas adicionales con separación semántica, antes de congelar datos.

- 11/10, revisión de datos: 24/240 referencias de entrenamiento anotadas con
  hashes en `results/recovery_reference_review_20261011/report.json`; 216 pendientes.
  Se detecta contexto visual ausente, respuestas indirectas y enunciados de código
  insuficientes. No se atribuye causalmente la degeneración a estos defectos.
- El auditor ahora excluye duplicados entre splits aunque no estén en resultados
  históricos, y rechaza sobrescribir cualquier directorio de auditoría existente.
  Cinco pruebas aprobadas, incluidos ambos casos. Nueva auditoría guardada en
  `results/recovery_data_audit_guarded_20261011`, con error explícito esperado por
  el manifiesto histórico corrupto; no se convierte en aprobación ni se modifica.

- 12/10: revisión ampliada a 96/240 referencias, completada la lectura de las
  80 de matemáticas. Dos explicaciones defectuosas con número final correcto,
  siete enunciados a aclarar, tres defectos menores y cinco supuestos registrados.
  Propuestas de respuesta para los dos errores guardadas separadamente; no se
  modifica entrenamiento histórico ni se reutilizará profesor de textos cambiados.
  Quedan 72 referencias de código y 72 de biología por leer, además de resolver
  observaciones y preparar desarrollo/test. No se inicia entrenamiento con datos
  todavía pendientes de revisión.

- 12/10: revisión ampliada a 168/240; completada lectura estática de las 80
  referencias de programación. Seis errores de código claros, cuatro defectos
  de borde/dominio y un mensaje explicativo incorrecto; casos ambiguos separados.
  Añadidos a cuarentena los defectos identificados para futura selección,
  preservando corpus y profesor históricos. Guardada instantánea de la revisión
  anterior (96) antes de actualizar. AST usado para comprobar precedencia, sin
  ejecutar referencias. Quedan 72 de biología y resolución de anotaciones.

- 12/10: primera lectura terminada, 240/240 referencias anotadas. Biología incluye
  explicaciones truncadas, soporte no relacionado y categorías incorrectas.
  Tres propuestas de reemplazo contrastadas con fuentes y guardadas sin aplicar;
  conservan exposición del ID padre y requieren tokens/profesor nuevos. No hay
  aprobación global de datos, verificación humana ni prueba causal sobre errores
  del modelo. Próximo paso: resolver anotaciones en una versión derivada con
  contratos de entrada explícitos y verificar código aisladamente; después
  preparar los dev/test nuevos y ejecutar el protocolo de recuperación.

- 12/10: preparado comprobador funcional de referencias con contenedor Linux,
  sin red/montajes y con límites. Ocho casos fijados por hashes; cinco pruebas
  del ejecutor aprobadas. Dry-run completado, intento real detenido explícitamente
  por Docker ausente (WSL devuelve acceso denegado). No se han ejecutado las
  referencias ni se presenta la simulación como validación del aislamiento.
  La revisión factual y la construcción de datos derivados pueden seguir aquí;
  la comprobación funcional real requiere un host con Docker Linux disponible.

- 12/10: creado borrador derivado de 240 referencias con 17 correcciones, padres
  y exposición conservados. Constructor integrado en el módulo de corpus;
  entrenamiento rechaza borradores no aprobados y revisiones train-only en test.
  Seis pruebas de auditoría/revisión y nueve de recuperación aprobadas.
  Tokenización real: 240/240 bajo 384, máximo 275 con EOS. No hay entrenamiento
  ni profesor nuevo. WSL consultado sin la restricción de sesión confirma cero
  distribuciones instaladas; sigue pendiente ejecución funcional aislada.

- 12/10: revalidadas las dos ejecuciones completas del banco (19 tareas,
  66.832 ejemplos; hashes de pesos/evaluador/manifiesto coincidentes). Informe:
  `results/router_124m_audit_20261012/full_suite_validation.json`. Iniciada
  suite completa ASI 37595 classified, lotes 4/512, todos los pesos residentes
  pero una sola pool activa; salida `results/full_suite_20261012/asi_classified.json`.
  Ejecución pendiente de terminar y validar; no incorporar métricas parciales
  a conclusiones. Después corresponde DeepSeek 19072 con las mismas tareas.
  Los tiempos no constituyen comparación controlada de velocidad: hay aplicaciones
  de escritorio activas y las implementaciones usan distintas cachés de generación.

- 12/10: ASI classified terminó 19 tareas, 66.832 ejemplos. Huellas de
  datasets, evaluador y recuentos coinciden con el banco; comparación en
  `results/full_suite_20261012/asi_bank_comparison.json`. DeepSeek 19072
  iniciado con suite completa y lotes 4/512; pendiente de terminar/verificar.
- 12/10: verificados mediante aritmética racional los 14 objetivos numéricos
  de referencias corregidas del borrador v3. Propuestas con hashes en
  `results/recovery_reference_review_20261011/numeric_targets_v3_review.json`;
  no aplicadas al corpus ni aprobación global de entrenamiento.

- 12/10: comparación completa protegida en `compare_full_suite` del módulo
  de informes existente: exige finalización, cobertura, mismos datasets y
  evaluador, recuentos y definición de accuracy; rechaza bloques solapados
  y variantes multimodo ambiguas. Cinco pruebas aprobadas y validación real
  ASI/banco correcta. DeepSeek sigue en ejecución, aún no incorporado.
  Preparado ZIP autocontenido para las seis referencias de código; Marc
  confirmó que Docker todavía no se había ejecutado.

- 12/10: v4 conserva 26 correcciones y 14 objetivos numéricos contrastados.
  Diez propuestas adicionales (cuatro de código y seis de biología) validadas
  junto con v4 en memoria: 36 padres distintos, 240 filas train-only y límite
  real de 384 tokens. No se escribió otro corpus ni se alteró el ZIP v3.
  `pending_proposals_validation.json` registra huellas y alcance; todavía
  no sustituye pruebas funcionales ni revisión factual de todas las filas.

- 12/10: figura completa ASI/banco generada y revisada visualmente en
  `results/full_suite_20261012/accuracy_full.png` y PDF. Su constructor
  `build_comparison.py` reutiliza la validación de suites y registra las
  fuentes por SHA256. Añadirá DeepSeek solo con informe completado y
  comparable. ASI no gana en todas las tareas: banco mayor en WSC,
  TruthfulQA local y marginalmente OpenBookQA; aritmética empata en cero.
  No se han inferido significación estadística ni ventajas causales.

## Routing freedom and paper organization (13/10/2026)

The requested control is quality under E top-2 across all 16 experts, not
all-resident versus selected-resident memory. The mistaken residency run was
stopped. The full ASI `all_router` evaluation is running with batches 4/512;
its protocol and source hashes are in `results/routing_freedom_20261013/protocol.json`.
The completed classified baseline is reused. E was bypassed by uniform-pool
training, so this is an untrained-gate ablation, not learned routing. Selection
and mixture weights both change. No new accuracy result is claimed yet.

E1 in the local paper now has a common comparison and three model-specific
subsections (ASI, free-routing DeepSeek, independent bank). Labels, figures and
tables are preserved. This local reorganization has not yet been compiled or
synchronized to Overleaf; the existing PDF still reflects the previous version.

- 13/10: Gate-only ASI pilot queued after untrained-E full suite. Existing
  OASST pilot train/validation; 200 updates, LR 1e-4, accumulation 4,
  validation every 50, best selection includes step zero. All other parameters
  frozen. CPU gradient/update test passed. Queue verifies saved frozen weights
  before full19 evaluation and validates dataset fingerprints/counts for the
  three-way comparison. Status: results/routing_freedom_20261013/queue_status.json.
  Paper source documents planned protocol only; no new results or Overleaf sync claimed.

- 13/10: Prepared configs/bank_active_matched_v1: eight dense 93,850,112-parameter specialists matching ASI participating inventory; 750,800,896 total. CPU recipe dry-run passed. Homelab training pending; separate from historical 124M-total bank. Local paper clarification added, not yet synced.

## Siguiente investigación autorizada: Colibri (13/10/2026)

Después de terminar la comparación ASI (pareja categorizada, E sin entrenar,
E ajustada) y actualizar el paper, investigar Colibri como referencia MoE
frente a AirLLM. Enlace proporcionado por Marc: https://lnkd.in/p/eFdVcpfc
El acceso web directo al enlace corto falló; todavía no se ha identificado
con certeza el repositorio o paper. Resolver el destino antes de atribuirle
características. Revisar arquitectura, selección de expertos, jerarquía de
memoria, transferencias, precisión, licencia y compatibilidad. Proponer una
comparación con mismo checkpoint, hardware, prompts y protocolo; distinguir
métodos que conservan routing original de restricciones que cambian calidad.
Valorar reutilizar componentes solo tras verificar aplicabilidad. No iniciar
otra carga GPU mientras sigue la evaluación actual.

- Referencia directa de Colibri confirmada por el enlace de Marc:
  https://github.com/JustVugg/colibri (destino del parámetro url del enlace
  de LinkedIn). Usar esta URL para la investigación posterior; todavía no
  se ha revisado su implementación ni verificado sus prestaciones.

- 13/10 20:00: ASI three-way full19 evaluation completed and provenance/count checks passed. Gate-only training completed 200 updates; only 11 gate matrices changed. Local paper table/figure/text now reflect mixed quality results; Overleaf synchronization and PDF verification still pending. Colibri follows that update.

- 13/10: Routing full-suite update synchronized to Overleaf; compilation 0 errors, 0 warnings, 8 underfull layout notices; 26-page PDF downloaded. Source, table, figure and bibliography synchronized. Proof: paper/evidence/overleaf_routing_sync_20261013.json. Colibri research is next.

- 13/10: User confirmed 1000 total updates (not epochs), gate-only OASST replay started in results/routing_freedom_1000_20261013. Same original initialization/data/seed/LR; optimizer state was absent from old checkpoint. Validate first200 trajectory, frozen weights and validation-selected checkpoint before benchmarking. This cannot establish joint-pretraining benefit. Colibri README now explicitly lists OLMoE CPU/Vulkan support and INT8 conversion; implementation/precision parity still to verify.

- Marc autoriza seguimiento E1000 y actualización del paper al terminar.
  13/10: siete tareas completas, BoolQ en marcha. Seguimiento reactivado cada
  30 minutos. Continuar entrenamiento solo si la comparación completa aporta
  mejoras claras y consistentes; evaluar magnitud y retrocesos. No hay umbral
  universal por número de parámetros. Antes de otra ampliación fijar presupuesto,
  guardar optimizador/RNG y seleccionar por validación con criterio de meseta
  (propuesta: <0,1% de mejora relativa durante tres comprobaciones).
  La suite ya inspeccionada debe etiquetarse exploratoria, no como test nuevo.

- 14/10: E1000 full19 completed; dataset/counts/evaluator/precision/GPU/batch/residency/KV match. Only gate weights changed. Results mixed, no automatic further training. Four-arm table and figure synchronized to Overleaf; compiled 0 errors/0 warnings, 6 typesetting notices; PDF downloaded and prior version preserved. Colibri/OLMoE remains next.
