# Hoja de ruta principal del TFG

## Objetivo pendiente (revisado 16/10/2026)

Cerrar la investigación del compromiso calidad/memoria/latencia mediante
recuperación controlada, comparación de entrenamiento equivalente y evaluación
del sistema dinámico; consolidar proyecto y paper con evidencia reproducible.
Esta lista sustituye la lista operativa anterior: los hitos terminados se
conservan abajo como evidencia, no como tareas por repetir.

| Orden | Trabajo pendiente | Criterio de cierre |
|---|---|---|
| 1 | Validar corpus de recuperación y congelar train/dev/test | Ejecutar referencias de código en aislamiento (paquete v5 y resto), revisar biología/matemáticas y solapamientos; hashes, exclusiones y criterios fijados antes de seleccionar modelos |
| 2 | Completar controles de selección, escala y decodificación | Confirmar selección y escala con datos nuevos; aislar efecto de decodificación y comparar original/restringida sin ajuste/recuperada. Controles aleatorio y global exploratorios ya terminados, no repetir |
| 3 | Ejecutar recuperación controlada | Router, LoRA y ambos con datos/presupuesto iguales; original y restringido sin ajuste como controles; selección por dev, evaluación reservada, varias semillas/categorías y generación larga. Resultado negativo también cierra una hipótesis si el protocolo es suficiente |
| 4 | Completar comparación equivalente local | Entrenar/evaluar pareja MoE idéntica salvo routing; recibir banco de ocho densos de 93.850.112 parámetros, auditar y evaluar con selección conocida/predicha. Igual capacidad activa no implica iguales FLOPs/exposición |
| 5 | Ampliar calidad funcional y rendimiento | Instrucciones verificables, código aislado y matemáticas; completar controles de generación pendientes. Repeticiones, orden alternado, prefill/TTFT/decode/total, RAM/VRAM y tráfico con precisiones y protocolos comparables |
| 6 | Validar sistema dinámico de extremo a extremo | Clasificación real, multilabel, cambios de tema, split/joint, límites RAM/VRAM, caché y anticipación. Medir calidad y coste incluyendo clasificación/preparación; la prueba split/joint actual falla en calidad |
| 7 | Terminar limpieza y paper | Consolidar código/documentación realmente redundantes, tablas por protocolo y figuras por pregunta; integrar resultados nuevos, explicar límites, revisar referencias/abstract/conclusiones, sincronizar y verificar PDF final |

### Dependencias actuales

- Homelab: Marc indica que los modelos densos están entrenándose; no se ha verificado un proceso remoto desde aquí. No considerarlos terminados.
- Paso 1 aplazado hasta que termine el entrenamiento del paso 4, por indicación de Marc.
- Docker: Marc confirma que no ejecutó el paquete; falta `code_execution_v5.json`. Sus diez casos no validan todo el corpus. No lanzar recuperación sobre el borrador v6 aún.
- HumanEval: se han detectado tareas centrales compartidas con MBPP; cero coincidencias exactas no prueba independencia. Dev/test nuevo sin congelar.
- Paper vigente: `output/pdf/asi_paper_selection_20261016.pdf`, 28 páginas, compilado sin errores/warnings; 12 avisos de espaciado. Revisión visual general de las 28 páginas completada, con inspección ampliada de páginas 23–24.
- Colibri: comparación temporal de candidatos terminada. Adaptación conjunta y modelos mayores quedan condicionados a calidad recuperada y compatibilidad validada; no iniciar otro motor como sustituto de esos controles.

## Hitos retirados de la lista de pendientes

| Hecho | Evidencia / límite |
|---|---|
| Revisar y ejecutar las siete máscaras OLMoE | `results/mask_intervention_20261010/report.json`: completed, 42 registros. Diagnóstico exploratorio; no confirma recuperación ni causalidad |
| Controles aleatorio y global sin categorías | 18 + 6 respuestas completas; planes previos y comparaciones validadas. Paper sincronizado y revisado, 16/10; seis preguntas exploratorias, sin recuperación |
| Suites completas de ASI, DeepSeek 19072 y banco original | `results/full_suite_20261012/comparison.json`: sin tareas pendientes, 19 tareas/modelo. Banco original pequeño, no el nuevo de 93M por especialista |
| Evaluar E0, entrenar/evaluar E200 y E1000 | Informes routing_freedom y auditoría de pesos; resultados mixtos. No ampliar automáticamente entrenamiento |
| Pilotos iniciales de recuperación y escala, mapas y generaciones | Evidencia histórica integrada; fallo de calidad documentado. No repetirlos como si estuvieran pendientes |
| Piloto Colibri Vulkan libre/32/16 | Comparación temporal terminada; no calidad equivalente ni generalización a modelos mayores |
| Primera consolidación y sincronización del paper | Tabla conjunta, banco incluido en controles históricos, figuras repetidas retiradas; `paper/evidence/overleaf_consolidated_sync_20261015.json` |

## Limpieza y figuras que guiarán el cierre

Mantener una tabla principal por protocolo y un generador de evidencia. Retirar
código sin consumidores o reemplazado solo después de comprobar dependencias;
conservar resultados, configuraciones y fallos necesarios para reproducibilidad.
No borrar tests que protegen invariantes de routing, caché o reanudación.

Figuras prioritarias: (a) arquitectura y flujo completo, (b) total/residente/activo
por familia, (c) calidad frente a memoria/latencia con protocolos compatibles,
(d) recuperación por validación y calidad de generación, (e) transferencias y
cambios de categoría. Paneles compartidos para condiciones comparables; apéndice
para diagnósticos detallados. No añadir curvas pendientes ni inferir intervalos
estadísticos a partir de una sola ejecución.

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

- 14/10: Colibri/AirLLM pilot started sequentially on local OLMoE checkpoint. See docs/COLIBRI_AIRLLM.md. AirLLM BF16 vs Colibri INT8 CPU/Vulkan is explicitly precision-confounded, not an engine-only speed claim. Official Windows release hash verified. Results pending in results/colibri_airllm_20261014.

## Consolidación vigente — 14/10/2026, después de Colibri

- Colibri/AirLLM 16 y 64 tokens completados: precisiones y presupuestos distintos;
  no son prueba causal del motor. Informes en results/colibri_*_20261014.
- Comparación Vulkan categorizada completada: seis controles + 27 repeticiones,
  mismos pesos INT8, top8, tier de expertos 2 GiB. Medianas 32 candidatos:
  1,99–2,53x; 16: 4,07–5,57x. Máscaras/trazas y all64 equivalentes validados.
  Calidad no recuperada; beneficio restringido al régimen y prompts observados.
  La tanda CPU fue cancelada y eliminada por petición expresa; no usarla.
- Revisión actual de siete máscaras: results/recovery_restart_audit_20261014/audit.json.
  42 salidas y 96 calibraciones completas, IDs diagnóstico/calibración disjuntos,
  máscaras válidas en 16 capas y tres categorías. old32/recalibrated32/shared8+
  category24/first2_free_budget512 suman 512 candidatos por categoría;
  first2_free/uniform36/last2_free suman 576. No comparar como igual memoria
  grupos de 512 y 576. La auditoría no demuestra independencia semántica ni calidad.
- Recuperación sigue pendiente: inventario de referencias no aprobado para
  entrenar. No reutilizar preguntas ya inspeccionadas como test independiente;
  IDs con sufijo test usados históricamente para entrenamiento/calibración
  contaminan esos benchmarks para una futura afirmación de test intacto.
- Próximo trabajo: resolver referencias/código aislado, congelar nuevo dev/test
  y protocolo, entrenar y validar recuperación; después exportar/validar pesos
  recuperados en Colibri. Las fases 4–7 siguen abiertas, no se reduce el objetivo
  principal a recuperación. Integración de estos resultados en paper pendiente.
- 14/10, recuperación: integrado borrador v5 con las 40 correcciones propuestas
  (antes v4 tenía 26), manteniendo 240 filas train-only y procedencia. Validación
  por hashes y plantilla chat real OLMoE: máximo 276 tokens, dentro de 384.
  Diez referencias de código corregidas preparadas con aserciones y hashes del
  corpus v5; ninguna ejecutada. Archivos en results/recovery_restart_audit_20261014.
  ready_for_training sigue false: no confundir integración con aprobación factual,
  funcional ni congelación de evaluación independiente. Objetivos del profesor
  históricos no reutilizables para respuestas cambiadas.
- 14/10: paquete autocontenido homelab_code_v5.zip preparado y dry-run correcto
  para 10 referencias/50 aserciones del corpus v5. Incluye verificación SHA256,
  runner sin dependencia de torch y contenedores con red desactivada. Docker
  sigue ausente en PATH local; ejecución funcional pendiente en homelab.
  Devolver code_execution_v5.json; este resultado no aprueba las otras 70
  referencias de código ni el corpus completo. No se ha ejecutado código en host.
- 14/10: E7 añadido al paper local para Colibri Vulkan categorizado; constructor
  regenera figura desde 33 ejecuciones auditadas y comprueba hashes de logs/trazas.
  Evidencias reconstruidas (40 informes). Texto separa speedup exploratorio de
  calidad recuperada, startup y límites de caché/precisión. Figura inspeccionada.
  Sincronización Overleaf, compilación y nuevo PDF todavía pendientes; no afirmar
  que el documento remoto o PDF publicado ya contienen E7.
- 14/10: intento de sincronizar Overleaf impedido por revisión automática de
  acceso (límite de uso); acción no ejecutada, no buscar vías alternativas.
  E7 permanece local, pendiente de sincronización/compilación.
- Validador validate_execution_report añadido al runner de referencias: exige
  finalización, ejecución, hashes exactos de corpus/casos/runner enviado, imagen
  inmutable, cobertura sin duplicados y éxito de cada aserción agrupada. Seis tests
  aprobados, incluidos informes obsoletos/incompletos/duplicados/fallidos. Usar el
  hash del check.py del paquete enviado, no el runner actual modificado. Esto no
  constituye atestación independiente ni aprobación global del corpus.
- 14/10: auditoría de exposición v5 ejecutada: las 240 filas conservan exposición
  histórica identificable, incluidas las 40 revisiones. Ninguna es un holdout
  nuevo. Detector reutilizable exposure_conflicts cruza ID/padre, origen y
  plantilla normalizada y rechaza asignar train_only a evaluación. Dos pruebas
  pasan. Su cobertura es el corpus histórico proporcionado, no toda exposición
  del proyecto ni deduplicación semántica; ausencia de conflicto no aprueba test.
- 14/10: constructor recovery_corpus acepta --exclude-corpus y filtra exposición
  antes de las cuotas, registrando hashes y alcance. Tres tests pasan (incluido
  filtro previo a cuotas). Iniciada construcción de unexposed_candidates_v1.json
  desde fuentes locales, excluyendo corpus v3 y borrador v5. Es una selección
  candidata, no evaluación independiente aprobada: aún faltan otras exposiciones,
  revisión factual/funcional y congelación. No entrenar ni evaluar automáticamente.
- Resultado de esa construcción: fallo explícito por computer_science_software_engineering/train
  vacío tras filtros; no se escribió corpus. No relajar exclusiones para llenar
  cupos: hace falta otra fuente o un diseño de evaluación nueva separado del
  entrenamiento histórico. Este intento no produjo un holdout independiente.
- 14/10: diagnóstico MBPP con filas originales: 974 filas, 860 excluidas por
  identidad/plantilla histórica; quedan 114 antes de longitud (113 del tema test,
  una train y ninguna dev bajo la partición léxica). No es un simple error de
  cuotas: no permite crear tres particiones nuevas con esa política. Evidencia
  mbpp_exhaustion_audit.json. Matiz: pertenecer al corpus histórico es exclusión
  conservadora; no demuestra que toda fila haya recibido una actualización de
  gradiente. Evitar describir las 240/853 filas como entrenadas sin traza de uso.
- 14/10: DeepSeek19072 tenía 15 tareas completas sin errores, faltaban RACE,
  MMLU, GSM8K y TruthfulQA. Huella del evaluador vigente coincide con bloque
  histórico. Reanudación en archivo nuevo deepseek_19072_remaining4_20261014.json,
  batch4/tokens512, CUDA, limit0; no sobrescribe el bloque parcial. Antes de
  combinar exigir mismas huellas de checkpoint/arquitectura/datasets y protocolo;
  marcar bloque histórico como completo solo mediante nueva evidencia agregada.
- 14/10: intento online restante detenido deliberadamente tras reintentos de
  conexión fallidos. Cuatro datasets verificados offline (huellas/filas iguales
  a ASI); acceso a lockfiles de caché requirió elevación autorizada. Nueva tanda
  deepseek_19072_remaining4_offline_20261014.json, mismos parámetros, sin red;
  logs previos preservados. No asumir progreso del intento detenido.
- 14/10: añadido recover_native_suite para recuperar tareas completas de un
  bloque interrumpido sin editarlo: exige continuación terminada, identidad de
  checkpoint/arquitectura/evaluador/precisión/runtime y cobertura/filas/datasets
  iguales a referencia terminada. Test de recuperación y rechazos aprobado.
  build_comparison espera la continuación offline terminada para crear agregado
  con hashes; no presenta aún DeepSeek como suite completa. RACE ha llegado a
  19.541 peticiones de puntuación; carga offline ya superada.
- 14/10: regresión de componentes modificados: seis tests de revisión de corpus
  y cinco de comparación completa pasan. Proceso offline DeepSeek confirmado
  vivo (worker31328); RACE aún sin resultado final. No se ha iniciado otra carga
  GPU ni cambiado fuentes del evaluador durante la ejecución.
- 14/10: Colibri ejecutable y mapa actuales coinciden con hashes de lanzamiento.
  Inventario posterior de 57 archivos (fuentes, máscaras, shaders), revisiones de
  Colibri/Vulkan-Headers y aclaración de 2 GiB solo para tier guardados en
  reproducibility_inventory.json. No atesta retrospectivamente otros archivos;
  shaders precompilados siguen limitación. Corregida frase heredada del runner
  CPU en futuras procedencias; original del experimento conservado intacto.

### 2026-10-15: suite DeepSeek final completa
Continuacion offline terminada: RACE 32.23%, MMLU 27.90%, GSM8K 1.29%, TruthfulQA local 25.46%. build_comparison.py valida identidad y protocolo y genera deepseek_19072_aggregated.json sin modificar el parcial: 19 tareas, 66832 ejemplos, junto a ASI y banco. Paper local actualizado con figura full_suite_models; no se interpreta el agregado como una ejecucion temporal continua. Sincronizacion/compilacion Overleaf pendiente por bloqueo previo de auto-review; no se ha eludido.

### 2026-10-15: recepción verificable de pruebas de referencias

Se añadió `--validate-report` al comprobador de referencias. Valida el JSON
recibido sin ejecutar código ni necesitar Docker local: huellas del corpus,
casos, runner archivado e imagen, cobertura exacta y todos los casos aprobados.
No se encontró todavía un informe de ejecución de código en results.
Se mantienen pendientes las pruebas funcionales en homelab y un holdout nuevo;
este verificador no autoriza entrenamiento por sí solo. Seis tests pasan.

Desde la raíz del proyecto, cuando llegue el JSON:

```powershell
python -m asi.analysis.reference_code_check --validate-report results/recovery_restart_audit_20261014/homelab_code_v5/code_execution_v5.json --corpus results/recovery_restart_audit_20261014/homelab_code_v5/corpus.json --cases results/recovery_restart_audit_20261014/homelab_code_v5/cases.json --runner results/recovery_restart_audit_20261014/homelab_code_v5/check.py --output results/recovery_restart_audit_20261014/code_execution_v5_validated.json
```

El runner indicado es el archivado en el paquete enviado, no el código actual.
El paquete y sus huellas no se han modificado.

### 2026-10-15: comparacion completa documentada
El builder genera full_suite_table.tex con 19 tareas y recuentos comunes para ASI, DeepSeek final y banco. Paper incluye tabla y analisis descriptivo: DeepSeek supera ASI en 13 tareas; ASI en 5; empate en aritmetica. No se afirma significacion, equivalencia ni efecto causal de categorizacion. Evidencia regenerada (42 informes). Compilacion visual del documento completo pendiente; la figura ya fue inspeccionada.

### 2026-10-15: alcance real de la evaluación dinámica

Revisados `asi/runtime/adaptive.py`, `asi/experiments/adaptive.py` y
`tests/test_adaptive.py`. Ejecutados los siete tests: todos pasan, incluido
CUDA (sin omisión), promoción desde almacenamiento INT8 y ausencia de
transferencias durante decode en el modelo pequeño del test.

La calibración actual evalúa ventanas monodominio por separado, incluso al
permitir varios pools. Por tanto NO demuestra resolución de preguntas
multidominio, conversaciones ni calidad del planificador de subtareas.
El planificador requiere evidencia ligada al checkpoint y no elimina etiquetas
para encajar en memoria. Esto verifica mecanismos, no rendimiento extremo a
extremo del checkpoint entrenado. Los caches ya sincronizan preparación CUDA;
no se modificó esa medición sin necesidad.

Siguiente evaluación integral requerida: fijar escenarios de continuidad de
categoría, cambio de categoría, combinación inseparable, tareas separables y
etiqueta no cubierta; repetir idénticos prompts con presupuestos suficientes e
insuficientes. Registrar clasificación, planificación, preparación, generación,
memoria y transferencias por separado, y calidad con referencias revisadas.
Incluir coste de carga inicial y coste por turno; distinguir bloqueo del
planificador de fallo de generación. Los escenarios sirven como desarrollo,
no como test independiente de recuperación.

### 2026-10-15: planificador y tiempos por subtarea
Ejecutados 24 planes de desarrollo (4 escenarios, capacidades 1/2 pools y presupuestos 0/1/2 cambios), archivados en results/dynamic_planner_20261015/report.json. Son etiquetas suministradas, sin clasificador ni generacion; no validan calidad. tasks.execute registra ahora prepare_seconds, generation_seconds por subtarea y execution_seconds sincronizados en CUDA. El total excluye inicializacion, clasificacion y planificacion, explicitamente. Seis tests existentes pasan. Falta evaluacion con preguntas no usadas para construir los fixtures y generacion real.

### 2026-10-15: subtareas reales con ASI37595 en CUDA
Perfil vinculado mediante SHA256 verificado a model_37595.pt y manifiesto. Ejecutado derive_then_code con 64 tokens por subtarea, salida results/dynamic_planner_20261015/derive_then_code_cuda.json. Un cambio math_physics a technology, 44 cargas de experto-capa (22 por pool), 22 expulsiones, 32440320 bytes H2D, sin transferencias durante cada generacion. Bucle 11.1675 s (excluye carga inicial/clasificador/planificacion). Las dos respuestas repiten instrucciones y no resuelven derivada ni codigo: fallo cualitativo de este caso, no evidencia de beneficio de calidad. Prompt espanol y formato estructurado, sin control ingles/formato ni baseline conjunto; no atribuir causalidad a la separacion.

### 2026-10-15: control conjunto del mismo prompt
Ejecutado derive_then_code_joint_cuda.json: mismo checkpoint/prompt, maximo128 tokens total, dos pools simultaneos. Informe split_vs_joint.json conserva hashes y salidas. Conjunto13.1445s frente separado11.1675s; ambos32,440,320bytes H2D y44 cargas, expulsiones0 frente22. Ambos repiten la solicitud sin resolverla. Conjunto activa4 expertos/capa uniformes frente2 por subtarea; no es control puro de residencia ni evidencia de eficiencia por respuesta correcta. Una repeticion, idioma/formato no controlados; no declarar speedup general.

### 2026-10-15: resultado dinamico negativo incorporado al paper
La seccion de limitaciones recoge el probe split/joint, sus contadores, tiempos y ambos fallos de respuesta. build_evidence.py verifica hashes de los informes crudos y contadores invariantes durante generacion antes de incorporarlos. No se presenta como validacion de calidad o eficiencia util. Evidencia regenerada; sincronizacion y compilacion remota siguen pendientes.

### 2026-10-15: supuestos aritmeticos resueltos en borrador v6
Tres referencias condicionales explicitan sus premisas: proporcion agua/calorias ficticia, consumo literal de4gal/milla y suma idealizada de duraciones de tanques. Resultados800,4000,6 comprobados aritmeticamente. training_draft_v6.json conserva240 filas y43 correcciones, train-only y no aprobado. Historial y paquete Docker v5 intactos; sus resultados validan el contenido v5, no automaticamente el hash global v6. Nuevos textos necesitan tokenizacion/revision final y recomputar profesor; no son holdout fresco.

### 2026-10-15: validacion v6 con tokenizacion de entrenamiento
237 textos intactos y3 modificados respecto a v5; los10 casos de codigo empaquetados conservan prompt/respuesta. Ninguno ejecutado aun. supervised_rows real acepta240 filas, maximo275 tokens bajo384. Un primer chequeo conto campos BatchEncoding en lugar de tokens; marcado invalido en draft_v6_validation.json y sustituido por draft_v6_training_lengths.json. No entrenar ni interpretar hashes de corpus distintos como ejecucion validada.

### 2026-10-15: alcance del error de tokenización
Revisadas todas las llamadas apply_chat_template de asi: public_moe.input_ids usa return_dict=False y Colibri tokeniza texto explícitamente. El error estaba en el chequeo ad hoc invalidado, no en estas rutas. Quince tests test_public_moe pasan. Estado principal actualizado para no mostrar DeepSeek todavía en ejecución.

### 2026-10-15: nueva fuente candidata de evaluacion de codigo
HumanEval oficial MIT descargado solo como datos/licencia, revision6d43fb980f9fee3c892a914eda09951f772ad10d. Inventario164 IDs unicos, cero conflictos exactos por detector frente corpus_v3 y borrador_v6; results/recovery_holdout_sources_20261015/humaneval_audit.json guarda hashes. No prueba ausencia de duplicados semanticos ni exposicion de preentrenamiento. No se ejecutaron soluciones ni modelos; no congelado como test. Pendiente revision semantica contra MBPP, protocolo de completado vs chat y evaluacion funcional aislada. Fuentes: https://github.com/openai/human-eval (MIT/formato/evaluador); https://github.com/LiveCodeBench/LiveCodeBench (alternativa temporal revisada, no incorporada).

### 2026-10-15: solapamiento semantico HumanEval/MBPP detectado
Busqueda TF-IDF local de164 enunciados contra MBPP historico; no requiere sklearn ni ejecuta codigo. Revision manual asistente de10 pares de mayor similitud:2 comparten tarea central (factor primo mayor y MCD),6 variantes relacionadas,2 falsos positivos lexicos. Informe semantic_overlap_review_01.json; quedan154 sin revisar. Por tanto cero duplicados exactos NO permite declarar HumanEval independiente. Mantener set oficial intacto; cualquier subconjunto filtrado debe etiquetarse como evaluacion local y no score oficial completo.

### 2026-10-15: revision semantica y limite del recuperador lexico
Revisados15 pares adicionales (25 candidatos en total). Se distingue compartir conceptos de duplicar una tarea; ninguna etiqueta different_task certifica independencia global. Seguimiento manual detecta otro duplicado central: HumanEval/31 y mbpp_train_4 piden primalidad; el vecino lexico inicial habia sido paridad. Evidencia semantic_overlap_primality_followup.json. Por ello no aceptar automaticamente puntuaciones bajas ni filtrar solo por el primer vecino. Al menos3 tareas centrales coinciden; no se ha seleccionado un test ni observado respuestas del modelo.

### 2026-10-15: recuperacion reproducible de candidatos de solapamiento
asi.analysis.code_overlap genera cinco vecinos por cada uno de164 problemas (820 pares), con hashes de fuentes y script, sin depender de sklearn. Ejecutado sobre HumanEval fijado y MBPP historico: overlap_top5.json. Los3 duplicados centrales ya conocidos aparecen entre los5; primalidad esta en cuarta posicion y antes se perdia. Este control retrospectivo no mide recall global ni certifica ausencia de fuga. Queda revision semantica; no entrenado/evaluado modelo.

### Sincronización 15/10/2026
Subidos a Overleaf asi_paper.tex, full_suite_table.tex, full_suite_models.pdf, colibri_candidates.pdf y asi_evidence.json. Compilación observada: 28 páginas, 0 errores, 0 warnings y 8 avisos de maquetación (incluido overfull vbox 6.70938pt). Descarga mediante navegador agotó tiempo y reinició la sesión de control; PDF local y captura pendientes de verificación. No repetir benchmarks por este fallo de descarga.

### Cierre de descarga y revisión parcial 15/10/2026
Resuelto el fallo de descarga: output/pdf/asi_paper_20261015.pdf conserva la versión compilada de 28 páginas. Revisión visual de páginas 10–15 y 20 sin cortes ni solapamientos visibles; tabla y figura de la suite completa presentes. Huellas y alcance en paper/evidence/overleaf_fullsuite_sync_20261015.json. No equivale a revisión integral de maquetación: quedan páginas restantes y aviso overfull vbox. PDF histórico conservado.

### Revisión general de las 28 páginas
Renderizadas y revisadas todas las páginas del PDF fechado 15/10. Sin páginas vacías, cortes grandes ni solapamientos visibles. Las figuras densas requieren una revisión de etiquetas a tamaño de publicación; el aviso overfull vbox sigue sin localizarse mediante log. No se ha recompilado ni alterado el contenido científico en esta comprobación.

### HumanEval: revisión de 50 pares adicionales
semantic_overlap_review_03.json revisa cinco candidatos para diez tareas nuevas. HumanEval/79 comparte conversión decimal-binario con mbpp_test_22 y mbpp_test_88 (añade envoltorio db); no usarlo como prueba de independencia por cero coincidencias exactas. Otros pares comparten subtareas, sin que esto implique automáticamente fuga. Los nombres históricos test/train no certifican exposición efectiva. No se consultaron respuestas del modelo ni ejecutaron soluciones; falta revisión del resto y congelación del protocolo.

### Consolidación editorial solicitada por Marc
El banco denso está entrenándose en homelab; la comprobación Docker sigue pendiente. Retirada la tabla exclusiva del banco que duplicaba su columna en la tabla conjunta de 19 tareas. Eliminado su generador de macros y etiquetas sin uso. Creada subsección general de suite completa para las tres arquitecturas; preservadas diferencias de protocolo y advertencia ARC-Easy. build_evidence.py termina correctamente con 45 fuentes. Cambios locales todavía no sincronizados ni recompilados; el PDF fechado conserva la versión anterior. Continuar depuración de scripts y comparaciones redundantes sin borrar evidencia única.

### Consolidación de la figura del banco y guía de benchmarks
Retirado del generador el panel histórico de calidad de 100 preguntas junto al control de memoria. La nueva asi_bank_residency conserva solo memoria, revisada visualmente; la calidad completa queda en la comparación conjunta. Fuentes históricas intactas. BENCHMARKS_RAPIDOS.md centraliza cinco bloques y sus límites de comparabilidad. Generador comprobado con 45 fuentes. Pendiente sincronizar figura nueva y fuente TeX; no presentar PDF anterior como versión consolidada.

### Código compartido de identidad binaria
Consolidadas tres implementaciones SHA-256 (carga DeepSeek original, piloto público y perfiles de recuperación) en asi.data.identity.file_sha256. Se mantienen aliases públicos para no romper imports y se distingue explícitamente del hash de manifiesto que normaliza saltos de línea. Control binario de más de 8 MiB coincide con hashlib; 15 tests public_moe y 2 recovery_hardware pasan. Sin cambios en pesos, protocolos ni evidencia histórica.

### Guías de benchmarks consolidadas
README dirige la comparación ASI/DeepSeek/banco a BENCHMARKS_RAPIDOS. COMPARISON aclara que su matriz original/classified/domain es otro experimento, evitando confundirla con las tres arquitecturas actuales. USAGE retira el comando parcial repetido de HellaSwag y remite a la guía vigente. Notas fechadas separadas de instrucciones actuales. Enlaces locales comprobados en los cuatro documentos; no se borran recetas ni resultados históricos.

### Unificación de controles históricos de calidad
Retirada figura de líneas de seis tareas que repetía la tabla de 100 ejemplos, junto con su código generador. Incorporado el banco a esa misma tabla tras controles de recuentos/evaluador/datasets existentes; eliminada enumeración repetida de sus seis scores en prosa. Suite completa y controles históricos permanecen separados. Generador de 45 fuentes pasa; referencias internas TeX resueltas. Falta sincronizar y compilar esta revisión editorial.

### Revisión consolidada sincronizada
Overleaf actualizado con fuente, macros, nueva figura de residencia del banco y evidencia. Compilación: 27 páginas (antes 28), cero errores/warnings y 12 avisos underfull; ya no aparece overfull. PDF nuevo output/pdf/asi_paper_consolidated_20261015.pdf, histórico intacto. Captura y hashes en overleaf_consolidated_sync_20261015.json. Pendiente revisión visual de la nueva paginación.

### Guía del paper actualizada
Revisadas páginas 8–14 de la revisión consolidada; tabla histórica unificada, residencia y suite completa legibles sin cortes. README del paper reemplaza estados antiguos contradictorios y texto mal codificado por índice vigente; registros de sincronización conservan historia. No certificada aún toda la paginación nueva.

### 16/10: prioridad del paso 2
Marc indica esperar a los modelos densos del paso 4 antes de continuar el paso 1. Se añade control aleatorio de máscaras con tres semillas, mismo presupuesto 512/8 y escala nativa; plan generado sin GPU, inferencia pendiente. Cuatro tests de máscaras pasan. No hay resultados nuevos de calidad ni selección por test.

### 16/10: controles aleatorios en ejecución
Añadidas opciones random-only/reuse-calibration; control de donante comprobado con 96 filas y seis prompts, tres donantes inválidos rechazados, cuatro tests de máscaras pasan. Tanda nueva iniciada PID 17220 con logs en results/random_mask_control_20261016. Comprobar proceso y salida antes de reanudar o lanzar otra GPU; no considerar resultados hasta completarse. Paso 1 sigue en espera por indicación de Marc.

### Protección de reutilización de calibración
Seis tests de máscaras pasan, incluyendo incompatibilidad de fuente/protocolo, calibración incompleta, solapamiento y capas ausentes; copiar calibración no copia resultados ni comparte objetos mutables. Proceso de controles aleatorios observado activo; sin resultados todavía en la comprobación. No se modifica su script durante la ejecución.

### Procedencia de la tanda aleatoria
Guardado execution_provenance.json con hashes de código/donante, versiones y GPU sin cargar otro modelo. Captura posterior al arranque, explícitamente no introspección del proceso. El plan de máscaras precede a las opciones de reutilización; sidecar identifica la revisión ejecutada. Donante histórico carece de fingerprint completo de runtime: no atribuir cambios temporales a selección ni presentar esta tanda como benchmark emparejado de velocidad. Worker 2624 observado activo, sin respuestas ni errores registrados en esta comprobación.

### Comparador de selección listo
Añadido --compare-with al informe existente, sin otro script. Validación con fixture emparejado y rechazos por ejecución incompleta, tokens, transferencias y duplicados. No genera conclusión con resultados parciales. Tanda observada en 3/18 respuestas; ninguna conclusión de calidad todavía.

### Tests persistentes del comparador
Añadidas pruebas al archivo existente test_mask_intervention.py, sin otro script: rechaza datos parciales, tokens/fuentes/protocolo distintos, transferencias, duplicados y CE no finita. Ocho tests pasan; salida mantiene explícitamente timing_comparable=false e independent_test=false. Tanda observada en 4/18; worker sigue activo.

### Revisión del PDF consolidado completa a escala general
Comprobadas las 27 páginas; sin cortes grandes ni solapamientos. Detectado error semántico en apéndice: E7 aún nombraba el banco, aunque E7 del cuerpo es Colibri. Corregido localmente; E1 ahora incluye suite completa, E0/E200/E1000 y banco. Pendiente sincronizar esa corrección junto al siguiente bloque de evidencia. Tanda aleatoria observada en 6/18; no cambiar configuración.

- 16/10: control aleatorio terminado (18 respuestas, tres semillas, seis preguntas compartidas). CE 3.242/3.583/3.064 frente a 2.515 recalibrada; no recuperación de generación. Evidencia y límites: `results/random_mask_control_20261016/REVIEW.md`. Falta control global calibrado sin categorías para separar especialización de utilidad global; no repetir este control aleatorio como pendiente.

- 16/10: lanzado control global calibrado sin categorías (`results/global_mask_control_20261016`), seis respuestas pendientes. Plan/máscaras/huellas fijados antes de ejecutar; diez tests de selección y validación pasan. Paso 1 continúa aplazado por indicación de Marc.

- 16/10: paper local incorpora el control aleatorio validado en una tabla de presupuesto 512; generador actualizado (47 informes). No se atribuye el efecto exclusivamente a categorías ni se afirma recuperación. Sincronización/compilación pendientes; control global sigue activo.

- 16/10: control global terminado y validado (6/6): CE 2.727, repetición media 0.771, EOS 1/6. Mejor CE que máscaras aleatorias, peor que categorías recalibradas (2.515); generación no recuperada. Cierra este control exploratorio, no confirmación independiente. Paper local actualizado; Overleaf pendiente.

- 16/10: cerrada sincronización de controles aleatorio/global en Overleaf. PDF `output/pdf/asi_paper_selection_20261016.pdf`: 28 páginas, 0 errores/warnings y 12 avisos de espaciado. Revisadas 28 páginas en vista general y 23–24 en detalle; tabla verificada sin recortes. Evidencia `paper/evidence/overleaf_selection_sync_20261016.json`.

- 16/10: fijado protocolo de decodificación `configs/decoding_control_dev_v1.json` (72 respuestas de desarrollo, original/restringido, greedy frente a muestreo con tres semillas). Implementación y ejecución pendientes; no equivale a test independiente ni inicia recuperación.

- 16/10: ampliado decoder compartido con muestreo reproducible por respuesta; cuatro tests CPU específicos pasan. Falta integrar el runner de las 72 respuestas y verificar GPU; no considerar ejecutado el control de decodificación.

- 17/10: lanzado control de decodificación (72 respuestas) en `results/decoding_control_dev_20261017`; validación sin modelo y cuatro tests específicos pasan. No interpretar tiempos como benchmark por actividad de escritorio. Esperar cierre y revisión semántica, sin resultados incorporados todavía.
