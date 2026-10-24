# Hoja de ruta activa: recuperación V2

Este plan sustituye desde el 17/10/2026 la lista operativa anterior, por petición
 de Marc. Fuente: [protocolo de Claude](../PROTOCOLO_RECUPERACION_V2.md), con
[aclaraciones metodológicas](RECOVERY_V2_ADOPTION.md).
La [hoja anterior](archive/ROADMAP_PRE_RECOVERY_V2_20261017.md) conserva los hitos,
resultados y decisiones históricas; sus pendientes no son órdenes activas.

## Pregunta principal

¿Podemos recuperar la calidad de OLMoE Instruct con 32 o 16 expertos residentes
por capa, ocho activos y pesos base congelados? Separaremos selección de expertos,
adaptación, generación y coste. Una meseta no demostrará por sí sola falta de capacidad.

## Orden de ejecución

| Fase | Trabajo | Condición para avanzar |
|---|---|---|
| F0 — activa | Mapa del código, pruebas, datos descontaminados, métricas, sandbox, equivalencia HF/caché, transferencias, baselines y presupuesto | G0: controles numéricos y de aislamiento aprobados, datos y baselines válidos, presupuesto medido |
| F1 | Cobertura del router y sensibilidad por capa/categoría | G1: decidir prioridad de máscaras generales/categorizadas y presupuesto uniforme/no uniforme |
| F2 | Selección de máscaras por frecuencia, masa, salida y reconstrucción; controles aleatorios | G2: mejora relativa de KL de al menos 15% con IC excluyendo cero; si no, conservar referencia |
| F3 | Reparación por capa: ganancias, router y compensador | G3: mejora de KL de al menos 10% con IC excluyendo cero; si no, omitir reparación |
| F4 | Destilación controlada: LoRA, router, ablaciones y escalera de tokens | G4: analizar pendiente y ajuste con incertidumbre; decidir siguiente presupuesto sin seleccionar con test |
| F5 | Destilación sobre trayectorias del estudiante para reducir bucles | G5: mejora de KL on-policy de 30%, sin deterioro teacher-forcing y bucles dentro del margen preregistrado |
| F6 | Frontera calidad/memoria/latencia: activos, fallos permitidos y cuantización | G6: comparar con modelo completo cuantizado bajo recursos medidos compatibles |
| F7 | Congelar candidatos, evaluación sellada, figuras y paper | Evaluación final una vez, evidencia trazable y revisión del documento |

Los detalles y umbrales completos del protocolo prevalecen sobre esta tabla resumida.
Si falla un gate, documentar DECISION.md y no avanzar de fase sin indicación de Marc.
Elegir perfil A o B después de medir F0; no lanzar toda la matriz anticipadamente.

## Estado comprobado

- F0.1: mapa creado y línea base de 229 pruebas aprobada.
- F0.3 parcial: KL, NLL, acuerdo top1, repetición, bootstrap pareado y McNemar;
  cinco pruebas sintéticas aprobadas. No equivale al harness M1–M8 completo.
- F0.2 en curso (18/10): 2.229 archivos registrados; dos JSON grandes procesados,
  un manifiesto corrupto aislado. Primer filtro: 10.672 candidatos, aún en cuarentena.
  68 grupos de preguntas relacionadas deben permanecer juntos al separar splits.
  18 identidades de datasets en 37 informes históricos; 30 de 79 cachés vinculadas
  a particiones de evaluación. Cruce textual terminado: 79 archivos sin fallos,
  hashes de entradas/salidas comprobados. Excluidos 2.653 candidatos SciQ por
  coincidencias con particiones de evaluación; 4.361 coincidencias de caché
  inicialmente sin resolver y 3.658 candidatos sin coincidencia de caché detectada.
  Resueltas después 2.782 coincidencias GSM8K como copias idénticas de su propio
  train (hash Parquet y 7.473 filas Arrow verificadas); 1.579 siguen pendientes.
  Esto no demuestra independencia semántica ni valida las referencias.
  BigCodeBench: 1.140 tareas descargadas con revisión y hashes fijados; sintaxis
  comprobada, sin ejecución de código ni asignación de splits. Faltan descontaminación,
  aislamiento funcional y validación de dominio/referencias. No hay test nuevo certificado.
- F0.4–F0.8 pendientes: equivalencia real, cargas, referencias dev, baselines y presupuesto.
- G0 pendiente; no hay entrenamiento V2 ni recuperación demostrada.
- Sandbox: tres controles Docker aprobados en homelab el 18/10, imagen Python
  fijada por digest. No equivale a validar referencias de BigCodeBench; falta
  imagen con dependencias y su control de aislamiento.

## Reglas de continuidad

Resultados nuevos en results/v2; código en recovery_v2. Preservar resultados previos.
No tocar el paper hasta F7. El test queda reservado y no se inspecciona en F0.
El control anterior de decodificación queda como diagnóstico histórico incompleto;
no se relanza automáticamente ni se convierte en test independiente.
Los entrenamientos en homelab pueden continuar, pero no se presume acceso ni GPU libre.
Comparación local ASI/DeepSeek/banco, sistema dinámico e integración Colibri quedan
como líneas posteriores, fuera de la secuencia activa V2, no como tareas completadas.
