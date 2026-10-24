# Adopción del protocolo de recuperación V2

Estado: F0 en preparación. Ningún gate aprobado, ninguna recuperación nueva demostrada.
Propuesta original conservada en `PROTOCOLO_RECUPERACION_V2.md`.

## Aclaraciones necesarias

- El profesor es una referencia, no un límite superior garantizado de calidad.
- El test no se genera con el profesor ni se inspecciona en F0: se reserva para F7.
  La construcción y descontaminación deberán aislarse de la selección de métodos;
  falta concretar ese mecanismo antes de crear los splits sellados.
- El inventario de datos quemados debe incluir todas las evaluaciones ya utilizadas,
  no solamente tres preguntas. No se presupone que MBPP/GSM8K/SciQ proporcionen
  suficientes ejemplos inéditos después de este filtro.
- Bytes transferidos divididos por tiempo total de generación no miden ancho de
  banda PCIe. F0 debe medir transferencias separadamente.
- KL top-k más cola es una aproximación agregada; no sustituye a la KL completa
  que evalúa equivalencia. Una meseta tampoco demuestra por sí sola un límite de capacidad.
- Un Job Object limita recursos, pero no garantiza aislamiento de red y archivos.
  La evaluación de código permanece pendiente hasta validar un sandbox real.
- Usar identificadores V2-F0…V2-F7 para no reutilizar los identificadores históricos.
- No publicar resultados ni modificar el paper hasta F7. No elegir perfil A/B
  hasta medir presupuesto real. No iniciar trabajos GPU simultáneos.

## Primer control

Ejecutar la suite unittest existente antes de implementar nuevos métodos. Registrar
salida y código de retorno; un fallo se diagnostica antes de pasar a otra fase.
Este control no basta para aprobar G0: quedan datos, harness, equivalencia,
sandbox, baselines y presupuesto medido.

## Auditoría de datos, 18/10/2026

El escaneo ampliado terminó sobre 2.229 archivos registrados. Los dos JSON de
más de 20 MB fueron procesados; JSONL, CSV, YAML, logs, textos y Parquet ya tienen
lectura explícita. Un manifiesto del profesor está corrupto: el corpus antiguo
asociado queda excluido conservadoramente. No se cargan pickles arbitrarios.
Los artefactos binarios se registran, pero no se presentan como decodificados.

Se excluyen completos los splits test locales de GSM8K/MBPP/SciQ y las
demostraciones de MBPP. El primer filtro exacto/13-gramas deja 2.783 candidatos
GSM8K, 41 MBPP y 7.848 SciQ. Son candidatos en cuarentena, no splits utilizables:
faltan duplicados entre candidatos, revisión semántica y validación de referencias.
SciQ contiene ciencias distintas de biología; falta filtrar dominio.

El cruce adicional con 79 archivos Arrow/Parquet de cachés está separado en
`results/v2/F0/cache_overlap/`. Excluye conservadoramente incluso descargas que
no demuestran evaluación; antes de adoptar sus exclusiones hay que distinguir
particiones realmente evaluadas de simples descargas mediante informes históricos.
La disponibilidad local de código es insuficiente para 200 dev + 200 test.
No reducir cuotas silenciosamente ni reciclar el test antiguo; buscar fuentes
adicionales con licencia y procedencia fijadas. G0 sigue pendiente.

### Resultado del cruce y resolución de cachés

79/79 archivos procesados sin fallos; consolidación con hashes comprobados en
`results/v2/F0/inventory_consolidated/`. Se excluyen 2.653 candidatos SciQ por
coincidencias con particiones registradas de evaluación. 3.658 candidatos no
tienen coincidencias detectadas en este cruce; no significa independencia.

De las 4.361 coincidencias inicialmente sin resolver, 2.782 GSM8K corresponden
solo a copias de su mismo train. Verificados Parquet idéntico por SHA-256 y
7.473 filas idénticas entre Arrow y Parquet. Evidencia en
`results/v2/F0/self_cache_resolution/report.json`. Quedan 1.579 candidatos con
coincidencias adicionales pendientes. Resolver procedencia no autoriza usar
ejemplos: faltan referencias, semántica y separación de grupos.
