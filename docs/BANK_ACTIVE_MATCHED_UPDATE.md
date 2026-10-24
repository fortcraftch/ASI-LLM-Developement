# Banco con capacidad activa equivalente: resultados completos

Los ocho checkpoints de `results/bank_active_matched_seed1337` han superado la
auditoría contra `configs/bank_active_matched_v1`: recetas, pasos finales,
finitud de pesos y SHA256. Cada especialista es denso, con 93.850.112 parámetros;
el banco completo suma 750.800.896. Solo uno responde a cada entrada.

La igualdad con ASI se refiere al inventario de parámetros participantes,
contando tablas de embeddings completas y excluyendo gates omitidos. No iguala
FLOPs, almacenamiento total ni exposición de cada backbone a las categorías.
El clasificador externo se contabiliza aparte.

## Ejecución

Evidencia nueva: `results/v2/F0/bank_active_matched_20261022/`.
`audit.json` verifica los ocho modelos. La cola ejecuta primero generación
y después los 19 benchmarks, sin solapamiento GPU.

```powershell
F:/.venv/Scripts/python.exe -m recovery_v2.bank_suite --output results/v2/F0/bank_active_matched_20261022 --resume
```

No ejecutar ese comando si la cola ya está viva. `STATUS.json` identifica la
tarea y proceso. Cada tarea conserva intentos y trazas. La reanudación omite
tareas terminadas y verificadas; una tarea interrumpida se repite completa.
No se accede al test sellado de recuperación v2 ni se seleccionan pesos con
estos benchmarks exploratorios.

Generación: ocho prompts históricos, 64 tokens, tres repeticiones tras
calentamiento, greedy, prefijo completo, pesos FP32 y autocast BF16. Separa
TTFT y velocidad de decode; añade salida con parada EOS, preparación y
memoria. Esta medición carga un especialista y mantiene los otros en disco;
no representa una caché de ocho modelos en RAM. NVML no disponible se marca
como null, no como cero.

Benchmarks: todos los modelos residentes para acelerar, uno activo por input,
categoría predicha, batch 2/512. Los checkpoints permiten batch 2; el histórico
usó 4/512. Debe declararse esta diferencia y comprobar huellas de datasets y
evaluador antes de agrupar resultados. Los tiempos de benchmarks no son
velocidad de generación. Las trazas permiten análisis por documento; mientras
no se calculen intervalos, no se afirmará significación.

## Cambios del paper una vez validados los resultados

1. Actualizar E1: banco nuevo ya entrenado, 93,85M activos / 750,80M totales.
2. Incorporar la suite nueva en la comparación principal, preservando las
   ediciones manuales de las tablas. Conservar banco de 15,50M como histórico
   claramente identificado; no sustituir sus cifras de memoria por etiquetas nuevas.
3. Incorporar generación/memoria nuevas al overview con TTFT separado de decode.
4. Revisar conclusiones y limitaciones: comparar capacidad participante sin
   atribuir causalidad a arquitectura ni equiparar presupuestos totales.
5. Descargar fuente actual de Overleaf, aplicar cambios selectivos, compilar y
   revisar visualmente PDF. No publicar métricas pendientes como completas.

Esta ejecución no cierra el gate de recuperación F0 ni avanza de fase.

## Resultados verificados (22/10/2026)

Suite completa: 19 tareas, 66.832 ejemplos; hashes de checkpoints, evaluador y
huellas de datasets verificados. Frente al banco pequeño: 15 mejoras, dos
empates y dos descensos. Frente a ASI: 9/9/1 (mejor/peor/igual); frente a DeepSeek:
5/11/3. No son afirmaciones de significación estadística.

Generación: 28,98 tokens/s de decode, 35,25 ms TTFT, 358,01 MiB de pesos GPU y
490,97 MiB de pico asignado. Medianas de velocidad, máximo observado de memoria.

`recovery_v2/bank_paper_update.py` reproduce esta revisión editorial desde el
snapshot de Overleaf del 22/10, con resultados y fuentes verificadas. No debe
usarse sobre futuras ediciones sin renovar y revisar ese snapshot. Los antiguos
`paper/build_editorial.py` y `paper/build_evidence.py` reconstruyen evidencia
histórica y no deben ejecutarse para sobrescribir esta revisión ni las tablas
manuales. El banco pequeño queda archivado, no reetiquetado.
