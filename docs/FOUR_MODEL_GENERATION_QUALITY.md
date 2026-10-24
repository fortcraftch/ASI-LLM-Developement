# Comparación exploratoria de generación: cuatro modelos base

- Configuración congelada antes de ejecutar: `configs/four_model_quality_20261024.json`.
- Resultados: `results/v2/F0/four_model_quality_20261024/`.
- 16 prompts nuevos en inglés, dos por cada una de ocho categorías. Son diagnósticos y no forman parte del test sellado ni sirven para seleccionar recuperación.
- Checkpoints idénticos a la auditoría SDPA, verificados por SHA256. ASI conserva pareja categorizada; DeepSeek routing libre; banco selecciona el especialista de categoría; GPT2 base denso. Sin SFT.
- Categoría preasignada, mismo texto literal en las cuatro familias; excluye errores del clasificador. Sin plantillas de chat especiales por modelo.
- SDPA, FP32 weights/BF16 autocast, greedy, seed1337, batch1, 4 threads, máximo128 tokens, EOS activo, sin penalización antirrepetición. Prefijo completo, sin KV incremental.
- Una generación por prompt y modelo; no estimamos variabilidad entre semillas ni significación estadística.
- Guardamos todo el texto y tokens, stop reason, hashes, TTFT/tiempo descriptivo, pico CUDA, RSS muestreado, NVML si disponible, H2D de expertos. Estas duraciones con EOS variable no reemplazan el benchmark de velocidad a longitud fija.

## Evaluación

1. Repetición: fracción de ocurrencias de tetragramas de palabras repetidas en la respuesta, minúsculas, antes del primer EOS. Una respuesta con menos de cuatro palabras no aporta valor a este promedio: no imputar cero.
2. Restricciones: proxy de exactamente una oración, exactamente dos líneas con guion, o coincidencia exacta según el prompt. No equivale a corrección semántica ni es un IFEval oficial. Se complementa con lectura manual.
3. Coherencia: revisión exploratoria no ciega por asistente, 0 incoherente/no respuesta, 1 comprensible con deriva o contradicciones, 2 coherente y relevante en toda la salida. La repetición y truncamiento se anotan por separado.
4. Corrección: 0 incorrecto/no responde, 1 parcialmente correcto, 2 respuesta central correcta sin errores materiales. Referencias y rúbrica prefijadas en configuración.
5. Instrucción completa: exige contenido y forma; no confundir un proxy estructural aprobado con una respuesta correcta.

Bootstrap descriptivo por prompt (10.000 remuestreos, seed1337). No ejecutar código generado. No publicar resultados incompletos ni promover mejoras basadas en esta tanda. No modifica el paper automáticamente.

## Ejecución / reanudación

```powershell
F:/.venv/Scripts/python.exe -m recovery_v2.four_model_quality --output results/v2/F0/four_model_quality_20261024 --resume
F:/.venv/Scripts/python.exe -m recovery_v2.four_model_quality_report
```

Antes de reanudar comprobar PID de STATUS y procesos vivos; no duplicar GPU. El runner guarda respuestas atómicamente, comprueba identidad de configuración/código y conserva errores. No hay optimizador: es inferencia determinista sin entrenamiento.
