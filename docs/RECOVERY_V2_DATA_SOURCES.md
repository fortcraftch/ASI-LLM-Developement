# Fuentes adicionales para F0 (candidatas, no admitidas todavía)

Revisión 18/10/2026. El filtro local deja solo 41 ejemplos MBPP antes del cruce
de cachés. No alcanza para dev/test de 200 ejemplos cada uno.

## Código: evaluación

[BigCodeBench](https://github.com/bigcode-project/bigcodebench) proporciona 1.140
tareas orientadas a ingeniería de software. Su
[ficha fijada](https://huggingface.co/datasets/bigcode/bigcodebench/blob/fd501fef82825d064962aa21d8520898c823dd3c/README.md)
declara Apache-2.0. Es candidato, no sustituto automático de MBPP: usa bibliotecas
y tareas más complejas, por lo que puede producir un efecto suelo con OLMoE.
Antes de admitirlo: fijar archivos y hashes, descontaminar contra el inventario,
agrupar variantes de una misma tarea y validar referencias en sandbox.
No separar versiones Complete/Instruct de una misma tarea entre splits.
Su publicación anterior al checkpoint no permite afirmar ausencia de exposición
durante el preentrenamiento.

## Código: entrenamiento

[Magicoder OSS-Instruct](https://github.com/ise-uiuc/magicoder) contiene instrucciones
sintéticas generadas a partir de código. Puede cubrir el déficit de prompts de
entrenamiento, pero no es un test independiente ni una fuente de respuestas
automáticamente correctas. Pendiente fijar revisión, licencia/condiciones de los
datos, filtro por lenguaje, validación y descontaminación contra dev/test.

BigCodeBench descargado el 18/10 a `data/v2/sources/bigcodebench`, revisión
`fd501fef82825d064962aa21d8520898c823dd3c`, 1.140 registros. Ficha y Parquet
con SHA-256 en `download_manifest.json`; permanece en cuarentena y sin splits.
La inspección sintáctica no equivale a ejecutar ni validar las soluciones.
Preparadas 1.140 filas con ambas versiones del prompt, solución y tests bajo
el mismo ID; sin splits. Escaneo exacto/13-gramas iniciado en
`results/v2/F0/bigcodebench_overlap` contra cachés, archivos históricos
indexados y fuentes antiguas. Checkpoints por archivo permiten `--resume`.
Este filtro puede excluir coincidencias de código rutinario: no se relajará
después de ver calidad del modelo. Revisar motivos y fijar política antes de splits.
Magicoder no se ha descargado ni incorporado. La selección de dificultad
se hará con criterios previos y un piloto dev, sin inspeccionar el test sellado.
