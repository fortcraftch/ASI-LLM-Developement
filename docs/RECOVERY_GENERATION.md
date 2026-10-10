# Comparación de generación OLMoE: original, 32 y 16

Las tres ejecuciones están completas. El análisis de errores, repeticiones y
controles aún pendientes está en [RECOVERY_REPETITION_ANALYSIS.md](RECOVERY_REPETITION_ANALYSIS.md).

Esta prueba compara **OLMoE-1B-7B-0924-Instruct** original con dos adaptaciones
de router + LoRA de la proyección de salida de los expertos. No corresponde al
modelo categorizado de 124M entrenado desde cero.

Todos activan ocho expertos por token y capa. El original elige entre 64; los
alumnos eligen entre los 32 o 16 residentes de la categoría durante toda la
respuesta. La categoría se toma del dataset, para aislar esta restricción.

## Selección y protocolo

- 32: `router-expert-lora/adaptation_best.pt`, paso 24, seleccionado por validación.
- 16: `curriculum_direct16/adaptation_best.pt`, paso 24. Es el mejor de los tres
  calendarios para la peor pérdida relativa por categoría en validación **con 16**.
  No confundirlo con `curriculum_mixed`, usado en el anterior perfil de hardware
  porque aquel seleccionaba conjuntamente sobre 32 y 16.
- Tres preguntas reservadas del piloto: quitar dígitos de cadenas en Python,
  calcular páginas leídas por semana y explicar la energía en un ecosistema.
- Greedy, máximo 256 tokens, parada EOS, mismo orden y proceso nuevo por variante.
- Caché RAM de expertos limitada a 2 GiB; presupuesto del proceso de 12 GiB.
- Sin calentamiento separado; caché del sistema operativo sin vaciar. Una pasada:
  son medidas exploratorias, no estimaciones estables de rendimiento.

El original no cabe completo en la RTX 3060 de 12 GiB. Conserva todas sus
posibilidades de routing, con una caché GPU global de 512 parejas (capa, experto).
Puede hacer swapping dentro de una respuesta; su tiempo incluye ese coste.
Los alumnos se precargan por categoría y se comprueba que **no hay cargas de
expertos durante la generación**. Sus adaptadores permanecen en GPU y cuentan
en la memoria medida.

## Reproducir

Desde la raíz del repositorio, en el entorno habitual. Usar otra carpeta de
salida si ya existen resultados: el programa evita sobrescribirlos.
Ejecutar los comandos secuencialmente, sin otros experimentos de GPU simultáneos.

```bash
python -m asi.experiments.recovery_generation --variant original --output results/generation_recovery_20261007/original
python -m asi.experiments.recovery_generation --variant 32 --checkpoint results/recovery_quick_20261006/router-expert-lora/adaptation_best.pt --output results/generation_recovery_20261007/32
python -m asi.experiments.recovery_generation --variant 16 --checkpoint results/recovery_quick_20261006/curriculum_direct16/adaptation_best.pt --output results/generation_recovery_20261007/16
python -m asi.analysis.recovery_generation_report results/generation_recovery_20261007
```

El informe verifica que las tres ejecuciones tienen los mismos prompts,
tokenización, protocolo, modelo original, versión de PyTorch y GPU. Las
identidades del corpus, calibración y checkpoint se verifican antes de inferir.
Los datos parciales se guardan tras cada respuesta; solo se comparan tres
ejecuciones completas.

## Leer los resultados

- `comparison.png`: tiempo total (preparar + generar), velocidad de decodificación,
  pico VRAM, RAM, transferencias durante generación y repetición de trigramas.
- `summary.md`: tabla detallada, espacio en disco y textos completos.
- `outputs.html`: las tres respuestas a cada pregunta lado a lado.
- Cada `report.json`: tokens, EOS, TTFT, tiempo de preparación, contadores de
  caché, lecturas lógicas de disco, memoria y procedencia verificable.

VRAM es memoria asignada por PyTorch, incluyendo preparación y generación;
también se guarda el pico reservado. RAM es RSS del proceso muestreada cada
50 ms: no incluye toda la caché del sistema operativo y puede omitir picos más
breves. Las lecturas lógicas no equivalen a bytes físicamente leídos del SSD.

Una respuesta corta puede ser rápida y equivocada. Un programa con sintaxis
correcta puede resolver otra tarea. La repetición léxica tampoco demuestra por
sí sola incoherencia. Deben leerse las respuestas junto a la referencia.

Esta comparación muestra cuánto falta para el original tras recuperar. Para
atribuir una mejora exclusivamente al finetuning hay que comparar también con
la misma restricción sin adaptar; no se debe confundir con la diferencia entre
el original libre y el alumno restringido.
