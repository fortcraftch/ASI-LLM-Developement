# Intervenciones de máscaras: calibración y capas libres

Protocolo exploratorio del 10/10/2026. Se ejecuta sobre **OLMoE original congelado**:
sin LoRA, sin entrenamiento y sin reutilizar adaptadores fuera de sus expertos
entrenados. El diagnóstico anterior encontró un 23,1 % de selecciones nativas
fuera de los pools de 32; eso motiva revisar la selección, no demuestra todavía
que las nuevas máscaras mejoren respuestas.

## Qué se compara

Todas las variantes activan **8 expertos por token y capa**, conservan la escala
de probabilidades del router original y precargan sus candidatos antes de responder.
Un experto residente se cuenta como un par `(capa, experto)`.

| Variante | Candidatos por capa | Total residente |
|---|---|---:|
| old32 | Pools anteriores, 32 | 512 |
| recalibrated32 | 32 elegidos con calibración ampliada | 512 |
| shared8_category24 | 8 comunes + 24 específicos, sin duplicados | 512 |
| first2_free | 64 en capas 0–1; 32 en las demás | 576 |
| uniform36 | 36 en todas | 576 |
| last2_free | 64 en capas 14–15; 32 en las demás | 576 |
| first2_free_budget512 | 64 en 0–1, 28 en 2–7, 27 en 8–15 | 512 |

Los 64 pares adicionales cuestan aproximadamente **768 MiB** en BF16.
`first2_free` frente a `uniform36` y `last2_free` permite estudiar dónde se
concentra la memoria. La variante de 512 pares conserva el presupuesto, pero
reduce candidatos en capas posteriores: ese cambio también puede afectar calidad.
«Libre» significa que el router puede elegir cualquiera de los 64 expertos de
esa capa, todos residentes. No significa hacer swapping en cada token.

## Datos y reglas previas a medir

- 96 referencias de entrenamiento (32 por categoría), con semilla fija.
  Se priorizan las referencias de la calibración antigua que sigan siendo aptas;
  no se promete contener todas si fueron rechazadas o exceden 384 tokens.
- Se excluyen los ejemplos rechazados en el piloto. No se eligen máscaras usando test.
- Se contabilizan selecciones del router original sobre prompt, respuesta y EOS;
  se excluye padding. Cada categoría se normaliza por sus tokens para el núcleo.
- Núcleo común: los ocho expertos con mayor **mínimo de frecuencia normalizada
  entre categorías**, por capa; desempate por identificador. Es una heurística,
  no una identificación causal de «driver experts».
- Los 24 específicos restantes siguen la frecuencia de la categoría, excluyendo
  el núcleo. Las demás máscaras usan frecuencia descendente y desempate por ID.
- Se reutilizan las seis preguntas del diagnóstico emparejado, ya observadas:
  son un conjunto exploratorio, no test final independiente.
- Generación greedy, hasta 256 tokens o EOS. CE sobre la misma respuesta de
  referencia; repetición de trigramas, textos completos, EOS, tiempos, memoria
  y cargas. No equiparar menor repetición con mayor corrección.
- La preparación de pools se registra aparte. Se exige **cero cargas de expertos
  durante CE y generación**. No hay estado de optimizador en las medidas de VRAM.

El artículo [What Gets Activated](https://arxiv.org/html/2601.10159v2)
destaca el papel de los primeros **tokens**. Dejar libres las dos primeras
**capas** es una hipótesis propia que este experimento pretende contrastar.

## Ejecutar y reanudar

```bash
python -m asi mask-intervention --output results/mask_intervention_20261010
python -m asi mask-intervention --output results/mask_intervention_20261010 --resume
```

Guarda progreso atómicamente después de cada lote de calibración y cada respuesta.
Una interrupción repite únicamente el lote/respuesta no guardado. Se verifican los
hashes del modelo, corpus, pools y diagnóstico anterior antes de reanudar.
`--calibrate-only` permite terminar tras la calibración; quitarlo con `--resume`
continúa la evaluación. Cambiar batch o presupuesto de tokens exige otro directorio.

Al terminar genera `analysis/index.html`, figuras PNG/PDF y `summary.json`.
También puede reconstruirse sin cargar el modelo:

```bash
python -m asi mask-intervention-report results/mask_intervention_20261010/report.json
```

El informe parcial solo compara variantes con las seis preguntas completas.
Las conclusiones necesitarán más preguntas y revisión semántica; este protocolo
no sustituye los benchmarks ni prueba una arquitectura universalmente superior.

## Control aleatorio añadido el 16/10/2026

`--random-seeds 1337 2027 31415` añade tres conjuntos uniformes sin reemplazo, 32 candidatos por capa. Cada semilla comparte máscara entre categorías: control ciego a la categoría. Conserva 512 pares residentes, 8 activos y pesos nativos; no se elige la mejor semilla mirando calidad. Usar salida nueva: la identidad de protocolo pasa a v2 y no puede reanudar un informe v1 como si fuera equivalente. Plan congelado en `results/random_mask_control_20261016/plan.json`; ejecución terminada con 18 respuestas. La opción sin `--random-only` también ejecutaría los siete brazos históricos; no usarlos de nuevo para reconstruir el informe. Las seis preguntas históricas son diagnóstico, no test independiente.

Ejecución incremental disponible: `--random-only --reuse-calibration results/mask_intervention_20261010/report.json`. Verifica identidad de modelo/fuentes, protocolo, cobertura y separación antes de copiar estadísticas; no copia respuestas. La tanda completada usó tres semillas y seis prompts (18 respuestas), salida `results/random_mask_control_20261016/run`, logs en la carpeta superior. No reejecuta los siete brazos. Comparación con históricos solo de calidad exploratoria; timings no emparejados temporalmente.

Reconstruir la comparación sin inferencia: `python -m asi mask-intervention-report results/random_mask_control_20261016/run/report.json --compare-with results/mask_intervention_20261010/report.json`. Rechaza informes incompletos, prompts distintos, variantes duplicadas y transferencias dentro de inferencia. Agrupa por variante y presupuesto; no agrega tiempos ni trata semillas como preguntas independientes. CE/EOS requieren revisión de respuestas para valorar calidad.


### Control global calibrado (16/10/2026)

`--global-only` ejecuta únicamente `global32`: el promedio de frecuencias por
 token de cada categoría de calibración determina 32 expertos por capa. Todas
las categorías usan la misma máscara; las categorías tienen igual peso aunque
sus textos tengan distinta longitud. Empates por índice de experto. Este control
separa selección informada global de selección condicionada a la categoría.

```powershell
python -m asi mask-intervention --output results/global_mask_control_20261016/run --global-only --reuse-calibration results/mask_intervention_20261010/report.json
```

Plan congelado antes del lanzamiento en `results/global_mask_control_20261016/plan.json`.
Ejecución terminada (6/6); no repetir para reconstruir resultados. Seis preguntas diagnósticas conocidas,
top-8, presupuesto 512 pares capa/experto, pesos congelados. No es confirmación
independiente ni recuperación; no comparar tiempos históricos como efecto causal.


## Resultados consolidados y lo que falta

| Control, todos con 512 pares residentes | CE media | Estado |
|---|---:|---|
| Categorías recalibradas | 2.515 | Completo |
| Núcleo 8 + categoría 24 | 2.518 | Completo |
| Categorías anteriores | 2.605 | Completo |
| Primeras dos capas libres, presupuesto igualado | 2.629 | Completo |
| Global calibrado sin categoría de entrada | 2.727 | Completo |
| Aleatorias 1337 / 2027 / 31415 | 3.242 / 3.583 / 3.064 | Completo |

La selección global usa igual peso por categoría de calibración, pero no consulta
la categoría al responder. La ventaja descriptiva de categorías no demuestra
especialización causal ni generalización. No se ha recuperado la calidad de
las generaciones. Revisiones completas y textos: `results/random_mask_control_20261016/REVIEW.md`
y `results/global_mask_control_20261016/REVIEW.md`, con los informes crudos conservados.

No queda pendiente repetir estos controles. Falta confirmar con datos nuevos y
protocolo reservado, comparar decodificación de forma controlada y posteriormente
recuperación con referencias verificadas. El paso de validación de corpus está
aplazado hasta que termine el entrenamiento de homelab por indicación de Marc.
El paper del 16/10 incluye la tabla y límites; ver `paper/README.md` para el PDF vigente.
