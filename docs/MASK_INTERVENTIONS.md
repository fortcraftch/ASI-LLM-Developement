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
