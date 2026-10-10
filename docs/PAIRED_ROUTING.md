# Routing emparejado: diagnóstico del piloto

## Resultado completado

Se ejecutaron las 24 combinaciones: seis secuencias de 80–238 tokens por cuatro
variantes. El informe está completo en `results/routing_paired_20261009/report.json`;
figuras PNG/PDF y resumen en su subdirectorio `analysis`.

| Variante | Coincidencia top-8 con original | Masa ejecutada media |
|---|---:|---:|
| Original | 100% | 0,425 |
| Restringido sin adaptar | 74,1% | 0,370 |
| CE200 | 71,6% | 0,362 |
| CE+KL50 | 73,0% | 0,367 |

La máscara deja fuera el **23,1% de asignaciones top-8 del original**:
12,1% en programación, 28,4% en biología y 25,2% en matemáticas. La masa absoluta
de probabilidad correspondiente a esas asignaciones excluidas es 0,088 de media;
no confundirla con el 23,1% de conteos. Las selecciones sustitutas recuperan parte
de la masa; no se resta directamente 0,088 a la salida final del modelo.

CE y CE+KL no acercan el top-8 ejecutado al original en este conjunto, pese a
mejorar formato/EOS en la evaluación anterior. No se deduce que el nuevo routing
sea peor por diferir: la adaptación puede necesitar otras combinaciones.
El descenso de masa puede ser relevante para la escala de los outputs MoE,
pero no mide la norma del residual ni demuestra una causa de incoherencia.
No hubo cargas de expertos durante ninguno de los 18 forwards restringidos.

La siguiente intervención defendible es comparar, a presupuesto fijo de 32,
las máscaras actuales con máscaras recalibradas en más datos de entrenamiento
y con un pequeño núcleo compartido que sustituya a expertos menos utilizados.
Separar ese cambio de cualquier entrenamiento adicional. Reservar nuevas
preguntas para evaluación final. No reducir a 16 antes de comprobar la cobertura
y evaluar si estos cambios recuperan tareas, no solo formato.

Entrada unificada: `python -m asi routing-paired`.
Reanudación: añadir `--resume`; conserva trazas originales completas y vuelve
a ejecutar los brazos restringidos. No borra checkpoints ni entrena pesos.
Análisis offline: `python -m asi routing-paired-report`.

Se eligen determinísticamente dos referencias por categoría del split test del
corpus, excluyendo los 276 IDs del piloto, calibración y ejemplos previamente
rechazados. Son ejemplos **diagnósticos nuevos**, no un futuro test final: una
vez inspeccionados no deben utilizarse para seleccionar y evaluar a la vez.
Las referencias no se califican aquí por calidad. Se limita la secuencia completa
a 384 tokens y se congela su tokenización en el informe.

Cada variante recibe exactamente prompt + respuesta de referencia + EOS, en un
forward teacher-forced, batch 1, sin muestreo ni generación libre. Se comparan:
original, 32 residentes sin adaptar, CE200 y CE+KL50. Todos usan ocho expertos
activos por token. Se mantienen las matrices del router en FP32 y cómputo bajo
autocast BF16, coherente con evaluación de recuperación. Adaptaciones rank8.

Por token/capa se conserva el top-8 ejecutado. También se guardan conteos por
experto, probabilidades medias, masa ejecutada, margen entre puestos 8 y 9 y
fracción/masa del top-8 sin restricción que queda fuera del pool. En el original,
esto mide qué rutas elimina la máscara; en otros brazos, la preferencia sin
restricción se calcula **sobre sus propios estados ocultos**, que ya difieren.
La coincidencia de conjuntos top-8 no depende del orden de sus ocho elementos.

El análisis pondera por tokens y capas, no por ejemplo: las secuencias largas
pesan más. El primer mapa muestra asignaciones ejecutadas, no representaciones
semánticas. No se confunde masa de probabilidad con fracción de asignaciones.
No se suman los expertos de capas distintas como si fueran el mismo experto.

## Límites

- Mismos tokens no equivale a mismos estados ocultos: es una comparación del
  efecto acumulado de cada variante, no una intervención aislada sobre el router.
- La instrumentación sincroniza GPU/CPU. Sus tiempos NO son un benchmark de
  velocidad. La prueba tampoco sustituye la evaluación de respuestas completas.
- Las rutas perdidas permiten formular hipótesis, no demostrar por sí solas qué
  experto causa un error. Para eso se requieren intervenciones a presupuesto igual.
- Modelo original verificado por hashes, caché GPU de 512 pares capa–experto,
  caché RAM hasta 2 GiB, presupuesto de proceso 12 GiB, expertos restantes en disco.
- Cada secuencia restringida comprueba que no haya cargas durante el forward.

## Organización de archivos

`expert_maps.py` conserva mapas históricos estructurales/de calibración;
`recovery_pilot_report.py` analiza el piloto terminado;
`routing_paired.py` ejecuta este diagnóstico;
`routing_paired_report.py` solo lee sus resultados. No son cuatro lanzadores del
mismo experimento: así regenerar una figura no vuelve a cargar un modelo.
La lectura con hashes y exportación PNG/PDF duplicadas se han centralizado en
`asi/analysis/evidence.py`. Se mantienen las evidencias históricas necesarias.

La fuente local del paper se actualiza con `python paper/build_evidence.py`;
incluye el diagnóstico únicamente cuando su resumen está completo. Overleaf y
el PDF exportado requieren sincronización/compilación aparte.
# Continuación: intervenciones de máscaras

El protocolo ejecutable de recalibración ampliada, núcleo compartido y capas
libres está en [MASK_INTERVENTIONS.md](MASK_INTERVENTIONS.md). Se estudia primero
el modelo base congelado, sin confundir cambios de máscara con nuevo fine-tuning.

