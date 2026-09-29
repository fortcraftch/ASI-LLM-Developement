# Recuperar coherencia al fijar expertos

El primer piloto fijaba dos decisiones: los expertos y sus pesos de mezcla.
Además, la asociación temática procedía de un solo prompt por categoría. Eso
elimina gran parte del comportamiento con el que se entrenó OLMoE: la pareja
tema/experto no implica que esos expertos puedan sustituir a todos los demás.
Las alteraciones se acumulan a través de las 16 capas; la pérdida de fluidez
no demuestra por sí sola que la idea de mantener expertos residentes sea inviable.

## Hipótesis y controles

Distinguimos **K residentes** en GPU de **A activos** por token. El objetivo de
evitar swapping requiere fijar los residentes, no necesariamente los activos ni
sus pesos de mezcla. El router cuesta muy poco comparado con los pesos de los
expertos y puede seguir ejecutándose usando solo candidatos residentes.

`python -m asi routing-study` ejecuta estos contrastes con los mismos ejemplos:

| K residentes por capa | A activos | Mezcla |
|---:|---:|---|
| 8 | 8 | Constante, control del piloto anterior |
| 8 | 8 | Probabilidades originales del router para los residentes |
| 8 | 8 | Como la anterior, reescaladas a la suma del top-8 original en cada token |
| 16 | 8 | Top-8 dentro de los 16 residentes |
| 32 | 8 | Top-8 dentro de los 32 residentes |
| 32 | 16 | Top-16 dentro de los 32 residentes; cambia también la activación entrenada |

La modalidad `restricted` calcula softmax sobre todos los logits del router y
conserva esas probabilidades para los expertos elegidos dentro de los candidatos.
No normaliza automáticamente a suma uno: OLMoE no lo hace en su routing original.
`restricted_native_mass` estudia por separado la hipótesis de compensar la masa
de mezcla perdida. Ninguna variante carga expertos fuera del grupo residente.
Los controles pequeños verifican que permitir todos los expertos con el top-k
original reproduce la salida nativa y que restringir candidatos no provoca cargas.

## Ejecución

```bash
python -m asi routing-study --native-report results/olmoe_disk_native.json --calibration-per-category 2 --new-tokens 24 --output results/olmoe_routing_study_v1.json
python -m asi routing-study --summarize results/olmoe_routing_study_v1.json --output results/olmoe_routing_summary_v1.json
```

El informe nativo aporta continuaciones congeladas y debe corresponder al mismo
checkpoint y tokens de entrada. Se recalibran los rankings con ejemplos de train;
se evalúa un prompt de validación por cada una de las tres categorías. Aumentar
`--calibration-per-category` amplía la calibración, no la evaluación final.
El programa guarda resultados parciales; `completed=true` indica finalización.
`--summarize` produce un resumen JSON y una gráfica PNG sin cargar pesos.
Si una configuración no cabe con la reserva de VRAM prevista, queda registrada
como bloqueada. No modifica la LLM ni entrena adaptadores.

Medidas: textos de 24 tokens, NLL de continuaciones del modelo original, KL entre
distribuciones original/restringida en esas posiciones, residencia, cargas,
lecturas lógicas, tiempos y memoria. Menor NLL/KL implica mayor acuerdo con esa
referencia, no necesariamente mayor corrección. La VRAM máxima incluye generación
y evaluación de la continuación; sus tiempos de generación sí se separan.

Este es un diagnóstico pequeño: tres prompts y una pasada no permiten establecer
calidad general, una categorización óptima ni una mejora estable de rendimiento.

## Resultado del primer piloto

La batería `results/olmoe_routing_study_v1.json` completó los 18 casos: seis
configuraciones y tres prompts. Se usaron dos ejemplos de calibración por categoría,
24 tokens de generación y los ocho tokens originales congelados para NLL/KL.

| Residentes / activos | Política | NLL media (menor es mejor) | VRAM máxima GiB |
|---|---|---:|---:|
| 8 / 8 | Constante | 10,02 | 2,41 |
| 8 / 8 | Router restringido | 5,15 | 2,41 |
| 8 / 8 | Router + reescala | 5,91 | 2,41 |
| 16 / 8 | Router restringido | 3,46 | 3,91 |
| 32 / 8 | Router restringido | 2,18 | 6,91 |
| 32 / 16 | Router restringido | 1,99 | 6,91 |

La referencia nativa obtiene 0,23 sobre esas continuaciones. Todas las variantes
restringidas mantienen cero cargas y cero lecturas de expertos durante generación.
Recuperar pesos dinámicos y ampliar residentes reduce la desviación, pero **ninguna
configuración recupera fluidez estable en los tres ejemplos**. Con 32/8 aparece
un comienzo comprensible sobre condiciones de carrera; matemáticas y biología
siguen degenerando en repeticiones. Activar 16 no evita esas repeticiones.
Esto no justifica entrenar inmediatamente una receta final de ocho expertos:
conviene ampliar el corpus y estudiar expertos generales y sensibilidad por capa
antes de fijar la selección del alumno para distilación.

El control adicional `results/olmoe_routing_native24.json` utiliza los mismos
prompts y 24 tokens, manteniendo top-8 nativo y una caché GPU global de 512
instancias. Los tres textos conservan fluidez hasta el corte. Por ejemplo:
«Two threads can lose an update when they are trying to modify the same variable
concurrently because of a phenomenon known as race conditions». No se observa
el colapso repetitivo de las variantes restringidas. Esto respalda que el daño
procede de la intervención en los expertos; no es una prueba de exactitud general.
La caché de este control es mayor que la del piloto nativo inicial de ocho tokens:
no comparar sus tiempos como si solo hubiera cambiado la longitud.

## Cuándo y cómo estudiar fine-tuning

Tiene sentido después de elegir K/A y una política que conserve suficiente señal.
La propuesta siguiente es adaptación con la misma restricción usada al inferir:

1. Seleccionar expertos con un corpus representativo, incluyendo respuestas y
   texto general de lenguaje, no solo nombres o definiciones de categorías.
2. Conservar una evaluación independiente; comparar siempre contra el modelo
   original y contra la misma restricción sin ajuste.
3. Generar con el original continuaciones/logits para distilación. Puede hacerse
   por lotes y guardarse en disco para no mantener profesor y alumno simultáneamente.
4. Ajustar primero una cantidad pequeña de parámetros (router y/o adaptadores
   LoRA), con la máscara de residentes activa durante el entrenamiento. Combinar
   pérdida de lenguaje y distilación; incluir dominios generales para detectar olvido.
5. Medir si la recuperación compensa el coste adicional de adaptación, VRAM de
   entrenamiento, datos y almacenamiento por categoría. Los pesos ajustados deben
   ser checkpoints nuevos con identidad propia, nunca sobrescribir el original.

Ya está implementado el [piloto de ajuste solo de routers](RECOVERY.md), con
profesor offline, expertos congelados, restricción activa durante entrenamiento
y separación train/dev/test. Reutiliza los parámetros GPU persistentes de los
routers y valida su backward. El adaptador de disco/caché no sirve directamente
para entrenar los pesos de expertos: sus parámetros en `meta` no son las copias
que ejecuta la caché. La versión v2 añade LoRA persistente de la proyección de
salida de expertos y entrenamiento sobre varias selecciones; las correcciones
sobreviven a las expulsiones. El ajuste completo de las matrices densas sigue
pendiente. Véase el protocolo de recuperación para sus controles y limitaciones.

La recuperación mediante distilación tras poda tiene precedentes, pero no es
garantía para nuestro recorte temático:
[MoE-Pruner](https://arxiv.org/abs/2410.12013).
La arquitectura original se describe en [OLMoE](https://arxiv.org/abs/2409.02060).

Archivos: `asi/models/hf_olmoe.py` implementa las mezclas; `asi/experiments/routing_study.py`
orquesta los contrastes. Corresponden a los pasos 2.1, 2.2 y 2.3 del TFG.
