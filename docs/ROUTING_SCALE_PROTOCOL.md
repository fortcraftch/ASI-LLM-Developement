# Protocolo de escala de routing (fase 2)

Estado: piloto de desarrollo **completado**, 36 respuestas sobre nueve preguntas.
Plan congelado en `configs/routing_scale_dev_v1.json`. Este piloto exploratorio
no certifica el test final ni cierra la auditoría histórica.

Para probabilidades `p = softmax(logits)` calculadas sobre los 64 expertos,
elegir los ocho mayores dentro del mismo conjunto residente R:

| Política del runtime | Peso ejecutado por experto | Pregunta |
|---|---|---|
| `restricted` | p, sin normalizar en OLMoE original | Control actual |
| `restricted_unit_mass` | p / suma de p seleccionadas | ¿Ayuda normalizar la mezcla a uno? |
| `restricted_native_mass` | p / suma seleccionada × masa top-8 sin máscara | ¿Ayuda conservar la masa que usaría el router sobre este estado? |

La selección inicial sobre **el mismo estado oculto** es idéntica; solo cambian
los pesos. En una generación completa los estados divergen y después pueden
cambiar las selecciones. La masa nativa se calcula sobre el estado actual del
brazo restringido: no es un contrafactual de los estados del modelo original.

Diseño inicial: mismos pesos base congelados, máscara old32 prefijada, 8 activos,
32 residentes por capa, prompts/referencias idénticos y sin LoRA. No elegir
uniform36 usando las seis preguntas ya observadas. Añadir original como
referencia de calidad, distinguiendo su memoria y tráfico.

Dos lecturas complementarias:

1. Diagnóstico de gate sobre estados idénticos: IDs, pesos y masas; prueba del
   mecanismo, no de calidad.
2. Inferencia completa: CE de referencia, salida libre, razonamiento/corrección,
   instrucciones, repeticiones/EOS, tiempos, memoria y cero cargas durante las
   respuestas restringidas. Registrar preparación por separado.

No mezclar inicialmente cambios de máscara, escala, adaptación y decodificación.
Posteriormente cruzar las mejores opciones de desarrollo con controles de
selección aleatoria/compartida y múltiples semillas. Congelar el test final
antes de ejecutar ese cruce. Los umbrales y el tamaño de muestra final siguen
pendientes en la hoja de ruta; este documento no declara recuperación.

## Piloto de desarrollo v1

Nueve preguntas del split propio de desarrollo, tres por categoría, con revisión
de referencia previa a la generación. No reutiliza los seis diagnósticos de
máscaras. La selección, referencias completas, hashes y observaciones de revisión
están congelados en el JSON. Se conservan las referencias originales; la CE
también refleja su redacción y no es una medida directa de corrección.

El manifiesto antiguo corrupto sigue sin validar: sus 424 shards coinciden con
el inventario anterior y ninguno corresponde a estos nueve IDs, pero no podemos
certificar metadatos perdidos. Se documenta esta limitación en cada resultado.
No se utilizan las probabilidades de ese profesor. El control de IDs, duplicados
exactos y plantillas no demuestra ausencia de duplicados semánticos o exposición
en preentrenamiento. No presentar el piloto como test independiente.

```bash
python -m asi routing-scale --dry-run
python -m asi routing-scale
# Tras una interrupción, conserva las respuestas completas:
python -m asi routing-scale --resume
```

Se guardan hasta 36 respuestas, sin entrenamiento, con generación greedy hasta
EOS o 256 tokens. Cache de 512 pares capa/experto; original puede hacer swapping,
los tres brazos restringidos deben mantener cero cargas durante CE/generación.
La preparación se mide aparte. El original es referencia de calidad y no una
referencia de velocidad con todos los expertos residentes. El orden fijo, una
sola repetición, el calentamiento por CE y las longitudes variables impiden sacar
conclusiones firmes de velocidad. Para ello se mantiene el benchmark específico.

El plan verifica que no cambien fuentes ni filas; la reanudación verifica además
código, versiones, modelo y GPU. No sobrescribir el inventario histórico que el
plan fija por hash: futuras auditorías deben usar otra carpeta de salida.

Criterio de lectura: revisar respuesta, explicación, formato y repetición por
separado. Una reducción de CE o más números detectados no autoriza afirmar
recuperación. Este piloto sirve para decidir qué comparaciones ampliar en
desarrollo; márgenes y tamaño del test confirmatorio se fijarán antes de usarlo.

## Resultados del 11/10/2026

Informe: `results/routing_scale_dev_20261011/report.json`. Las nueve preguntas
usan los mismos tokens de entrada en cada brazo. Se verificaron las 36 filas,
32 residentes por capa en los tres brazos restringidos y cero cargas durante
sus 27 ejecuciones de CE/generación. Sin LoRA ni entrenamiento.

| Política | CE media | Repetición trigramas | EOS | Mediana decode tokens/s |
|---|---:|---:|---:|---:|
| Original, caché limitada | 1,922 | 23,65% | 5/9 | 0,54 |
| Restringido, escala existente | 2,359 | 36,32% | 5/9 | 7,27 |
| Restringido, masa uno | 7,767 | 9,06% | 1/9 | 7,24 |
| Restringido, masa nativa | 2,422 | 42,62% | 3/9 | 7,18 |

Normalizar a uno el OLMoE ya entrenado no soluciona este piloto: aumenta mucho
la CE y aparecen respuestas ilegibles. **Menos repetición no implica mejor
calidad**: texto incoherente también puede variar mucho. Restaurar masa nativa
no muestra mejora agregada de CE o repetición aquí. La revisión semántica inicial
del asistente está registrada por respuesta; no convertir nueve ejemplos en una
conclusión general. Anotaciones y huellas de los 36 textos en
`results/routing_scale_dev_20261011/analysis/qualitative_review.json`.
No es una evaluación humana independiente ni se ejecutó código generado.

Todos los brazos tienen una caché de 512 pares capa/experto, hasta 6 GiB de pesos
de expertos y aproximadamente 6,95 GiB de asignación CUDA máxima. Por tanto,
**este experimento no demuestra reducción de VRAM frente a su propio control**:
el control también tiene memoria limitada y hace swapping. El control registra
63.752 cargas, 747,09 GiB de tráfico acumulado host→GPU durante CE y generación;
no son gigabytes distintos del modelo ni espacio ocupado en disco. Las
restricciones registran cero cargas en ese intervalo.

La preparación del primer brazo restringido suma 18,66 GiB H2D; los otros dos
reutilizan su selección y registran cero preparación adicional por el orden del
piloto. No presentar ese cero como coste de arranque propio de cada política.
Velocidades descriptivas con longitudes variables, calentamiento y sin repeticiones:
no extrapolar el cociente a un benchmark general ni a igual calidad.

Gráficas, resumen y textos completos:
`results/routing_scale_dev_20261011/analysis/index.html`.
Se regeneran con el generador común, sin duplicar una herramienta de gráficos:

```bash
python -m asi mask-intervention-report results/routing_scale_dev_20261011/report.json
```

Decisión de desarrollo: conservar la escala restringida existente como control
de recuperación; no activar masa uno por defecto en modelos ya entrenados.
El nuevo entrenamiento pareado de 124M sí puede usar la misma normalización
desde la inicialización en ambos brazos: es otra pregunta y otro protocolo,
documentado en `MATCHED_MOE_124M.md`.
