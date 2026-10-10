# Mapas offline de expertos — 9 octubre 2026

Actualización tras finalizar el piloto: ya hay mapas t-SNE de DeepSeek y de
pesos del router OLMoE antes/después, con sus límites de interpretación, en
[RECOVERY_PILOT_ANALYSIS.md](RECOVERY_PILOT_ANALYSIS.md). Los mapas originales
de este documento siguen describiendo datos archivados de calibración.

Generación: `python -m asi.analysis.expert_maps` desde la raíz del proyecto.
Salida: `results/expert_maps_20261009`, siete figuras en PNG y PDF,
perfiles numéricos de DeepSeek en CSV y fuentes con SHA-256 en `report.json`.
Solo lee tres informes JSON archivados. No carga checkpoints, no importa
PyTorch y limita los hilos numéricos a uno. No modifica el piloto.

## Qué muestran

| Figura | Datos | Interpretación |
|---|---|---|
| `deepseek_routing` | Checkpoint 07874, tres prompts, 11 capas × 8 expertos | Fracción de asignaciones top-2 por experto, separando prefill y generación |
| `deepseek_pca` | Los mismos 88 pares capa–experto; seis características | Proximidad entre perfiles de selección observados |
| `olmoe_selection_32`, `olmoe_selection_16` | Calibración v3: 24 ejemplos de entrenamiento | Máscaras seleccionadas para programación, biología y matemáticas |
| `olmoe_overlap_32`, `olmoe_overlap_16` | Intersección/unión de esas máscaras | Cuánto se comparten los pares capa–experto entre categorías |
| `asi_pool_structure` | Configuración archivada del checkpoint 37595 | Reparto impuesto de 16 expertos en ocho pools, repetido en 11 capas |

En DeepSeek se divide cada vector de conteos por su suma: es proporción de
asignaciones, no probabilidad del router. Cada posición aporta dos asignaciones.
La generación cuenta la última posición de cada forward, sin sumar prefijos
recalculados. PCA centra las seis características, sin estandarizarlas: los dos
ejes conservan un 84,7% de su variación. Los ceros observados no significan que
un experto nunca sea útil. Empates de preferencia se resuelven por orden del
prompt; los nunca observados se marcan aparte. Tres prompts no validan una
especialización por dominio. Cada punto identifica un experto de una capa;
el experto 0 de dos capas distintas son dos unidades diferentes.

En OLMoE el archivo conserva selecciones top-N, no todos los conteos ni las
representaciones. Gris significa excluido de la selección, NO activación cero.
No proyectamos estas máscaras como si fueran embeddings. No usamos sus pesos
almacenados como probabilidades originales, porque fueron reajustados al
seleccionar el subconjunto. Estos mapas son de calibración, no del resultado
del piloto de recuperación que sigue en marcha.

En ASI las franjas de colores proceden de la arquitectura y del orden de pools
guardado en el checkpoint. No demuestran que se haya aprendido especialización.
Los expertos compartidos adicionales no están representados en estos mapas.

## Primeras observaciones

- OLMoE con 32 residentes selecciona 512 pares por categoría. Hay 250 comunes
  a las tres categorías y 775 en su unión. Con 16 son 256 por categoría,
  65 comunes y 495 en la unión.
- Jaccard programación–matemáticas: 54,2% con 32 y 45,5% con 16;
  programación–biología: 41,6% y 17,2%; biología–matemáticas: 51,9% y 25,2%.
  Son medidas descriptivas de estas máscaras, no evidencia de superioridad
  sobre selecciones aleatorias ni de causalidad en la calidad.
- Esto motiva comprobar si conservar un núcleo compartido mejora la recuperación.
  Todavía es una hipótesis: requiere igual presupuesto y un control comparativo.

## Después del piloto y de su análisis

1. Recoger perfiles completos sobre las mismas entradas reservadas, antes y
   después de adaptar router/LoRA, con condiciones de generación iguales.
2. Comparar dentro de cada arquitectura utilizando una proyección conjunta;
   añadir activaciones funcionales, normalización explícita y estabilidad entre
   semillas antes de aplicar t-SNE/KMeans. La dependencia sklearn no está
   instalada actualmente; estos mapas usan PCA de NumPy, no son una réplica
   del t-SNE del artículo.
3. Medir cambios de uso, balance y solapamiento junto a coherencia, corrección,
   repetición y recursos. No comparar coordenadas entre modelos distintos.
4. Identificar expertos de influencia causal mediante intervenciones y cambio
   de salida/KL. Solo entonces tendría sentido añadir estrellas como los
   «driver experts» del artículo; los clusters por sí solos no los identifican.
