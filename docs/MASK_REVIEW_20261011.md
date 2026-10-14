# Revisión inicial de máscaras (11/10/2026)

Fuente: `results/mask_intervention_20261010/report.json`, 42 respuestas de siete
variantes del modelo base congelado. Revisión del asistente, no ciega ni una
evaluación humana independiente. No se ejecutó el código generado.

## Hallazgos verificables

1. **Referencia MBPP problemática (`mbpp_test_135`).** El enunciado pide el ASCII
   de los caracteres de una cadena, pero el código de referencia retorna dentro
   de la primera iteración: para una cadena no vacía solo devuelve `ord(str1[0])`.
   Las respuestas que suman los códigos de todos los caracteres siguen una
   interpretación plausible del enunciado. No deben declararse incorrectas
   por discrepar con esta referencia. Excluir este ítem de futuras selecciones
   y calcular una sensibilidad de CE sin él; conservar las cifras originales.
2. **Número correcto, explicación incorrecta (`gsm8k_test_1082`).** En varias
   máscaras, incluida uniform36, aparece 2 horas pero se describe un trayecto
   de 4+4 millas, omitiendo la secuencia real 2+2+4. La coincidencia del resultado
   no acredita el razonamiento solicitado. `last2_free` expresa 2 horas en prosa
   pero no con la etiqueta esperada: separar el contenido del incumplimiento
   del formato, sin modificar retrospectivamente el marcador automático.
3. **Programación (`mbpp_test_114`).** Varias soluciones completas comienzan el
   extremo del substring en `i+2`, omitiendo los de longitud uno. Para la entrada
   `"0"` se infiere estáticamente que retornan 0 en vez de 1. Otras respuestas
   quedan truncadas. Compilar sintácticamente no demostraría corrección funcional.
4. **Biología.** Detectar la palabra de referencia no basta: aparecen nombres
   corruptos y explicaciones inventadas. El análisis futuro debe evaluar por
   separado el término solicitado y la explicación añadida.

## Qué se sostiene y qué no

Se verificó cero cargas durante inferencia en todas las variantes. Uniform36
obtuvo menor CE y repetición media en estos seis ítems que los otros brazos,
pero no demuestra recuperación de razonamiento ni es un ganador validado en
test independiente. La referencia problemática afecta a todos los brazos;
su exclusión es una sensibilidad posterior, no una nueva selección objetiva.

La evidencia apoya mantener controles de mismo presupuesto. Liberar primeras
capas requiere distinguir memoria adicional de ubicación, y redistribuir el
presupuesto original reduce candidatos en otras capas.

## Pendiente para cerrar la fase 1

- Obtener revisión humana independiente de las anotaciones del asistente y
  conservar desacuerdos. Las 42 respuestas ya tienen anotación inicial.
- Auditar más referencias antes de entrenar con ellas.
- Separar números/palabras detectados, formato y validez de explicación.
- Emitir sensibilidad de CE, registrar IDs defectuosos y recopilar todos los
  IDs históricos antes de crear nuevo desarrollo/test.
- Fijar márgenes y potencia/tamaño de evaluación antes de probar recuperación.

## Sensibilidad de CE sin la referencia ambigua

Análisis posterior, cinco preguntas; no usar para seleccionar configuraciones.

| Variante | CE original (6) | CE sin MBPP 135 (5) |
|---|---:|---:|
| old32 | 2.605 | 2.653 |
| recalibrated32 | 2.515 | 2.596 |
| shared8_category24 | 2.518 | 2.599 |
| first2_free | 2.441 | 2.469 |
| uniform36 | 2.412 | 2.446 |
| last2_free | 2.463 | 2.527 |
| first2_free_budget512 | 2.629 | 2.678 |

Cifras y hash en `results/mask_intervention_20261010/analysis/reference_sensitivity_20261011.json`.

## Anotación completa del asistente

`results/mask_intervention_20261010/analysis/qualitative_review_20261011.json`
contiene las 42 respuestas anotadas, con hash del informe y de cada texto.
Se separan respuesta central, validez de explicación, formato, calidad del
lenguaje, EOS y métricas automáticas originales. No se calcula un agregado
que oculte estas diferencias. Es revisión posterior no ciega del asistente;
no sustituye a evaluación humana independiente.

- En James, seis variantes dicen 2 horas, pero describen un recorrido distinto
  del solicitado. El acierto del número y el error de explicación se registran
  por separado. En Grandma ninguna proporciona una solución correcta completa.
- En programación, las siete soluciones de MBPP 114 están truncadas o tienen
  un error visible por inspección. Ninguna respeta la instrucción de entregar
  únicamente código en un bloque. No se ejecutó código generado.
- MBPP 135 conserva la marca de referencia ambigua. El código que suma ASCII
  es plausible, pero tres variantes inventan resultados para `Hello, World!`:
  la suma es 1129 y la longitud 13. El defecto de referencia no justifica esos
  ejemplos incorrectos ni el incumplimiento del formato.
- En biología, algunas respuestas centrales son correctas y la explicación
  inicial pertinente; otras inventan nombres o degeneran en repetición. No
  etiquetar todas como correctas por detectar `ecology` o `bacteria`, ni todas
  como totalmente incorrectas por tener defectos de redacción.

## Referencias nuevas: cautelas antes de ampliar

La inspección de candidatos de desarrollo detecta otros riesgos que no se han
usado en el piloto de escala: `gsm8k_test_1161` interpreta «otro 1/4 de su tierra»
como 1/4 del resto sin que el enunciado lo precise; `gsm8k_train_618` aplica una
reducción específica de una enfermedad a la probabilidad conjunta y presupone
disjunción no indicada. No usar esas referencias como juez automático sin revisión.
Además, este último está etiquetado `geometry` por contener «area», aunque es
un problema de probabilidad: la separación temática automática requiere revisión.

Estos hallazgos refuerzan la necesidad de revisar referencias y temas antes de
congelar un test confirmatorio. El piloto de nueve preguntas es solo desarrollo.
