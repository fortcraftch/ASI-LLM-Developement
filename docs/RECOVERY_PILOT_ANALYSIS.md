# Piloto de recuperación y mapas — 9 octubre 2026

Continuación implementada: [routing emparejado](PAIRED_ROUTING.md), con trazas
por token sobre entradas idénticas. Su estado y resultados están en
`results/routing_paired_20261009`; solo interpretar informes con `completed=true`.

## Resultado principal

Hay recuperación parcial de formato, terminación y reducción de repetición.
No se ha recuperado el razonamiento del original en esta muestra.
No seleccionar un ganador definitivo ni entrenar más usando estos 18 ejemplos
como validación: ya se han inspeccionado y pasan a ser evidencia diagnóstica.

Informes originales: `results/recovery_200_screened_20261008`.
Galería, **las 72 respuestas completas**, seis figuras PNG/PDF y datos:
`results/recovery_pilot_analysis_20261009/index.html`.
Generador: `python -m asi.analysis.recovery_pilot_report --tsne`.
La dependencia opcional está aislada en `results/analysis_dependencies`
(scikit-learn 1.7.2, joblib 1.5.2, threadpoolctl 3.6.0); no se modificó el venv
de entrenamiento. Se cargan solo adaptaciones pequeñas en CPU, no el modelo.

Se verificaron hashes de informes e inputs congelados, finalización, 200 pasos
por brazo, igualdad del orden de ejemplos/máscaras, los mismos 18 IDs de test,
salidas iniciales idénticas y matrices iniciales del router idénticas.

| Variante | CE test | Matemáticas | EOS | Repetición mediana / media* | Tokens medianos |
|---|---:|---:|---:|---:|---:|
| Original | 2,277 | 4/6 | 12/18 | 9,7% / 11,1% | 213,5 |
| Restringido 32 | 2,350 | 0/6 | 5/18 | 18,2% / 32,1% | 256 |
| CE, paso 200 | 1,219 | 1/6 | 17/18 | 0,0% / 5,1% | 39 |
| CE+KL, paso 50 | 1,518 | 1/6 | 17/18 | 1,9% / 8,2% | 68 |

*Secuencias de cuatro tokens repetidas. El selector original utiliza trigramas;
no confundir ambas métricas. Cero en la mediana no elimina fallos individuales.
CE es media por ejemplo sobre referencias bajo teacher forcing. No se compara
`mean_loss` de CE+KL con CE: aquella mezcla dos objetivos.

## Lectura de las respuestas

Revisión cualitativa del asistente contrastada con prompt y referencia; no
anotación humana independiente ni test oficial de código. No se ejecutó código
generado. La galería permite revisar todos los textos, no solo casos favorables.

### Matemáticas: seis preguntas

| ID abreviado | Esperado | Original | CE | CE+KL |
|---|---:|---:|---:|---:|
| train_3671, bicicleta | 6 h | 6 | 4 | 2 |
| test_1205, perros | 4 | 0,33 | 5 | -2 |
| train_3042, velocidad | 63 | 63 | 63 | 63 |
| test_1120, viajes | 560 | 280 | 280 | 280 |
| train_4992, combustible | 4 | 4 | -337 | 31 |
| train_1317, corredores | 240 s | 240 | bucle, sin resultado correcto | 90 |

El original tampoco es infalible y puede añadir explicaciones erróneas después
de acertar. Por ejemplo, en bicicleta repite el razonamiento; CE da una respuesta
corta pero omite sumar las dos primeras horas. Menos repetición no equivale a
mejor razonamiento. Los prefijos train/test de estos IDs pertenecen al dataset
de origen: los 18 están en el split test de nuestro piloto, sin usarse para ajustar.

### Programación: seis preguntas

| Tarea | Observaciones de las salidas |
|---|---|
| Primera palabra repetida | Original, CE y CE+KL buscan la primera palabra que aparece más de una vez; no la primera repetición detectada como la referencia. En `a b b a` devuelven `a`, la referencia `b`. Prompt algo ambiguo: declarar el contrato antes del benchmark funcional. |
| Regex `a...b` | CE utiliza `search`, como la referencia; original y CE+KL utilizan `match`, que exige empezar ahí. La especificación debe aclarar coincidencia interna o desde inicio. Los ejemplos explicativos del original contienen errores. |
| Concatenación repetida | CE solo compara longitudes; falla en `abab, ab`. Original y CE+KL también tienen lógica incorrecta; el indexado puede salir de rango. |
| Cadena binaria | CE y CE+KL comprueban los caracteres; original impone incorrectamente longitud par. Falta acordar el caso vacío y el tipo de retorno, que difieren de la referencia. |
| Ordenar cada sublista | CE ordena la lista exterior por longitud: tarea equivocada. El código principal de original y CE+KL sí ordena cada sublista; el ejemplo explicado por el original es inconsistente. |
| Longitud de supersecuencia mínima | CE devuelve dos veces la longitud menor: `ab, ab` produciría 4 en vez de 2. CE+KL devuelve una cadena, no la longitud pedida. Original termina con código truncado. |

La sintaxis y el cumplimiento de un bloque de código no son corrección funcional.
Hace falta un contrato claro y pruebas de casos normales/límite antes de asignar
una tasa de acierto de programación.

### Biología: seis preguntas

| Tema | Observaciones |
|---|---|
| Ecosistema | CE da la respuesta esperada de manera breve, con errata. CE+KL añade palabras inventadas y reiteraciones. Original responde pero se extiende. |
| Agua dulce | La referencia pide `limnology`; CE escribe `pology`, CE+KL `Aquogy/Aquology`. Original usa un término más amplio, `Hydrobiology`; no resolver equivalencia semántica con substring. |
| Variación de especies | CE usa `biodiversity`, como la referencia; original y CE+KL hablan de `species richness`, concepto más específico. Evitar una puntuación automática definitiva en este enunciado. |
| Densidad poblacional | CE y CE+KL responden el concepto. Original añade un ejemplo aritmético incorrecto: 10.000/100 no es 100.000. |
| Hongos | CE responde material vegetal muerto, más estrecho que la referencia; CE+KL materia orgánica muerta con errores de forma. Original explica el mecanismo de absorción más que la fuente solicitada. |
| Poblaciones sin descendencia fértil | CE contesta `no offspring`, sin la conclusión solicitada. CE+KL se aproxima a especiación, pero repite y deforma términos. Original da aislamiento reproductivo y especies separadas. |

## Recursos y velocidad

Decodificación mediana: original 0,51 tokens/s; restringido 6,67; CE 6,71;
CE+KL 6,72. El descenso de duración de 38,6 s a 6,15 s entre restringido y CE
se explica sobre todo por pasar de 256 a 39 tokens medianos y terminar con EOS;
no por multiplicar la velocidad de cálculo tras el fine-tuning.

En las 18 generaciones, las variantes restringidas registran cero cargas de
expertos y cero H2D de expertos durante generación. El original registra
105.098 cargas y 1.231,6 GiB acumulados transferidos (repetidos, NO tamaño del
modelo). Se excluye preparación/cambio de contexto. Esto apoya la eliminación
de swapping por token para esta implementación y estos prompts.

**La memoria no es una comparativa limpia de inferencia**: los brazos adaptados
registran aproximadamente 7,24 GiB, pero conservan potencialmente optimizador
y gradientes del entrenamiento; original 6,95 GiB se midió en otro proceso.
No afirmar que la arquitectura adaptada necesita más VRAM ni ahorro relativo
a partir de estos valores. Medir en procesos nuevos antes de publicarlo como
benchmark de memoria. Tampoco extrapolar la velocidad a otros backends.

## t-SNE: qué se ha representado

- **DeepSeek:** 88 pares capa–experto, seis fracciones de routing de tres
  prompts, con semillas 0/42 y perplexity 10/30. KMeans k=4 se calcula en el
  espacio original, no en el dibujo. La estructura local se conserva razonablemente
  (trustworthiness@5 de 0,958–0,976), pero no demuestra especialización general.
- **OLMoE:** una fila de 2.048 pesos del router por experto, normalizada L2.
  Mapas separados para capas 0, 7 y 15; cada uno ajusta conjuntamente 64 expertos
  × tres estados (inicial, CE200, CE+KL50). PCA a 30 dimensiones, después t-SNE;
  colores KMeans k=4 en PCA. Dos configuraciones: (semilla0, perplexity15) y
  (semilla42, perplexity30). No es un estudio factorial completo de estabilidad.
  Los colores no corresponden a categorías ni son comparables entre capas.
- **ASI:** se conserva el mapa estructural anterior. Proyectar sus etiquetas
  de pools no produciría evidencia nueva de aprendizaje; faltan activaciones.

Los tríos de puntos del mismo experto están próximos: los ajustes no reorganizan
por completo esta geometría del router. El cambio relativo L2 mediano de sus filas
es 9,21% para CE200 y 5,09% para CE+KL50. No atribuir la diferencia a KL: también
son distintos números de pasos. La normalización del mapa omite cambios de norma;
el heatmap de cambios usa pesos sin normalizar. No incluye el efecto de LoRA.
Trustworthiness alto puede estar favorecido por los tríos casi duplicados;
no es una medida de capacidad, causalidad o calidad de categorías.

K=4 es exploratorio, no un número de dominios descubierto. La cercanía en t-SNE
no mide directamente cooperación, y las distancias globales/islas dependen de
la proyección. Ningún punto se marca como experto causal. Metodología de la
herramienta: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

## Qué conviene hacer ahora

1. **Activaciones emparejadas**, sobre un conjunto diagnóstico fijo distinto
   del futuro test: mismos tokens teacher-forced para original, 32 sin adaptar,
   CE y CE+KL. Medir frecuencias, masas del router, márgenes top-k y coincidencias
   por capa. No usar continuaciones diferentes como si fueran entradas idénticas.
2. Contrastar los fallos de matemáticas y código con cuánto routing del original
   queda fuera de los 32 permitidos, y con cambios de las salidas de expertos.
   Es correlación; validar causalidad con intervenciones posteriores.
3. Medir memoria y rendimiento en procesos frescos, igual longitud fija además
   de EOS. Mantener medición separada de preparación/cambio de categoría.
4. Revisar referencias y aumentar el dev numérico: el selector decide con solo
   dos preguntas matemáticas. CE+KL mejora CE de dev mientras empeora su proxy
   de generación; no prolongar a ciegas. Comparar pasos iguales antes de concluir
   que KL no sirve. Probar peso de KL menor como hipótesis, no como resultado.
5. Solo después probar 16 residentes o un núcleo compartido a presupuesto igual.
   Hay una mejora real de usabilidad, pero aún falta conservar resolución de tareas.
