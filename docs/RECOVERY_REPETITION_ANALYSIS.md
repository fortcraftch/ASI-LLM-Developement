# Análisis de las repeticiones — 07/10/2026

**Conclusión:** se ha recuperado parte de la pérdida y del seguimiento de formato,
pero no la generación fiable. Los bucles persisten con 32 y son especialmente
graves en las respuestas de 16. No tenemos todavía una descomposición causal
que permita asignar todo el daño a un único motivo.

Este análisis es de **OLMoE-1B-7B-0924-Instruct**, no del modelo propio de 124M.
No justifica atribuir los errores de ambos modelos a una misma falta de parámetros.

## Evidencia observada

Las tres ejecuciones están completas en
[`results/generation_recovery_20261007`](../results/generation_recovery_20261007).
Son tres preguntas reservadas, greedy, hasta 256 tokens, con parada EOS. Las
categorías son conocidas desde el dataset: aquí no interviene el clasificador.

| Variante | Matemáticas: trigramas repetidos | Biología: trigramas repetidos | Observación |
|---|---:|---:|---|
| Original | 26,4 % | 9,9 % | No se detecta un bloque de hasta ocho palabras repetido consecutivamente tres veces; sí hay errores de contenido |
| 32 sin ajustar, evaluación previa | 36,6 % | 42,1 % | Biología acaba con `that` 69 veces seguidas |
| 32 recuperado | 33,0 % | 50,4 % | Matemáticas repite `we` diez veces; biología repite `energy is the factor that` doce veces, con finales distintos |
| 16 recuperado | 65,7 % | 90,9 % | Matemáticas repite `one hour later` 29 veces; biología repite `which` 231 veces |

Los porcentajes son **repetición léxica, no porcentajes de incoherencia**. En
matemáticas es normal volver a utilizar unidades y expresiones; en código también
se repiten nombres. La comprobación de bloques contiguos y la lectura del texto
permiten distinguir esos casos de los bucles extremos. Los recuentos de palabras
usan minúsculas y omiten puntuación; están definidos en el script reproducible.

![Repetición a igual longitud](../results/generation_recovery_20261007/repetition_prefixes.png)

Las curvas comparan los primeros 32, 64, 128, 192 y 256 tokens cuando existen.
No prolongan artificialmente respuestas que han terminado. El original de
biología finaliza en 205 tokens; el de 32 en matemáticas, en 226. Por eso algunas
curvas terminan antes. Esto evita comparar exclusivamente porcentajes calculados
sobre respuestas de distinta longitud.

La comparación previa sin ajuste no se ha vuelto a generar: se reutiliza
`joint32_generation256_retry/report.json`. Se verificó que **las tres salidas
ajustadas de aquel informe son idénticas token por token a las actuales de 32**.
Se usa para estudiar texto antes/después, nunca para mezclar sus tiempos con
los de los procesos nuevos. No hay aquí un control equivalente de generación
larga de 16 sin ajustar: queda pendiente.

### Corrección y coherencia son problemas distintos

- **Código original:** elimina caracteres numéricos, pero junta la lista en una
  cadena, añade un ejemplo inválido y muestra un resultado inconsistente.
- **Código con 32:** el bloque es breve y tiene sintaxis válida; filtra cadenas
  completamente numéricas en lugar de quitar los dígitos dentro de cada cadena.
  Para `['4words']` dejaría `'4words'`; la referencia espera `['words']`.
- **Código con 16:** usa `isdig()`, que no es un método de `str`, y convierte a
  entero. La sintaxis válida no implica que funcione. No ejecutamos código generado.
- **Matemáticas:** ninguna de estas tres salidas obtiene la respuesta correcta,
  660: antes son 600/40 = 15 horas; después son 11 horas a 60 páginas/hora.
  El original confunde horas y velocidad; 32 acaba con el marcador literal
  `<number>`; 16 comienza con algunos cálculos correctos y después entra en bucle.
- **Biología:** el original identifica energía y mantiene un texto legible, pero
  llamarla «sustancia» y describir el calor como si pasara al siguiente nivel
  trófico son imprecisiones. 32 identifica energía pero enumera variaciones de
  la misma frase. 16 empieza con una respuesta equivocada y colapsa en `which`.

## Por qué puede repetirse un modelo

Un generador no tiene una regla automática que diga «esto ya lo he dicho».
Calcula probabilidades para el siguiente token a partir del texto anterior.
Nuestra decodificación **greedy** elige el de mayor puntuación sin penalización
de repetición. Si una frase repetida sigue teniendo la puntuación ganadora,
volverá a elegirla; al añadirla al contexto puede mantener ese mismo patrón.
EOS solo detiene la generación cuando resulta elegido, no cuando una persona
considera que ya se ha contestado.

La degeneración repetitiva y su dependencia del método de decodificación están
documentadas por [Holtzman et al., The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).
El trabajo de [Welleck et al., Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319)
estudia objetivos que penalizan continuaciones repetidas. Son antecedentes que
explican mecanismos posibles y motivan controles; **no identifican por sí solos
la causa concreta de nuestros resultados**.

## Qué cambia al restringir expertos

Los tres modelos activan **ocho expertos por token y capa**. Cambiamos los
candidatos: 64 en el original, 32 o 16 en el grupo residente. No estamos
comparando ocho expertos activos contra cuatro o dos. Además, los grupos se
eligieron por estadísticas de routing sobre datos categorizados: eso no prueba
que cada grupo sea un módulo autónomo capaz de expresar todo su dominio.

Cuando el router original necesita un experto fuera del grupo, el alumno elige
otro disponible. Cambia la transformación del estado interno y puede cambiar la
siguiente elección en capas posteriores. Que el conocimiento siga almacenado en
el archivo no garantiza que esta ruta restringida lo pueda utilizar correctamente.
Esta es una hipótesis de alteración de la cooperación aprendida, apoyada por la
degradación observada al intervenir en los expertos; no hemos localizado todavía
las capas o los expertos responsables de cada bucle.

Hay también un efecto de escala potencial: OLMoE tiene `norm_topk_prob=false`.
La política `restricted` conserva las probabilidades de la softmax original,
pero selecciona dentro del grupo permitido. Para unos mismos logits, la suma de
sus ocho pesos no puede superar la del top-8 libre. Por tanto, puede cambiar tanto
**qué expertos contribuyen** como **la magnitud de su contribución conjunta**.
No se ha medido cuánto explica cada efecto en estos checkpoints recuperados.

No es una corrección obvia «normalizar todo a uno»: eso cambiaría también la
escala usada en el entrenamiento original. Ya existe `restricted_native_mass`
para estudiar una compensación distinta. En el piloto previo de ocho residentes
su NLL fue 5,91 frente a 5,15 sin reescalar: no solucionó aquel caso.
Véase [ROUTING_STUDY.md](ROUTING_STUDY.md); era otro conjunto de prompts y no se
deben trasladar esos números como resultado de las recuperaciones actuales.

## Por qué puede bajar la pérdida y seguir apareciendo el bucle

El objetivo implementado combina CE de la respuesta de referencia y KL con el
profesor, con peso 0,5 para cada término. **Ambos se calculan sobre los prefijos
de la respuesta de referencia**, incluyendo EOS. Durante generación, en cambio,
el modelo se condiciona a sus propias palabras, también cuando se equivoca.
Mejorar la predicción sobre prefijos correctos no garantiza que pueda salir de un
prefijo erróneo como `which which which`.

Este piloto empleó 24 ejemplos de entrenamiento, seis de validación y seis de
test, con 24 actualizaciones. Adaptó el router y LoRA de la proyección de salida
de expertos; no reentrenó todas las matrices. Es evidencia insuficiente para
declarar que esa adaptación recupera la cooperación necesaria en lenguaje largo.
Tampoco demuestra que simplemente entrenar durante más tiempo vaya a arreglarlo.

La auditoría de las 24 respuestas de entrenamiento no encuentra estos grandes
bloques de palabras repetidas: el máximo detectado es una variable `n` tres veces
en una referencia de código. No parece una copia literal de esos bucles desde
las respuestas del piloto. Esta revisión superficial **no certifica la calidad
de los datos ni la calibración de las probabilidades del profesor**.

El profesor no suministra en este piloto las respuestas libres defectuosas
mostradas arriba como si fueran soluciones correctas: sus distribuciones se
calculan condicionadas a las respuestas de referencia. Aun así, sus probabilidades
pueden ser imperfectas. No debe confundirse destilar distribuciones con copiar
estas generaciones completas.

## Qué podemos descartar y qué no

| Posible explicación | Evidencia y alcance |
|---|---|
| Swapping en mitad de la frase | Cero cargas durante generación en 32 y 16; no explica directamente esos bucles |
| Clasificador que cambia de tema | No se ejecutó: el grupo permanece fijo según la etiqueta conocida |
| EOS ignorado globalmente | La opción de parada está activa; ambos alumnos terminan el código con EOS. En las respuestas truncadas EOS no fue elegido |
| Cuantización adicional | No se aplicó en esta comparación; las tres variantes usan la ruta BF16 documentada |
| Corrupción de la caché/KV | El test pequeño reproduce logits y siguiente paso con KV frente al modelo nativo. No sustituye una comprobación numérica completa sobre el checkpoint público |
| Pocos candidatos o selección inadecuada | Compatible con los resultados; aún falta separar el tamaño del grupo, su composición y las diferencias entre adaptaciones |
| Greedy como único responsable | No comprobado. Se usó igual en todos; puede amplificar el daño, pero falta un contraste de decodificación |
| Finetuning suficiente | No: mejora ciertos indicadores, pero no recupera estas respuestas largas |

## Controles que faltan antes de afirmar una causa en el paper

1. Congelar los mismos pesos y grupos y comparar greedy con muestreo top-p bajo
   varias semillas, conservando el límite de longitud y EOS. Medir corrección
   además de repetición. Menos bucles sin más aciertos sería mitigación, no recuperación.
2. Registrar probabilidades de EOS y de las palabras repetidas, y masa del router
   por capa, antes y durante el bucle. Hoy conocemos lo elegido, no toda esa traza.
3. Completar el control largo de 16 sin ajuste y contrastar grupos categorizados,
   grupos de igual tamaño de control y expertos generales compartidos, sin usar
   las respuestas de test para elegirlos.
4. Comparar la misma adaptación y exposición a datos bajo cada restricción, y
   controlar la política de escala. Los modelos recuperados 32 y 16 actuales
   tienen adaptaciones distintas: su diferencia no aísla solo el número residente.
5. Ampliar datos y semillas de entrenamiento solo después de fijar estos controles;
   seleccionar también con métricas de generación en validación, conservando un
   test nuevo que no se utilice para diseñar la solución.

Los prompts de este informe ya son ejemplos de diagnóstico examinados: para las
decisiones futuras no deben presentarse como un test final intacto.

## Artefactos y reproducción

```bash
python -m asi.analysis.recovery_repetition
```

Produce `repetition_audit.json`, con hashes de los informes de origen y recuentos,
y `repetition_prefixes.png`. No carga el modelo ni modifica la decodificación.
Los textos completos y tiempos siguen en `summary.md` y `outputs.html` de la
comparación. El protocolo de inferencia está en [RECOVERY_GENERATION.md](RECOVERY_GENERATION.md).

Validación del código: 26 tests de recuperación y 13 del backend público superados.
Incluyen identidades, comparación de prompts, recuentos de bucles y equivalencia
del backend/caché/KV en un modelo pequeño. Esta verificación no equivale a probar
las hipótesis de la tabla ni a un benchmark de calidad general.


## Control de decodificación preparado (16/10/2026; no ejecutado)

Plan `configs/decoding_control_dev_v1.json`, ligado por SHA256 al plan de nueve
preguntas de desarrollo del control de escala. El runner existente `decode_study`
estudia caché del modelo local con greedy; no implementa este control OLMoE.
Se ampliará el decoder compartido, conservando greedy como comportamiento por
defecto y su equivalencia, en lugar de duplicar un bucle de generación.

Dos brazos congelados (original y restricción nativa de 32) por nueve preguntas.
Greedy una vez; muestreo temperatura 0.7/top-p 0.9 con tres semillas prefijadas:
72 respuestas. Mismos pesos, máscaras, tokens, EOS y límite 256; sin penalización
de repetición ni ajuste. El muestreo debe usar generador independiente por respuesta
para reanudar sin cambiar los tokens. No se escoge una semilla ganadora.

Se observarán solución central, seguimiento de instrucciones y repeticiones por
separado. Menos repetición no implica recuperación. No atribuir cambios de CE a
muestreo ni presentar estas preguntas ya inspeccionadas como test independiente.
Primero faltan implementación, pruebas CPU y verificación del plan; no hay una
nueva ejecución GPU iniciada ni resultados que añadir al paper.

16/10, implementación parcial: el decoder compartido ya admite temperatura, top-p
y semilla explícita por respuesta. Greedy mantiene argmax y sigue siendo el valor
por defecto. Cuatro tests CPU cubren EOS, empate greedy, token que cruza el umbral
nucleus, exclusión de cola y reproducción independiente del RNG global. Falta
el runner del protocolo y validación real en GPU; no hay resultados nuevos.

17/10: ejecutor integrado en `routing-scale`, sin otro bucle de generación.
Lanzado `python -m asi routing-scale --plan configs/decoding_control_dev_v1.json
--output results/decoding_control_dev_20261017/run`. Reanudación con `--resume`.
Valida protocolo, hashes y cobertura por brazo/política/semilla/pregunta; guarda
cada respuesta. Uso GPU concurrente del escritorio observado: tiempos descriptivos,
no benchmark. Plan conserva su estado histórico previo; el progreso real está en
`results/decoding_control_dev_20261017/run/report.json` y logs superiores.


### Reconstrucción del informe de decodificación

Al completar la tanda (no durante una cobertura parcial):

```bash
python -m asi mask-intervention-report results/decoding_control_dev_20261017/run/report.json
```

El analizador detecta `decoding_plan` y genera `analysis/summary.json` e
`analysis/index.html` con todas las salidas escapadas como texto. Rechaza cobertura
incompleta, duplicados, preguntas desparejadas, valores no finitos y transferencias
en generación restringida. Resume las semillas dentro de cada pregunta y después
las preguntas; no interpreta EOS o repetición como corrección. El campo
`quality_review_pending` permanece verdadero hasta realizar una revisión semántica
separada. No usar la tabla de pérdidas de este control como mejora por decodificación:
la respuesta de referencia usada en teacher forcing es la misma.


### Criterios de revisión del control de decodificación

Fijados durante la ejecución, antes de leer las nuevas salidas completas. Son
criterios de revisión exploratoria, no una métrica validada ni revisión ciega:
el evaluador conoce el diseño y los ejemplos ya se han usado en desarrollo.

1. **Respuesta central:** correcta / parcialmente correcta / incorrecta / ausente /
   indeterminada. Justificar con el fragmento concreto; una palabra esperada en
   una enumeración no basta. En matemáticas distinguir resultado de razonamiento.
2. **Instrucciones:** cumple / no cumple / ambiguo. Valorar explícitamente el
   formato solicitado, no confundir sintaxis de Python con corrección funcional.
3. **Coherencia:** completa / error localizado / contradicción o bucle dominante /
   truncada. Registrar el motivo, sin resolver silenciosamente ambigüedades.
4. **Terminación:** EOS o límite, longitud generada y repetición automática.
   Una respuesta vacía o demasiado corta no mejora por tener menos trigramas repetidos.
5. **Código:** lectura estática solamente en esta máquina. Estado funcional
   **no ejecutado**, hasta disponer del entorno aislado autorizado; no convertir
   la inspección en una tasa de tests aprobados.

Revisar todas las semillas, incluidas las malas. Comparar por pregunta y resumir
las semillas dentro de ella; no escoger el mejor output. Informar desacuerdos
entre corrección y repetición. La confirmación independiente y adjudicación
humana siguen pendientes; no usaremos este diagnóstico para elegir un checkpoint.
