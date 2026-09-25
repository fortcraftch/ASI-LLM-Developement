# El proyecto explicado sin dar por sabidos los nombres

Queremos que un modelo grande funcione en una GPU pequeña. Para conseguirlo,
intentamos mantener en la GPU solo los expertos que necesita una conversación y
dejar el resto en la RAM del ordenador. La dificultad es hacerlo sin estar
moviendo expertos continuamente y sin perder demasiada calidad.

Las [gráficas de las pruebas realizadas](../results/project_overview_v1/index.html)
se generan leyendo informes guardados: no vuelven a ejecutar el modelo.

## Qué es un experto y qué estamos moviendo

Un experto es una parte de la red neuronal, con sus propios pesos. No es un
chatbot independiente ni una persona que sabe exclusivamente de una profesión.
El nombre «experto en programación» es una etiqueta que asignamos o una
especialización que intentamos conseguir; hay que medir si realmente se cumple.

Nuestro modelo de prueba tiene 12 capas: una sin este sistema y once con ocho
expertos enrutados cada una. En cada una de esas once capas elige dos expertos
para procesar cada fragmento de texto. Eso son **88 instancias de expertos** en
total: el experto 2 de una capa no es el experto 2 de otra.

**Usar dos no significa almacenar solo dos.** El modelo normal puede tener los
ocho cargados y utilizar una pareja distinta con cada token. Nosotros hemos
probado a dejar solo dos cargados por capa: 22 instancias repartidas por el modelo.
Además siguen existiendo partes comunes, expertos compartidos y memoria para el
contexto. Por eso ahorrar el 75 % de los pesos de expertos no ahorra el 75 % de
toda la memoria de la GPU.

GPU y CPU son procesadores. VRAM y RAM son memorias. Los expertos se almacenan
en RAM cuando no están en GPU; no «dentro de la CPU». En nuestras pruebas de
descarga, el cálculo del modelo sigue en GPU y los pesos se copian desde RAM
cuando hace falta. Tenerlos en RAM no es, por sí mismo, más rápido que en VRAM.

## Traducción de los nombres que hemos utilizado

| Nombre del código | Qué quiere decir |
|---|---|
| `original` | El modelo con su forma original de elegir expertos. Sirve como referencia. |
| `classified` | El modelo ya entrenado al que después hemos asignado etiquetas temáticas a sus expertos, observando su uso. Clasificarlo no modifica sus pesos por sí mismo. |
| `domain` | El modelo que entrenamos asignando desde el principio grupos de expertos a temas, por ejemplo programación o matemáticas. |
| `oracle` | Una prueba en la que damos al sistema la etiqueta conocida del texto. Si sabemos que el texto pertenece al conjunto de matemáticas, le decimos «matemáticas». No conoce la respuesta ni elige expertos perfectos. |
| `predicted` | El sistema tiene que adivinar el tema mediante un clasificador. |
| `global` | Usamos siempre los expertos más frecuentes, sin distinguir temas. Es un control para comprobar si las etiquetas aportan algo. |
| `resident` o `native` | En estos informes, la referencia con todos los expertos disponibles en GPU y elección original. |
| `fixed` | La pareja de expertos queda fijada durante la pregunta y su respuesta. |
| `restrict` | El modelo aún decide, pero solo puede escoger entre los candidatos permitidos. |
| `prefetch` | Copiar expertos a GPU antes de que se pidan, intentando anticiparse. |
| `pool` | Grupo de expertos o de datos asociado a un tema. No es una GPU distinta. |

Hay dos decisiones diferentes. El **clasificador externo** intenta reconocer el
tema de la pregunta. El **router interno**, asociado a los vectores E, decide qué
expertos se utilizan en cada capa y token. En la prueba de expertos fijos
sustituimos esa decisión interna por una pareja constante durante el turno.

Oracle y predicted son maneras de elegir la etiqueta durante una prueba;
classified y domain describen cómo se han organizado los expertos. No son
cuatro modelos distintos. Una prueba puede ser, por ejemplo, «modelo classified
con etiqueta oracle y dos expertos fixed».

## Qué hemos construido y comprobado

1. **Organizamos los datos por temas.** Se prepararon categorías y ocho grupos,
   con un manifiesto que indica qué archivos pertenecen a cada grupo. Así podemos
   entrenar y evaluar por tema. El corpus actual no tiene validación de IA:
   no hemos inventado resultados ni usado train para cubrir ese hueco.
2. **Añadimos entrenamiento por grupos.** Los datos de un tema se encaminan a los
   expertos permitidos para ese tema. Se corrigió la lectura de shards pequeños
   que producía batches incompletos. Los tests verifican que los expertos
   inactivos no se actualizan; las partes comunes del modelo sí pueden aprender
   de todos los temas. El checkpoint de diez pasos solo demostró funcionamiento.
3. **Hicimos visibles las decisiones.** Ahora podemos registrar capa, experto,
   rutas permitidas, uso de memoria y movimientos de pesos. No observamos
   violaciones de máscara en la auditoría guardada. Eso prueba que se respetaron
   las restricciones, no que el conocimiento de cada experto sea exclusivo.
4. **Analizamos el modelo ya entrenado.** El checkpoint 4750 tiene ocho expertos
   por capa y elige dos. Observamos qué expertos activa con textos de cada tema
   y construimos una clasificación posterior. Son asociaciones medidas, no una
   demostración de que un experto «solo sabe matemáticas». Es nuestro modelo
   pequeño de prueba, no el DeepSeek-V3 oficial completo.
5. **Construimos la caché RAM/GPU.** Permite limitar cuántos expertos permanecen
   en GPU, copiar los necesarios y retirar otros. Probamos cargar a demanda,
   anticipar los más frecuentes, anticipar según el tema y usar un predictor de
   uso. El predictor todavía no ha demostrado ventaja sobre la popularidad global.
6. **Probamos dos plazas por capa manteniendo E.** Las respuestas y rutas
   comprobadas coincidieron con el modelo completo, pero las parejas cambiaban
   tanto que había cargas en cada paso de generación. Ahorrábamos memoria, pero
   todavía no conseguíamos evitar el swapping continuo.
7. **Probamos dos expertos fijos por tema, sin E.** La pareja se prepara al
   empezar un turno y se mantiene durante su lectura y respuesta. En las pruebas
   registradas no hubo cargas internas de expertos después de esa preparación.
   A cambio, empeoró la predicción de texto. Cambiar de tema puede preparar otra
   pareja; algunas clases comparten parejas y no siempre hay que sustituirlas.
8. **Añadimos carga desde disco con RAM limitada.** Exportamos pesos separados y
   verificamos un chat fijo con unos 4 MiB de caché RAM de expertos. Ese límite no
   incluye toda la memoria del programa ni la caché del sistema operativo. Las
   lecturas registradas son lógicas, no una medición de actividad física del SSD.
9. **Ordenamos el proyecto y lo vinculamos al TFG.** Hay módulos para datos,
   modelos, ejecución, análisis y experimentos, con una entrada común
   `python -m asi`. El mapa de archivos y pasos está en PLAN y COMPARISON.
10. **Preparamos la comparación extensa.** Hay 1.258 ventanas congeladas, 40
    conversaciones y 742 trabajos previstos antes de exclusiones. Comparará las
    tres alternativas con condiciones explícitas. **No la hemos ejecutado**:
    esperamos a que estén entrenados los tres modelos.

También existe una exportación INT8, pero exportar pesos comprimidos no demuestra
que la inferencia ya los ejecute comprimidos en GPU. Ese desarrollo sigue pendiente.

## Qué nos dicen los números actuales

Estos tres resultados proceden de ensayos diferentes. Son conclusiones de cada
ensayo, no una tabla para comparar sus tiempos o NLL directamente.

| Prueba | Qué aprendimos |
|---|---|
| Caché exacta | Podemos mover pesos y conservar las decisiones y predicciones comprobadas del modelo original. Las comprobaciones no cubren necesariamente todos los valores internos. |
| Dos plazas con E | Se reducen los pesos de expertos en GPU de 115,5 a 28,875 MiB. Durante decode hubo cargas en todos los pasos: unos 15,42 MiB transferidos por paso en el piloto. |
| Dos expertos fijos, tema predicho | No hubo cargas de expertos dentro de la respuesta después de preparar la clase. En 48.256 tokens evaluados, el acierto del siguiente token pasó de 32,66 % a 25,22 %; la perplexity se multiplicó por aproximadamente 2,11. |

En la última prueba, dar el tema conocido obtuvo un resultado muy parecido a
predecirlo. Además, las etiquetas temáticas mejoraron poco frente a usar siempre
una pareja global. Esto apunta a que **reconocer el tema no basta para recuperar
lo que perdemos al fijar los expertos** en este checkpoint. No demuestra que esa
idea vaya a fallar con cualquier arquitectura o entrenamiento.

No afirmamos que el modelo haya pasado de acertar el 32 % de las preguntas al
25 %. Medimos fragmentos de texto predichos exactamente. Una respuesta puede ser
válida aunque utilice otras palabras, y una predicción frecuente puede no resolver
bien una pregunta. Faltan tareas externas con respuestas verificables para medir
corrección de programación, matemáticas y otras áreas.

## Cómo leer las gráficas sin confundir medidas

**Expertos en GPU** cuenta instancias cargadas. **Cambio de pareja** cuenta cuándo
el modelo escoge otra pareja en una capa. **Carga** cuenta un experto que hubo que
traer. **Evicción** cuenta uno que salió para dejar sitio. Son cosas diferentes:
si todos están cargados, puede cambiar la pareja sin mover ningún peso.

**Correctitud de ejecución** significa que el programa respeta las restricciones
o reproduce el control cuando debería hacerlo. **Calidad de respuesta** significa
que resuelve bien la tarea. Los gráficos de coincidencia con el original miden
lo primero; el acierto de tokens y la NLL aproximan una parte de lo segundo.

**Tiempo de lectura** es lo que tarda en procesar la pregunta. **Tiempo de decode**
es lo que tarda en calcular nuevos tokens. Clasificar el tema y preparar los
expertos también tarda. Los gráficos indican qué partes incluyen. La velocidad
de decode excluyendo esas etapas no es la velocidad completa que ve el usuario.

**MiB de expertos** no es memoria total. El modelo conserva partes comunes y
memoria del contexto. PyTorch también reserva espacio para reutilizarlo. Las
instantáneas antiguas no permiten reconstruir la RAM total del proceso ni su
pico continuo: cuando no hay medición se muestra «No medido», no cero.

Las repeticiones de una misma conversación sirven para observar tiempos; no se
convierten en conversaciones nuevas. Las auditorías de diez pasos y los modelos
aleatorios se muestran como pruebas de funcionamiento, separados del checkpoint
entrenado 4750. El log de entrenamiento disponible solo contiene unos pocos pasos:
no se dibuja una curva inventada de los 4.750 pasos.

## Cómo volver a generar el panel

Con el entorno activado y desde la raíz:

```powershell
python -m asi graphs --output results/project_overview_v2
```

El nuevo archivo Python es `asi/analysis/plots.py`. Lee los informes y trazas,
genera PNG y SVG para el TFG, guarda los valores y hashes, y crea `index.html`
para recorrerlos con explicaciones. Se elige una carpeta nueva para conservar
cada versión del panel. No carga archivos `.pt`, no descarga modelos ni inicia
las pruebas pendientes.

La prioridad siguiente sigue siendo obtener los tres modelos terminados,
comprobar que su entrenamiento permita una comparación justa y ejecutar los
controles de la batería preparada antes de interpretar sus intervenciones.
