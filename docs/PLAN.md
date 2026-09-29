# Mapa del TFG y protocolo

La numeración **1.1–3.3** corresponde a cada elemento de las listas de los pasos
1, 2 y 3 de [TFG_original.txt](TFG_original.txt). El documento propone hipótesis;
el código y los pilotos permiten comprobar solo una parte de ellas.

La [revisión del documento](TFG_actualizado_2026-09-27.txt) añade los pasos **2.4**
(programa dinámico) y **3.4** (modelos independientes). Su implementación parcial
está registrada en `plan.json`; el [diseño de 124M](TRAINING_124M.md) explica las
recetas disponibles y las mediciones aún pendientes.

El registro completo está en [plan.json](plan.json). Consultarlo con
`python -m asi plan` o `python -m asi plan 1.4` no carga modelos.

## Estado y archivos por paso

La fuente única del estado, archivos asociados y tareas pendientes es
[plan.json](plan.json). Evitamos mantener aquí otra tabla manual que pueda
quedarse desactualizada. Consultar todo el mapa o un paso concreto:

```bash
python -m asi plan
python -m asi plan 2.4
```

Las guías del [índice principal](../README.md) explican los comandos y límites
de cada componente. «Implementado» significa que existe el mecanismo; no que
se haya demostrado la hipótesis con modelos entrenados. El piloto público
[OLMoE/AirLLM](PUBLIC_MOE.md) distingue pruebas pequeñas de mediciones de 7B.

## Conceptos e interpretación

Un experto se identifica por **(capa, ID)**. El experto 2 de una capa y el de otra
son pesos distintos. Un experto calibrado puede asociarse a varias etiquetas.

**E** son los pesos del router. La selección depende también del estado oculto h,
activación, sesgos y reglas top-K. Comparar vectores E aislados no basta para
asignar una profesión: se observan E·h y selecciones con entradas etiquetadas,
contrastando con popularidad global.

Los expertos no se llaman entre sí como funciones independientes: cada capa
selecciona expertos y transmite su representación a capas posteriores. Una
arista de una traza es una asociación, no una prueba causal de comunicación
indebida. Para estudiar dependencias hacen falta ablaciones y medir sus efectos.

En PyTorch, `device=cpu` significa almacenamiento en **RAM**. Estas cachés
transfieren los pesos a GPU para ejecutar sus FFN. RAM fijada puede facilitar
transferencias; el beneficio debe medirse. Backbone, expertos compartidos y
buffers también ocupan memoria. El modo de shards añade carga disco/RAM/GPU;
su límite RAM corresponde a tensores de expertos, no al RSS total ni a la caché
de páginas del sistema operativo. El registro de bytes leídos es lógico y no
demuestra cuántos bytes alcanzaron físicamente el SSD.

El criterio principal es **cuántos pasos de decode no necesitan cargar expertos**,
acompañado por bytes transferidos y capacidad GPU/RAM. Un 90% de aciertos por
experto puede seguir implicando cargas en todos los tokens si falla alguna capa.
La latencia es un coste que se debe documentar, no una promesa de superar al modelo
residente. Dos expertos significa dos **enrutados** por capa; los compartidos,
backbone y KV se contabilizan aparte.

En el experimento **N sobre N**, una sola clase determina N expertos por capa y
todos se ejecutan. La clase no cambia dentro de la respuesta. No se hace unión
multietiqueta, porque aumentaría N; distintas clases pueden compartir expertos.
El modo uniforme asigna `route_scale/N` a cada experto sin calcular E. Un control
calibrado asigna a cada uno `masa_media_train/N`, también constante, para estudiar
el efecto de escala: el softmax original puede dar a sus N seleccionados una suma
menor que `route_scale`. Ambos cambian la función del modelo y exigen evaluación
de calidad; no son variantes de caché exacta.

## Orden y criterios del estudio

1. **Controles y corpus (1.1, 2.1).** Versionar manifiestos, tokenizer, tokens,
   hashes y arquitectura. Reservar calibración, desarrollo y test; revisar
   duplicados. Ventanas no solapadas no garantizan independencia del entrenamiento.
2. **Categorizar el checkpoint (2.1).** Equilibrar tokens por dominio, comparar
   rankings con popularidad global y repetir muestreo. Revisar estabilidad de
   etiquetas y señalar explícitamente dominios sin validación.
3. **Equivalencia y coste (2.2, 1.3).** Comparar residencia completa y prefetch
   con mismas entradas y dtype; verificar rutas, NLL y logits con tolerancia
   declarada. Probar restricción como una modificación de calidad distinta.
4. **Predicción y sesiones (1.4, 1.5).** Secuencias estables, transiciones habituales
   y cambios bruscos. Comparar LRU, popularidad, clasificación y predictor aprendido
   con igual capacidad. Una política que conozca el futuro es solo una cota ideal.
5. **Entrenamiento controlado (1.1, 1.2, 2.3).** Igual arquitectura base, datos,
   tokens, tokenizer, optimizador y semillas emparejadas. Informar parámetros
   totales/activos y cómputo. Variar tamaño de pool por separado y declarar cambios
   de presupuesto; cambiar el router de un checkpoint no sustituye ese control.
6. **Compresión y escalado (2.3, 3.3; luego 3.1–3.2).** Comparar compresión con
   coste RAM/GPU antes de escalar. Verificar disponibilidad y arquitectura del
   destino denominado V4.1 flash en el documento: no es una dependencia actual.

Fijar antes de cada ensayo los márgenes aceptables de calidad y latencia según
el uso previsto. Un miss no invalida el sistema por sí solo: importa su frecuencia
y coste, y el objetivo conjunto de memoria, calidad y latencia.

## Cómo publicar un resultado comprensible

Conservar comando, configuración, semillas, versiones, hardware, hashes, entradas
exactas, exclusiones, salidas y resumen humano. Usar una carpeta nueva por ensayo.
Separar etiquetas conocidas del corpus de predicciones del clasificador.

| Métrica | Interpretación y cautela |
|---|---|
| NLL y delta | Menor es mejor sobre esas entradas; requiere igual tokenización/protocolo. No equivale a calidad de tareas. |
| Cobertura top-K | Fracción de selecciones nativas dentro de candidatos; no mide precisión de respuestas. |
| Preferencia por dominio | Contrastar con popularidad global, tokens observados y variabilidad entre muestras. |
| Hit/miss | En caché nativa cuenta expertos únicos por capa/forward, no tokens. Acompañar de precargas y bytes. |
| Tiempo | Separar clasificación, precarga, prefill/decode cuando sea posible. Calentar, sincronizar CUDA, repetir, alternar orden; publicar p50/p95 y dispersión. |
| Memoria | Separar expertos, backbone/shared, KV/buffers, clasificador, temporales, allocator, RAM total y disco. No sumar reservada y asignada como cantidades disjuntas. |
| Calidad de tarea | Evaluación externa por dominio y entradas mixtas, criterios previos y ejemplos de fallos. |

Cada conclusión debe presentar pregunta, control, intervención, tamaño de muestra,
resultado, limitaciones y decisión siguiente. Los tests verifican mecanismos;
las hipótesis requieren experimentos. Véase [FINDINGS.md](FINDINGS.md).

La opción 1 tiene una [batería preparada de tres modelos](COMPARISON.md), vinculada
a 1.2, 1.3, 1.5, 2.2 y 2.3. El bundle actual congela entradas y trabajos; sus
evaluaciones quedan pendientes hasta disponer de los tres modelos entrenados.
La guía detalla la correspondencia de cada módulo y evita interpretar esta
preparación como un resultado experimental.
