# Capacidad y tamaño de expertos: qué podemos comprobar

Preparado el 05/10/2026. **Diseño de experimentos, no resultados de entrenamiento.**

Actualización: ya existe un [piloto controlado ejecutable](CAPACITY_PILOT.md) con
datos locales, ventanas fijas, inicialización pareada y telemetría de expertos.
Sus resultados cortos se separan del barrido completo de preentrenamientos descrito
aquí; no sirven para establecer un umbral de comprensión.

## No existe un umbral demostrado por experto en este proyecto

Un experto FFN transforma representaciones internas; no dispone por sí solo de
tokenizador, atención, contexto ni cabeza de vocabulario. Probarlo aislado puede
medir aproximación de una función o imitación de activaciones, pero no permite
decidir cuándo un modelo comprende lenguaje o deja de cometer errores.

Las leyes de escalado estudian conjuntamente capacidad, cómputo, datos y routing:
[Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169).
La granularidad también importa: repartir capacidad entre expertos menores no es
equivalente a reducir toda la capacidad activa, según
[Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871).
Estos trabajos no proporcionan un umbral universal transferible a nuestro modelo.

En el checkpoint actual, cada experto SwiGLU tiene tres matrices sin sesgos:
`3 × 512 × 120 = 184320` parámetros. Dos seleccionados aportan 368640 y los
compartidos otros 368640 por capa MoE: 737280 parámetros FFN activos. A ello se
añaden atención, embeddings, salida y la primera capa densa. No se puede comparar
184320 directamente con todos los parámetros de GPT-2.

La menor fluidez también puede relacionarse con datos y exposición por pool,
optimización, desbalance, contexto, entrenamiento de instrucciones y selección
de expertos. La comparación actual con GPT-2 cambia varias de esas variables.
Que el modelo falle no demuestra que el tamaño del experto sea la causa; que
mejore al ampliarlo tampoco exculpa por sí solo la categorización.

## Diseño preparado sin cargar pesos

```bash
python -m asi capacity-study --output results/capacity_study_otro
```

`asi/experiments/capacity_study.py` cuenta parámetros sobre el dispositivo `meta`
de PyTorch, sin reservar los pesos reales ni entrenar. El diseño ya generado está
en `results/capacity_study_20261005/plan.md` y `plan.json`. Las configuraciones
incluidas en ese directorio antiguo son propuestas. Ahora `configs/capacity_study_v1/recipes/`
contiene diez recetas ejecutables validadas, su manifiesto congelado y `commands.sh`
con los 30 comandos comentados (tres semillas). No se ha lanzado el barrido.
Se generan con `python -m asi capacity-study --freeze --output DIRECTORIO_NUEVO`.

Dos ejes:

1. **Capacidad:** ancho de experto 60, 120 y 240 con dos activos, 16 expertos
   totales y ancho compartido constante. El backbone queda igual. Esto cambia
   parámetros totales y cómputo: no presentarlo como comparación a igual coste.
2. **Granularidad:** ancho/activos/total = 240/1/8, 120/2/16 y 60/4/32. Se conserva
   la suma de anchos activos y la capacidad compartida. Los parámetros del router
   y los costes de ejecución no son idénticos: medirlos por separado.

Cada configuración tiene un control de routing aprendido libre y una versión
restringida por las ocho categorías. El aprendido debe entrenarse realmente con
todos los expertos disponibles; abrir el router sin entrenar del checkpoint
uniforme no constituye ese control. Tres semillas: 1337, 2027 y 31415. Reutilizar
las configuraciones comunes a ambos ejes: son diez arquitecturas/políticas únicas,
30 ejecuciones con tres semillas. No se han lanzado estos entrenamientos.

Los manifiestos, las recetas y sus presupuestos están congelados. Antes de ejecutar
hay que fijar los criterios de selección y comprobar los datos disponibles.
Algunas variantes alcanzan 154,8M; el límite del experimento original no se modifica. Empezaría por la
pareja aprendida/restringida del tamaño actual, ya planificada, antes del barrido.

## Protocolo para atribuir los cambios

- Mismo tokenizador, backbone, particiones de datos, orden por semilla, contexto
  y exposición en tokens. Medir tokens y gradientes recibidos por cada experto.
- Validación por categoría, pérdida global, balance de routing y posibles expertos
  sin uso. La pérdida debe medirse sin fuga de datos o respuestas al clasificador.
- Comparación principal a igual número de tokens; comparación adicional a igual
  cómputo, sin confundirlas. Registrar FLOPs estimados, tiempo y memoria reales.
- Evaluar antes y después del mismo SFT, con igual corpus y presupuesto, reservado
  por conversaciones. No comparar un GPT-2 ajustado contra un MoE solo preentrenado.
- Primero evaluar cada modelo con su routing nativo entrenado; después imponer
  restricciones de residencia. Así se separa capacidad inicial de degradación.
- Usar los mismos benchmarks reservados y respuestas completas revisadas a ciegas;
  informar incertidumbre y variación entre semillas. No seleccionar con test.

Un ensayo aislado de FFN con entradas sintéticas serviría como diagnóstico de
implementación o capacidad de aproximación. **No lo usaríamos para declarar un
mínimo de parámetros para hablar.** Para medir lenguaje hacen falta modelos
completos entrenados, aunque sean pequeños y no requieran descargar otro LLM.

El resultado defendible sería «bajo este corpus, presupuesto y arquitectura, la
configuración mínima probada que alcanza un criterio predefinido es X». No sería
«todo MoE por debajo de X es inválido». El criterio de utilidad se debe fijar antes
de ver el test y no puede exigir ausencia universal de errores.
