# Paso 9: piloto de capacidad y granularidad

Seguimiento: [routing con tres semillas y más actualizaciones](CAPACITY_FOLLOWUP.md).

Esta fase verifica que podemos comparar tamaños de experto de forma controlada.
No intenta encontrar en unas pocas actualizaciones un umbral para comprender
lenguaje. Los checkpoints de homelab y sus evaluaciones quedan fuera del piloto.

## Qué ejecutamos

Diez variantes inicializadas desde cero, obtenidas del diseño ya preparado:

| Eje | Ancho de experto | Expertos totales | Activos por token |
|---|---:|---:|---:|
| Capacidad pequeña | 60 | 16 | 2 |
| Capacidad actual | 120 | 16 | 2 |
| Capacidad mayor | 240 | 16 | 2 |
| Granularidad gruesa | 240 | 8 | 1 |
| Granularidad fina | 60 | 32 | 4 |

Cada fila tiene routing libre aprendido y routing uniforme restringido por
categoría. La configuración actual también es el punto central de granularidad;
se reutiliza, no se entrena dos veces. El ancho compartido sigue siendo 240.
Los totales van de 106,1M a 154,8M: la variante grande supera explícitamente el
presupuesto del experimento principal, sin modificar ese experimento.

El piloto usa 32 actualizaciones por variante, batch 2, contexto 128 y acumulación
4: **32.768 tokens por variante**. En cada grupo de ocho pasos aparecen las ocho
categorías una vez, en orden aleatorio reproducible. Cada pool aporta 4.096 tokens.
Se comparan pérdidas de validación al principio, a mitad y al final, sobre las
mismas 4.096 posiciones objetivo reservadas (512 por pool).

## Controles implementados

- Todas las variantes reciben exactamente las mismas ventanas y su orden. Las
  ventanas se guardan en `windows.npz` con SHA-256; no se vuelven a muestrear al
  reanudar el estudio.
- Mismos hiperparámetros: AdamW, LR máximo 1e-4, warmup de cuatro pasos, betas
  0,9/0,95, weight decay 0,1 y clipping de norma a 1. Después del warmup el LR es
  constante. El presupuesto se iguala en tokens, no en cómputo.
- Inicialización por nombre de parámetro y semilla: los parámetros comunes de
  igual forma son idénticos entre variantes, independientemente del orden de
  construcción del modelo. Se registra su hash. Esto es un control específico del
  piloto; no asumir que lo cumplen entrenamientos históricos de otros scripts.
- El modelo restringido activa el pool correspondiente al dato. El libre ve todos
  los expertos y entrena su router. No se usa un router congelado sin entrenar
  como sustituto del control libre. No se añade una pérdida auxiliar de balanceo.
- Cada experto registra asignaciones de tokens y pasos con gradientes no nulos.
  La validación no incrementa esos contadores. El informe resume expertos sin uso
  y concentración del reparto; recibir gradientes no demuestra especialización.
- Se mide tiempo de entrenamiento y pico CUDA. El tiempo incluye la instrumentación
  y transferencia de los datos; excluye inicialización y validación. No extrapolar
  directamente estas velocidades a un entrenamiento de producción sin telemetría.

Los datos locales tienen las ocho categorías, pero **su manifiesto es diferente
del dataset completo de homelab**. No se ha desactivado la verificación del entrenador
principal ni se ha hecho pasar este corpus por el completo. El piloto guarda su
identidad, los shards utilizados y las ventanas exactas. Train y val usan sus splits
respectivos; no se usan los benchmarks para actualizar pesos o decidir variantes.

## Ejecución y resultados

```bash
python -u -m asi capacity-pilot --output results/capacity_pilot_nuevo
```

Para continuar tras una interrupción, repetir el mismo comando y ajustes con
`--resume`. Omite variantes completadas y reinicia desde cero la variante
interrumpida con las mismas ventanas e inicialización. No reanuda un optimizer
a mitad de una variante. El piloto no conserva pesos: conserva las mediciones.

Para reconstruir únicamente la tabla y la gráfica:

```bash
python -m asi capacity-pilot --output results/capacity_pilot_20261005 --report-only
```

Consultar `results/capacity_pilot_20261005/summary.md`, `capacity_pilot.png` y los
JSON de cada variante. El plan distingue `running` de `completed`; una variante
parcial no aparece como resultado final. Los fallos no se convierten en resultados.

## Qué sigue después

1. Comprobar cobertura de expertos, estabilidad y que el coste cabe en el hardware.
   Un fallo de implementación debe resolverse antes de comparar calidad.
2. Cuando termine DeepSeek, verificar su configuración, datos y exposición reales
   antes de aceptarlo como control del categorizado. Si no coinciden, la comparación
   será descriptiva y hará falta un control adicional.
3. Para una comparación científica nueva, fijar previamente el corpus y su split,
   contexto, presupuesto de tokens y checkpoints de evaluación; entrenar ambas
   políticas con tres semillas. Las recetas grandes están en
   `configs/capacity_study_v1/recipes/`, pero no se lanzan desde este piloto.
4. Comparar NLL global y por categoría, benchmarks completos con los mismos ajustes
   y respuestas revisadas a ciegas, antes y después del mismo SFT. Solo después
   estudiar restricciones de residencia para separar capacidad de degradación.
5. Definir un criterio de utilidad específico de la aplicación antes de elegir la
   configuración mínima. No declarar un mínimo universal de parámetros, ni elegir
   un ganador a partir de estas 32 actualizaciones y una sola semilla.

Todavía no hemos comprobado comprensión, fluidez ni calidad final de estas diez
arquitecturas. Este piloto comprueba el experimento y expone desequilibrios tempranos.

## Resultado del piloto local del 05/10/2026

Las diez variantes finalizaron. Se verificaron hashes idénticos del backbone y
parámetros compartidos iniciales, y que el total de asignaciones de cada capa
coincide exactamente con tokens procesados multiplicados por expertos activos.
El tiempo acumulado de entrenamiento instrumentado fue 331,4 segundos, además
del tiempo de inicialización, validación y elaboración de informes.

- Todas redujeron la NLL, aproximadamente de 10,92 a 8,88–8,97. NLL es la pérdida
  al predecir el siguiente token, no una puntuación de inteligencia. No aparece
  una relación monotónica clara entre tamaño y pérdida con tan pocos pasos.
- Ningún experto quedó completamente sin tokens. Sin embargo, las variantes libres
  con 16 expertos tuvieron entre 5,14 y 5,28 expertos efectivos por capa de media:
  muchos tokens fueron a una minoría. «Efectivos» resume la concentración mediante
  entropía; no significa que solo se ejecutaran cinco ni que los demás estuvieran
  apagados. En las variantes restringidas el reparto fue uniforme por construcción.
- El pico CUDA asignado estuvo aproximadamente entre 1,95 y 3,64 GiB. Corresponde
  a este contexto y batch pequeños y no es una estimación de los requisitos del
  entrenamiento completo en homelab.
- 133 pruebas automatizadas superadas. El informe y la gráfica se generaron con
  las diez variantes completadas.

El siguiente diagnóstico del checkpoint libre terminado debe incluir el reparto
de tokens por experto. No podemos atribuir todavía su posible degradación al
tamaño del experto sin comprobar ese reparto y la exposición de entrenamiento.
