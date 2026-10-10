# Generación y comparación de los modelos locales de unos 124M

## Conversar con DeepSeek libre

El evaluador ya existía (`deepseek-eval`). Ahora `chat deepseek` carga la misma
arquitectura original y el checkpoint de forma estricta, sin convertir sus pesos
al modelo categorizado. No necesita un manifiesto ni un clasificador.

```bash
python -m asi chat deepseek \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_07874.pt \
  --temperature 0.7 --top-k 40 --max-new-tokens 128
```

`exit` termina. Cada turno es independiente. El checkpoint parcial es un modelo
de continuación: el formato predeterminado es `completion`. Para un checkpoint
ajustado con OASST usar `--chat-format oasst`. Para guardar una pregunta:

```bash
python -m asi chat deepseek \
  --checkpoint ../Creating-DeepSeek-V3-From-0/model_07874.pt \
  --prompt "The derivative of x squared is" --temperature 0 \
  --output results/deepseek_generation.json
```

El JSON conserva el texto, tokens exactos, parada, tiempos e identidad del modelo.
No se sobrescriben resultados. Esta interfaz recalcula el prefijo completo;
no representa la máxima velocidad posible con una implementación KV optimizada.

## Comparación de memoria y velocidad

```bash
python -u -m asi local-generation \
  --deepseek-checkpoint ../Creating-DeepSeek-V3-From-0/model_07874.pt \
  --domain-checkpoint model_37595.pt \
  --gpt2-checkpoint ../Creating-GPT-2-From-0/models/gpt2Base.pt \
  --new-tokens 64 --repeats 3 \
  --output results/local_124m_generation_comparison
```

Se ejecutan secuencialmente cinco variantes para evitar tener varios modelos
ocupando la GPU a la vez:

| Variante | Expertos en GPU por capa MoE | Activados | Qué comprueba |
|---|---:|---:|---|
| DeepSeek libre | 8 | 2 | Referencia con router aprendido |
| ASI pool seleccionado | 2 | 2 | Residencia mínima para el pool |
| ASI pool, todos residentes | 16 | 2 | Mismo cálculo con más pesos residentes |
| ASI todos, mezcla uniforme | 16 | 16 | Ablación que cambia el cálculo y la distribución de mezcla |
| GPT-2 denso | No aplica | No aplica | Referencia densa, arquitectura diferente |

Las ramas compartidas se suman a los expertos enrutados indicados. ASI y DeepSeek
no tienen el mismo tamaño por experto: igualar parámetros totales no iguala
parámetros activos, cómputo, entrenamiento ni calidad.

Protocolo: ocho prefijos iguales, tres repeticiones por prefijo, batch 1, pesos
FP32 y autocast BF16 en CUDA, greedy, una ejecución de calentamiento por prefijo
y modelo. Todos recalculan el contexto completo por token y producen logits de
todas las posiciones, sin comparar una caché KV con otra implementación sin ella.
La prueba cronometrada genera exactamente 64 tokens aunque aparezca EOS: sus
textos no se presentan como evidencia de calidad. Otras generaciones separadas
sí paran en EOS y se guardan en `review.md`.

Las categorías se declaran en los ejemplos, no se predicen. Esto aísla el coste
de la LLM. Se excluyen carga inicial, clasificador y preparación de pools de la
velocidad mostrada; el JSON registra la preparación por separado. Un archivo
alternativo de prompts es una lista JSON de objetos `{"prompt": "...", "pool":
"biology"}`. Los prompts demasiado largos se rechazan, sin truncar de manera
distinta para cada modelo.

Se distinguen pesos totales, pesos en GPU, buffers, pico CUDA asignado, tamaño de
checkpoint en disco, tiempo al primer token y tokens/s. Los checkpoints de
entrenamiento pueden incluir estados del optimizador: un archivo mayor no
significa una LLM mayor. El pico asignado no incluye la memoria de otros procesos
ni equivale a toda la VRAM que muestra `nvidia-smi`. La GPU de escritorio puede
tener carga ajena: las mediciones no son una garantía de rendimiento.

El informe usa medianas y rangos observados; no son intervalos de confianza.
Los cambios de selección dentro del conjunto residente no son swapping. La
ausencia de cargas dentro de la respuesta no elimina las transferencias al
cambiar de categoría.

Para regenerar las gráficas sin inferencia:

```bash
python -m asi local-generation-report \
  --report results/local_124m_generation_comparison/report.json \
  --output results/local_124m_generation_comparison
```

La calidad se contrasta con los benchmarks en [DEEPSEEK_EVAL.md](DEEPSEEK_EVAL.md).
No se deduce corrección de tokens/s ni coherencia de la ausencia de repeticiones.

## Similitudes, diferencias y calidad ya medida

Los dos MoE locales tienen 12 bloques (11 con MoE), anchura 512, dos expertos
enrutados activos por posición y una rama compartida equivalente a dos expertos.
El DeepSeek pequeño es una implementación inspirada en DeepSeek-V3, no el
checkpoint oficial de cientos de miles de millones de parámetros.

| Propiedad | ASI categorizado | DeepSeek local libre |
|---|---:|---:|
| Parámetros totales | 122.33 M | 123.63 M |
| Expertos enrutados por capa | 16 | 8 |
| Anchura intermedia de cada experto | 120 | 224 |
| Parámetros por experto enrutado | 184 320 | 344 064 |
| Selección | Pareja del pool; mezcla uniforme | Top-2 aprendido, cambia por posición |
| Checkpoint disponible | `model_37595.pt` | `model_07874.pt`, parcial |

Así, compartir un presupuesto de unos 124M no implica la misma capacidad por
experto ni el mismo cómputo activo. La implementación ASI usa su ruta rápida
para pools fijos; DeepSeek conserva su despacho original. Los tiempos comparan
estas implementaciones completas, no un efecto causal aislado del categorizador.
En concreto, el MoE nativo convierte los contadores de routing a una lista en
CPU y despacha expertos en un bucle; ASI aprovecha que la pareja es fija para
evitar esa clasificación interna de tokens. Para atribuir la diferencia a la
política y no al despacho haría falta otra ablación con kernels equivalentes.

Los siguientes aciertos proceden del piloto histórico de 100 ejemplos por tarea,
no de nuevas evaluaciones ejecutadas con el ensayo de generación:

| Benchmark | ASI categorizado | DeepSeek libre parcial | Diferencia ASI, puntos porcentuales |
|---|---:|---:|---:|
| HellaSwag normalizado | 38% | 34% | +4 |
| LAMBADA | 14% | 20% | −6 |
| PIQA | 66% | 63% | +3 |
| ARC-Easy | 42% | 38% | +4 |
| BoolQ | 57% | 69% | −12 |
| Aritmética | 0% | 0% | 0 |

La procedencia de evaluador y datasets coincide entre ASI y DeepSeek en estos
informes. No coinciden las exposiciones de entrenamiento: la diferencia de pasos
no debe interpretarse sin sus respectivos batch globales y datos. La muestra es
pequeña y no se afirma significación estadística ni superioridad general.

El control con todos los pesos residentes y solo un pool activo debería conservar
los mismos tokens que la residencia de dos expertos; el informe comprueba esa
igualdad. Abrir los 16 expertos **activos** es otro experimento: puede modificar
calidad y coste, porque altera la mezcla que el modelo aprendió.

## Ejecución completada el 08/10/2026

RTX 3060 de 12 GB, ocho prefijos, tres repeticiones, 64 tokens. Se completaron
120 generaciones cronometradas y otras 40 con parada EOS, además del calentamiento.
[Gráfica y textos](../results/local_124m_generation_20261008/review.md),
[JSON original](../results/local_124m_generation_20261008/report.json).

| Variante | Pesos GPU, MiB | Pico CUDA, MiB | Tokens/s, mediana |
|---|---:|---:|---:|
| DeepSeek libre, 8 residentes / 2 activos | 471.62 | 576.17 | 7.52 |
| ASI, 2 residentes / 2 activos | 358.35 | 464.91 | 20.04 |
| ASI, 16 residentes / 2 activos | 466.63 | 573.19 | 19.34 |
| ASI, 16 residentes / 16 activos | 466.63 | 573.19 | 6.77 |
| GPT-2 denso | 474.84 | 587.65 | 68.71 |

**Beneficio aislado:** pasar de todos residentes a solo la pareja reduce un
23.2% los pesos GPU del mismo ASI. Coinciden los tokens de las 24 ejecuciones
cronometradas y las ocho salidas EOS del control. El pico asignado baja un 18.9%.
Las velocidades de esas dos variantes tienen rangos solapados: no se demuestra
que sacar pesos inactivos acelere el cálculo.

**Comparación entre implementaciones:** ASI con la pareja es unas 2.66 veces más
rápido que este DeepSeek nativo, pero GPT-2 es unas 3.43 veces más rápido que ASI.
Son observaciones con arquitecturas, kernels y entrenamiento diferentes. El orden
de ejecución fue fijo y no se aisló el equipo de aplicaciones de escritorio;
conviene repetir en un equipo inactivo antes de publicar cifras definitivas.

**Límites del beneficio:** todas las variantes caben en VRAM y registran cero
cargas de expertos dentro de la respuesta. Aquí demostramos ahorro de residencia,
no una ventaja de swapping frente a un modelo que no cabe. Las continuaciones
de ambos MoE siguen mostrando errores y bucles: el router libre tampoco resuelve
automáticamente la coherencia en este checkpoint parcial y con greedy.
