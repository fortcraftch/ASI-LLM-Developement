# AirLLM frente a este proyecto

Revisión de README y código de la rama `main`, consultados el 25/09/2026.
No se ha instalado ni ejecutado AirLLM; no hay mediciones comparativas locales.
Los enlaces apuntan a una rama que puede cambiar.

Compartimos el objetivo de ejecutar modelos mayores que la VRAM disponible.
La diferencia relevante es cuánto tiempo permanecen cargados sus pesos y si
modificamos las decisiones del modelo.

AirLLM divide el checkpoint y carga módulos antes de ejecutarlos. Sus hooks
liberan los pesos después y pueden anticipar la carga siguiente. También tiene
streaming de expertos individuales cuando la arquitectura y los shards lo
permiten: `_expert_pre_hook` carga los pesos y `_expert_post_hook` los devuelve
a `meta`. Por ello, mi lectura del código es que sigue necesitando transferencias
durante la generación autoregresiva; no mantiene una pareja temática durante
toda la respuesta. [Implementación](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_base.py#L681-L836).

No sería correcto describirlo como «solo mueve capas»: el adaptador Kimi declara
`expert_prefix` para activar esa carga individual. No debe suponerse que todos los
MoE usan automáticamente ese camino. [Adaptador](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_kimi_k3.py).

| Pregunta | Nuestra implementación actual |
|---|---|
| ¿Qué dejamos fuera de GPU? | Principalmente expertos enrutados; backbone, expertos compartidos y KV permanecen en GPU. |
| ¿Cuándo cargamos? | La caché exacta carga al necesitarlo; la variante fija prepara la clase antes del turno. |
| ¿Conservamos la decisión original? | Sí en caché exacta. No al restringir candidatos o fijar expertos. |
| ¿Qué hemos demostrado? | El mecanismo funciona en nuestro prototipo. Dos expertos fijos evitaron cargas dentro de las respuestas registradas, pero empeoraron la predicción. |

El README anuncia, entre otros casos, 70B en 4 GB y DeepSeek-V3 en unos 12 GB.
Son cifras del proyecto, no resultados que hayamos reproducido ni garantías de
velocidad conversacional. También ofrece compresión de pesos y señala costes
de almacenamiento. [README](https://github.com/lyogavin/airllm#readme).

El repositorio incluye entrenamiento LoRA: mantiene congelada la base, mueve
capas y entrena adaptadores. Eso es diferente de nuestro entrenamiento que
asigna datos temáticos a grupos de expertos. [Entrenamiento](https://github.com/lyogavin/airllm/blob/main/air_llm/airllm/airllm_lora.py).

Para el TFG, AirLLM es trabajo relacionado y un posible control futuro. Nuestro
argumento no puede ser simplemente «cargar solo lo necesario»: eso tiene
antecedentes claros. La hipótesis específica es conservar pocos expertos según
el contexto para reducir las transferencias durante una respuesta, midiendo la
degradación y el efecto de entrenarlos por temas.

Queda otra limitación nuestra: ahorrar expertos no basta si las partes comunes
del modelo ya exceden la GPU. El streaming de otras partes podría complementar
el sistema, aunque volvería a introducir transferencias. Esa combinación sería
otro experimento, no una propiedad ya implementada.

Una comparación futura necesita el mismo modelo compatible, precisión, entradas,
longitudes y hardware; medir calidad, tiempo hasta el primer token, tiempo por
token, bytes RAM/GPU, lecturas lógicas, RAM y VRAM. Nuestro checkpoint personalizado
no debe considerarse compatible sin adaptar y verificar su carga. No se ha
alterado la batería congelada ni añadido resultados ficticios de AirLLM.
