# Diseño de capacidad (sin entrenamientos ni resultados)

Las recetas ejecutables se encuentran en `recipes/`, con manifiesto y presupuestos propios. No se ha lanzado ningún entrenamiento.

| Variante | Parámetros/experto | Activos MoE/capa | Compartidos/capa | Total modelo |
|---|---:|---:|---:|---:|
| capacity_h60_k2_uniform_pool | 92160 | 184320 | 368640 | 106105344 |
| capacity_h60_k2_learned | 92160 | 184320 | 368640 | 106105344 |
| capacity_h120_k2_uniform_pool | 184320 | 368640 | 368640 | 122325504 |
| capacity_h120_k2_learned | 184320 | 368640 | 368640 | 122325504 |
| capacity_h240_k2_uniform_pool | 368640 | 737280 | 368640 | 154765824 |
| capacity_h240_k2_learned | 368640 | 737280 | 368640 | 154765824 |
| granularity_h240_k1_uniform_pool | 368640 | 368640 | 368640 | 122280448 |
| granularity_h240_k1_learned | 368640 | 368640 | 368640 | 122280448 |
| granularity_h120_k2_uniform_pool | 184320 | 368640 | 368640 | 122325504 |
| granularity_h120_k2_learned | 184320 | 368640 | 368640 | 122325504 |
| granularity_h60_k4_uniform_pool | 92160 | 368640 | 368640 | 122415616 |
| granularity_h60_k4_learned | 92160 | 368640 | 368640 | 122415616 |

Las variantes repetidas entre ejes son el mismo control y deben reutilizarse, no entrenarse dos veces.
En granularity se conserva k × ancho y el ancho compartido; en capacity solo aumenta la capacidad enrutada. Los parámetros del router varían con el número de expertos. El total no mide FLOPs ni memoria temporal.
No se puede convertir esta tabla en un mínimo de parámetros para comprender lenguaje sin entrenar y evaluar modelos completos.

Actualización: `recipes/` contiene diez recetas ejecutables validadas y comandos comentados. El barrido no se ha entrenado. Las variantes de 154,8M tienen presupuesto explícito propio, superior al control original de 124M; no se han lanzado.
