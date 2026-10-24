# Control libre equivalente a ASI (DeepSeek-style)

Este lanzador reutiliza el entrenador ASI y la receta existente `base_16_learned.json`. No usa el entrenador histórico con expertos de anchura224: ese cambiaría arquitectura, datos y calendario. La nueva comparación cambia el routing, no el tamaño de los expertos.

## Configuración

- 122.325.504 parámetros totales, igual que ASI.
- 16 expertos enrutados por capa, 2 activos/token; anchura120.
- Rama compartida equivalente a2 expertos, 12 bloques, 11 capas MoE, dimensión512; MLA idéntica.
- Routing aprendido top2 sobre los16, sin restricción por categoría; no se fuerza ningún reparto equilibrado adicional.
- ASI usa pareja fija y pesos uniformes; este control aprende la selección y los pesos de mezcla con el router libre.
- Mismo dataset categorizado y mismo manifiesto congelado; el muestreo sqrt usa las categorías para seleccionar datos, NO para limitar expertos.
- Seed1337, batch8, longitud1024, batch efectivo262144 tokens (32 microbatches),37596 actualizaciones, LR3e-4, warmup1000, mínimo0.1×LR, weight decay0.1, orden random.
- Batch8 se verificó en `training_settings` y `config` de `model_37595.pt`; prevalece sobre el default2 de la receta. La receta base y domain solo difieren en family y routing_mode.
- Se entrena todo desde cero: backbone, expertos compartidos/enrutados y router. No reanudar pesos del ASI categorizado.
- Guardado cada25 actualizaciones, conserva3; máximo5 reinicios CUDA con15s de espera. Ctrl+C no se reinicia; pierde como máximo el progreso desde el último checkpoint.
- `--pairing-audit` conserva hashes de inicialización y batches para futuras comparaciones. Sin auditoría histórica completa no se afirma identidad bit a bit de todos los datos vistos por ASI.

## Homelab

Desde la raíz ASI-LLM-Developement, con el entorno activado y estos archivos copiados/actualizados:

```bash
bash configs/deepseek16_v1/train_homelab.sh results/dataset_trial --dry-run
bash configs/deepseek16_v1/train_homelab.sh results/dataset_trial
```

`results/dataset_trial` es un ejemplo: usar la carpeta real con `manifest.json` y shards categorizados con la que se entrenó ASI. No usar `/results/...` salvo que realmente esté en la raíz del sistema. El lanzador verifica identidad del manifiesto; no lo modifica para esquivar un error.

Salida: `results/deepseek16_asi_matched_seed1337`. Para reanudar tras cerrar la sesión, elegir el último checkpoint de esta ejecución:

```bash
bash configs/deepseek16_v1/train_homelab.sh results/dataset_trial --resume results/deepseek16_asi_matched_seed1337/model_XXXXX.pt
```

Para que sobreviva a desconectar SSH, ejecutarlo dentro de tmux. No lanzar dos copias a la vez. La configuración aumenta candidatos respecto al control antiguo, pero mantiene el tamaño y parámetros de ASI. La velocidad, ocupación y convergencia pueden diferir por el routing.

## Validación local

Recetas comparadas, parámetros contados en meta, ajustes contrastados contra checkpoint final ASI; test de activación base comprueba acceso a todas las pools. No se ha lanzado el entrenamiento largo ni validado consumo de VRAM en homelab. La preparación anterior de153.95M quedó descartada antes de entrenar; su dry-run sintético es histórico y no corresponde a este control.
