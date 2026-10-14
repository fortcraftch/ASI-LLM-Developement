# Comparación equivalente MoE: protocolo v2

Estado: recetas preparadas, control de escala implementado y verificación CPU.
**No iniciado el entrenamiento de 124M**. El smoke test CPU del trainer verifica
pesos iniciales y lotes iguales entre libre y categorizado, y reanudación idéntica
a la ejecución continua. No se ha probado aún en la GPU de homelab. No interrumpir
el banco en curso.

## Pregunta y controles

Los checkpoints históricos ASI y DeepSeek 19072 no aíslan la restricción de
expertos: difieren en arquitectura, exposición y mezcla. Se conservan como
referencias exploratorias. La nueva comparación usa el mismo código del modelo
inspirado en DeepSeek, no los pesos del checkpoint 19072.

| Receta en `configs/matched_moe_124m_v2` | Candidatos por capa | Activos por token | Ponderación |
|---|---:|---:|---|
| `free.json` | 16 | 2 | Aprendida, suma seleccionada 1 |
| `pool_learned.json` | 2 del pool de entrada | 2 | Aprendida, suma seleccionada 1 |
| `pool_uniform.json` | 2 del pool de entrada | 2 | Constante 1/2 |

Comparación principal: libre frente a pool aprendido. La tercera receta permite
aislar ponderación aprendida frente a uniforme. En el pool 2/2 no hay elección
de subconjunto: el router solo puede cambiar cuánto pesa cada experto.

Los tres tienen 122.325.504 parámetros, 12 capas (11 MoE), ancho 512, FFN de
experto 120, 2 expertos compartidos y los mismos 8 pools disjuntos. Se cuentan
también los parámetros de gate aunque la variante uniforme no los utilice.
El clasificador externo queda fuera de este presupuesto y debe medirse aparte.

## Por qué controlar la normalización

El comportamiento histórico calcula softmax entre candidatos y **no** vuelve
a normalizar el top-2 cuando usa softmax. Entre 16 candidatos, los dos elegidos
suman menos de uno. Con solo dos candidatos suman uno. Así, restringir cambia
tanto candidatos como amplitud de la contribución MoE al residual.

La opción serializable `normalize_selected_weights: true` aplica normalización
tras seleccionar top-k, antes de `route_scale`, igual en los dos brazos aprendidos.
No cambia la selección para los mismos logits. Los checkpoints sin este campo
conservan `false`: no se reinterpretan resultados ni se altera su inferencia.
El control introduce una arquitectura experimental nueva; no es una réplica
exacta del DeepSeek histórico. Una ablación con escala histórica será separada.

## Entrenamiento fijado

- Misma semilla 1337, inicialización, arquitectura, tokenizador y manifiesto.
- Mismos pools muestreados con raíz del número de tokens y mismas ventanas
  aleatorias con reemplazo, mediante el mismo trainer y RNG independientes de
  la evolución del modelo. No equivale a una pasada exhaustiva sobre los datos.
- Batch de 8 × 1024, acumulación 32, 262.144 tokens por actualización.
- 37.596 actualizaciones: 9.855.565.824 tokens por brazo.
- LR máximo 0,0003, warmup 1000, mínimo 0,1 del máximo, AdamW decay 0,1.
- Guardado cada 25 actualizaciones y retención de tres checkpoints; supervisor
  de errores CUDA. Reanudar una ejecución posterior con `--resume` explícito.
- Evaluación final con el mismo número de tokens entrenados y suite completa;
  no elegir el mejor checkpoint usando test. Publicar datos por semilla; una
  sola semilla no demuestra un efecto robusto.

Comprobación de configuración, sin entrenar ni validar shards completos:

```bash
python -m asi train --recipe configs/matched_moe_124m_v2/free.json --pool-manifest configs/matched_moe_124m_v2/expert_pools.json --dry-run
```

Cuando homelab esté disponible, desde la raíz del proyecto (elegir un brazo):

```bash
export ASI_MOE_BACKEND=bmm
python -m asi train --recipe configs/matched_moe_124m_v2/free.json \
  --pool-manifest configs/matched_moe_124m_v2/expert_pools.json \
  --data-root ./results/dataset_trial --device cuda --pairing-audit \
  --log-dir results/matched_v2/free --auto-restart --save-interval 25 \
  --keep-checkpoints 3 --max-restarts 5 --restart-delay 15
```

Para los otros brazos cambiar `free` por `pool_learned` o `pool_uniform` tanto
en receta como en salida. No cambiar semilla, batch, duración o LR en uno solo.
Tras cerrar el proceso, añadir `--resume ruta/al/model_XXXXX.pt` para continuar.
La auditoría se conserva al reanudar; rechaza código de entrenamiento modificado
y no inventa una inicialización para checkpoints antiguos sin esta evidencia.
No ejecutar los tres brazos automáticamente mientras el equipo esté ocupado.

La opción `--pairing-audit` registra SHA-256 de parámetros iniciales y de los
tokens reales de cada actualización, antes de subirlos a GPU. Su coste de CPU
es parte del entrenamiento instrumentado; no usar ese tiempo como benchmark de
inferencia. Comparación de lo realmente observado:

```bash
python -m asi training-pairing-report --runs results/matched_v2/free \
  results/matched_v2/pool_learned results/matched_v2/pool_uniform \
  --output results/matched_v2/pairing_report.json
```

El informe distingue coincidencia en pasos comunes de cobertura del calendario
completo. No verifica pesos finales ni calidad por leer logs: los checkpoints
se validarán y evaluarán por separado. Conserva logs al reanudar; si solo se
copia el último fragmento, la cobertura aparecerá incompleta correctamente.

## Evaluación prevista

Misma suite, prompts, muestreo, precisión y contexto. Separar categoría conocida
de categoría predicha; la primera aísla la arquitectura. Para residencia,
comparar pool seleccionado con todos los pesos en VRAM manteniendo exactamente
el mismo cómputo. Para calidad, contrastar libre con categorizado. Medir también
instrucciones, respuestas completas, memoria y tiempo de preparación. El banco
independiente es otra arquitectura y no comparte todos estos controles.
