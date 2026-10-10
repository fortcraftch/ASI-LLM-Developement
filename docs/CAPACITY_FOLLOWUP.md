# Punto 9: profundizar en la concentración del routing

El primer piloto encontró que las variantes libres de 16 expertos repartían sus
tokens como aproximadamente cinco expertos uniformemente utilizados. No estaban
apagados los demás: era concentración del trabajo. Este seguimiento estudia si
esa concentración persiste y si está relacionada con la categoría del texto.

## Protocolo fijado antes de ejecutar

Configuración: `configs/capacity_followup_v1.json`.

- Solo la arquitectura actual: 122.325.504 parámetros, 16 expertos enrutados y dos
  activos por token, con los mismos expertos compartidos y backbone.
- Dos políticas nativas: pool uniforme restringido y router softmax libre aprendido.
- Tres semillas de inicialización: 1337, 2027 y 31415. Inicialización de parámetros
  comunes idéntica dentro de cada pareja.
- Semilla de datos 1337 independiente de la inicialización. Las seis ejecuciones
  reciben las mismas ventanas, orden de pools y objetivos de validación.
- 128 actualizaciones por ejecución, 131.072 tokens por modelo, contexto 128,
  batch 2 y acumulación 4. Son cuatro veces los pasos del piloto inicial.
- Validación cada 32 actualizaciones sobre 16.384 tokens (2.048 por categoría).
  Se informa la media de las ocho categorías y sus pérdidas individuales.
- AdamW, LR 1e-4, warmup de cuatro pasos, weight decay 0,1 y clipping a 1.
  No se añade regularización de balanceo ni se cambia la política durante el ensayo.

Las métricas principales se calculan **por intervalos de 32 pasos**. Un promedio
de todo el entrenamiento podría esconder que los expertos dominantes cambian
o que el desequilibrio se agrava después.

## Qué significan las mediciones

1. **Expertos efectivos:** resume la concentración de las asignaciones de tokens
   mediante entropía. Puede bajar a cinco aunque los 16 hayan recibido tokens.
   El número de expertos ejecutados para cada token sigue siendo dos.
2. **Asociación entre categoría y experto:** información mutua normalizada por la
   entropía de las categorías. Cero indica que el reparto observado no depende de
   la categoría; uno, que los expertos separan completamente las categorías. No
   prueba especialización semántica ni ausencia/presencia de conocimiento.
   El restringido alcanza uno por construcción, no porque haya demostrado aprender.
3. **Suma de pesos de mezcla:** el pool uniforme suma uno. El routing libre actual
   toma dos pesos de un softmax sobre 16 sin renormalizarlos, por lo que puede sumar
   menos de uno. Esto cambia la amplitud de la rama enrutada y es un factor distinto
   de cuántos parámetros tiene el modelo. Se mide sin modificarlo.
4. **Pérdida de validación:** comparación pareada dentro de cada semilla, con datos
   idénticos. Se muestran las diferencias y su variación, sin presentar tres semillas
   como prueba de significación estadística ni como evaluación de lenguaje final.

No se identifica el mismo experto entre semillas: sus índices no garantizan la
misma función aprendida. Las métricas de reparto permiten comparar concentración
sin asumir esa correspondencia.

## Reproducir

Desde la raíz del proyecto, usar una salida nueva:

```bash
for seed in 1337 2027 31415; do
  python -u -m asi capacity-pilot \
    --output "results/capacity_followup_nuevo/seed_${seed}" \
    --seed "$seed" --data-seed 1337 \
    --steps 128 --eval-interval 32 --val-batches 8 \
    --variants capacity_h120_k2_uniform_pool capacity_h120_k2_learned \
    --device cuda || break
done
python -m asi capacity-compare \
  --runs results/capacity_followup_nuevo/seed_1337 \
         results/capacity_followup_nuevo/seed_2027 \
         results/capacity_followup_nuevo/seed_31415 \
  --output results/capacity_followup_nuevo/comparison
```

Si hubo una interrupción, repetir el comando del piloto afectado con `--resume`:
se omiten variantes terminadas y se reinicia desde cero la interrumpida. No se
guardan pesos ni se reanuda el optimizer a mitad de una variante.

`capacity-compare` rechaza ejecuciones incompletas, semillas duplicadas, datos o
ajustes distintos, presupuestos de tokens diferentes y parejas con inicializaciones
comunes distintas. Comprueba el contenido de las matrices de datos, no solo el
nombre de un archivo. Los JSON retienen las matrices categoría × experto por capa
y las mediciones acumuladas que permiten reconstruir cada intervalo.

La ejecución local está en `results/capacity_followup_20261005/`. Incluye una copia
del protocolo y de los módulos principales usados al iniciar el ensayo. El informe
combinado se genera únicamente cuando terminan las seis ejecuciones.

## Qué se puede decidir después

### Resultado completado, 06/10/2026

Actualización 07/10: el control adicional `--normalize-selected` también terminó
con tres semillas. El cambio de NLL libre es −0,0242 / −0,0338 / +0,0035 (media
−0,0182), mientras el control uniforme sin intervención cambia +0,0108 / −0,0003 /
+0,0264. No hay una mejora consistente atribuible únicamente a normalizar.
Los expertos efectivos siguen alrededor de 4,4–4,7 y la asociación con categoría
permanece baja. Véase `results/capacity_normalized_20261006/comparison/normalization_effect.md`.

Las seis ejecuciones finalizaron. En el último intervalo, el router libre presenta
5,11 / 4,63 / 4,20 expertos efectivos para las tres semillas y asociación con el
pool de 0,0137 / 0,0120 / 0,0103. Los pesos seleccionados suman de media
0,304 / 0,315 / 0,333. El restringido tiene 16 expertos efectivos, asociación uno
y suma uno por construcción del reparto equilibrado.

La diferencia media pareada de NLL (restringido menos libre) es −0,0100.
No demuestra significación ni una diferencia en capacidad de conversación.
El informe y las gráficas están en
`results/capacity_followup_20261005/comparison/summary.md`.
Queda completado este seguimiento corto; el barrido de preentrenamientos largos
y el control separado de normalización siguen pendientes.

Si persiste la concentración con poca asociación a las categorías, el siguiente
control debería aislar la normalización de los pesos seleccionados, antes de
atribuirlo al tamaño de los expertos. Después se podría contrastar un mecanismo
de balanceo, manteniendo datos y presupuesto. Son hipótesis pendientes: este
seguimiento observa las políticas existentes, no demuestra una causa.

Este estudio sigue usando una sola muestra local de datos, contexto corto y muy
pocos tokens para preentrenar una LLM. No permite estimar un mínimo de parámetros
para hablar bien ni reemplaza la futura evaluación de DeepSeekV3 entrenado.
