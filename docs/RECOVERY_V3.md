# Recuperación ampliada: datos públicos, profesor y tres ajustes

Estado verificado el 29/09/2026: corpus construido, calibración terminada con 24
ejemplos train y cuatro referencias reales del profesor guardadas/verificadas.
Estas cuatro referencias ocuparon 74,47 MiB y su pasada agrupada tardó 70,71 s
en la RTX 3060 con caché RAM de 256 MiB, sin incluir carga inicial del modelo.
No es una estimación fiable del tiempo en homelab con una caché RAM mayor.
Los nueve entrenamientos ampliados, el control nativo completo y la revisión
humana siguen pendientes; el perfil parcial no es un resultado de recuperación.

La batería pasó **110 tests**. Dos actualizaciones técnicas de router + LoRA en
la RTX 3060 también terminaron: una con CE+KL real (169 tokens, pico 7,59 GiB) y
otra con CE sobre el ejemplo train más largo (384 tokens, pico 8,53 GiB). Ambas
mantuvieron cero cargas de expertos dentro del paso. Los pesos se descartaron;
no son resultados de calidad ni garantía de memoria para toda ejecución. Registro:
`results/recovery_v3_gpu_probe.json`.

Continuación de los pasos 2.2 y 2.3 del TFG. El piloto v2 mostró recuperación
parcial con 32 residentes; **este protocolo no contiene todavía resultados de los
entrenamientos ampliados**. El objetivo es averiguar qué se recupera y a qué coste,
sin aumentar el rango de LoRA ni cambiar su proyección de salida.

## Datos y separación

`python -m asi data recovery-corpus` construye el corpus desde versiones fijadas:

| Fuente | Uso | Licencia declarada |
| --- | --- | --- |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Problemas con explicación y resultado numérico | MIT |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | Instrucciones para escribir Python; referencias y tests | CC-BY-4.0 |
| [SciQ](https://huggingface.co/datasets/allenai/sciq) | Preguntas y explicaciones de biología seleccionadas por vocabulario | CC-BY-NC-3.0 |

Se conservan repositorio, revisión, hashes de archivos, ID y split originales.
El corpus derivado incluye material SciQ no comercial. No se ejecuta código de
los datasets al importarlos: solo se leen archivos Parquet.

La construcción local produce **3.745 train, 168 dev y 180 test**:

| Categoría | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| Matemáticas | 1.500 | 60 | 60 |
| Programación | 745 | 48 | 60 |
| Biología | 1.500 | 60 | 60 |

Los grupos reservados son geometría/velocidades, ordenación/cadenas y
genética/ecología. Se asignan por reglas léxicas transparentes; una pregunta que
coincida con varios grupos usa la primera regla. Se eliminan duplicados de la
pregunta normalizada sustituyendo cifras, incluidos grupos que crucen particiones.
Se rechazan ejemplos que excedan 384 tokens completos, sin truncarlos.

**No son los splits oficiales de esos benchmarks.** Los splits originales se
reagrupan deliberadamente; no publicar estos resultados como puntuaciones GSM8K,
MBPP o SciQ oficiales. Los grupos léxicos y la normalización no prueban ausencia
de paráfrasis o contaminación semántica. Tampoco podemos excluir que el modelo
original viera estas fuentes durante su preentrenamiento. El filtrado SciQ puede
incluir temas fronterizos de química/medioambiente y explicaciones con referencias
a figuras ausentes; revisar la calidad del corpus antes de extraer conclusiones.

```bash
python -m asi data recovery-corpus --download --output results/recovery_corpus_v3/corpus.json
```

Omitir `--download` para reconstruir a partir de `data/recovery_sources_v3`.
Usar una salida nueva. El corpus y las fuentes están en directorios de datos/resultados
ignorados por Git; copiar el corpus al otro equipo o reconstruirlo con este comando.

## Calibración y profesor compartido

```bash
python -m asi recovery calibrate --examples results/recovery_corpus_v3/corpus.json --mask-sizes 32 16 --calibration-per-category 8 --reference-batch-size 4 --max-length 384 --output results/recovery_calibration_batched_v3
python -m asi recovery-study plan --output results/recovery_study_v3
```

La calibración solo utiliza train. Los tamaños 16 y 32 se fijan antes de ajustar
pesos. La CLI comprueba la identidad del corpus y que ningún ID de calibración
proceda de dev/test. El batching derecho usa máscara de atención; el relleno no
se cuenta al clasificar expertos. No cambia el batch del alumno, que sigue siendo 1.

El profesor usa el routing **nativo sin restricciones** sobre respuestas de referencia.
Con `--teacher-eos` también predice el final de respuesta. La pérdida del alumno es
0,5 × CE de referencia + 0,5 × KL profesor/alumno sobre tokens del asistente.
Las probabilidades se guardan completas en FP32: no se introduce aproximación top-k.
No generamos respuestas sintéticas del profesor para sustituir automáticamente las
referencias; imitarlo no garantiza que su respuesta o la referencia sean correctas.

Los tres ajustes comparten exactamente los mismos archivos del profesor. La identidad
v3 depende de fuente, corpus y objetivo; no de la selección de expertos del alumno.
Los checkpoints del alumno sí quedan vinculados a sus selecciones. El lector guarda
solo metadatos en RAM y carga **una distribución por ejemplo** al calcular la pérdida.
Verifica hashes e identidad antes de entrenar; un manifiesto parcial no es válido.

Estimación local del almacenamiento de probabilidades: **65,96 GiB**, más archivos
auxiliares y checkpoints. El modelo público ocupa otros aproximadamente 13 GiB.
Estar por debajo de 650 GB no garantiza velocidad: con poca RAM puede dominar la
lectura del disco. No confundir las lecturas lógicas con tráfico físico de SSD.

El plan permite hasta 16 GiB de caché de expertos en RAM, dentro del presupuesto
de proceso de 32 GiB (`--expert-ram-gib 16 --ram-gib 32`). No los reserva ni exige:
el almacén reduce su caché según memoria disponible y RSS. En un equipo con poca
RAM cabrán menos expertos; en homelab se podrá aprovechar más RAM para el profesor.
El valor histórico por defecto, fuera del plan, sigue siendo 0,25 GiB. Esto cambia
el coste de lecturas, no la selección matemática del profesor.

Para medir cuatro referencias antes del trabajo completo, una vez calibrado:

```bash
python -m asi recovery prepare --examples results/recovery_corpus_v3/corpus.json --study results/recovery_calibration_batched_v3/report.json --teacher-eos --max-length 384 --reference-batch-size 4 --prepare-limit 4 --teacher-dir results/recovery_study_v3/teacher --output results/recovery_teacher_profile_v3
```

El informe registra tiempo y número de ejemplos nuevos; su `completed` permanece
falso porque es un perfil parcial. No extrapolar cuatro preguntas como un benchmark
estable. Para completar, `recovery-study run` detecta ese manifiesto y añade
`--resume-teacher`. Ante otra interrupción, `recovery prepare --resume-teacher`
requiere el mismo teacher-dir y un output nuevo; conserva y verifica lo ya guardado.

## Comparación inicial: solo 32 residentes

El plan contiene tres semillas (1337, 2027 y 31415), una época por variante y 8
expertos activos por token. Las variantes son **router**, **expert-lora** y
**router-expert-lora**, con LoRA de rango 8 en la salida de los expertos. Cada una
ve los mismos ejemplos en el mismo orden dentro de una semilla. Los tamaños de
categoría son distintos: no se presenta este corpus como balanceado.

```bash
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage teacher
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage native
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage router_1337
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage expert-lora_1337
python -m asi recovery-study run --plan results/recovery_study_v3/plan.json --stage router-expert-lora_1337
```

Repetir las tres últimas etapas para 2027 y 31415. Las etapas son explícitas y
secuenciales: el comando `plan` no inicia nueve entrenamientos. Una salida existente
no se sobrescribe. El autostart de `asi train` pertenece al entrenador categorizado;
no se aplica a `recovery train`, que todavía no reanuda su optimizador.

Dev selecciona el checkpoint por el peor cociente de pérdida entre categorías,
incluyendo el modelo sin ajustar. Test se reserva para la comparación final. Las
generaciones tienen EOS y un máximo de 256 tokens; registrar respuestas truncadas,
no suponer que todas finalizaron. El control nativo usa los mismos prompts test y
presupuesto, con sus propias transferencias de expertos. Las comparaciones de
latencia son exploratorias sin repeticiones ni control de caché fría/caliente.

```bash
python -m asi recovery-study summarize --reports results/recovery_study_v3/native/report.json results/recovery_study_v3/router_1337/report.json results/recovery_study_v3/expert-lora_1337/report.json results/recovery_study_v3/router-expert-lora_1337/report.json --output results/recovery_summary_v3
python -m asi graphs --results results/recovery_summary_v3 --output results/recovery_graphs_v3
```

Añadir los informes de las otras semillas al mismo comando para obtener media y
desviación entre semillas. Se comprueban exposiciones, profesor, corpus, fuente y
preguntas generadas. No mezclar resultados de diferentes semillas como si fueran
repeticiones de latencia.

## Corrección, coherencia y seguimiento de instrucciones

El sistema puntúa automáticamente el número final de matemáticas, formato exigido,
sintaxis Python y presencia de la respuesta corta SciQ. **Sintaxis válida o mencionar
una palabra no significa responder correctamente.** No ejecuta código generado en
el equipo. Los tests de referencia se entregan para revisión en un entorno aislado.

```bash
python -m asi recovery-study review --examples results/recovery_corpus_v3/corpus.json --reports results/recovery_study_v3/native/report.json results/recovery_study_v3/router_1337/report.json results/recovery_study_v3/expert-lora_1337/report.json results/recovery_study_v3/router-expert-lora_1337/report.json --output results/recovery_review_v3
```

`review.jsonl` presenta pregunta, referencia y respuesta sin variante identificable
por metadatos. La rúbrica pide puntuar 0/1/2 en corrección, coherencia y cumplimiento.
Los valores permanecen vacíos hasta una revisión real; no se inventan puntuaciones.
Mantener `key.json` y los chequeos automáticos fuera de la vista del revisor hasta
terminar. Guardar también longitud, repetición, tiempo, memoria, transferencias y
la comprobación A→B→A de los informes originales.

## Después: bajar a 16

Ya existe el calendario `--mask-sizes 32 16 --curriculum-32-steps N`: comienza con
32 durante N pasos y después alterna 32/16 con exposición reproducible. Compararlo
con alternancia desde el principio, con el mismo total de pasos y presupuesto;
documentar que la exposición a cada máscara difiere por diseño. Es un estudio
separado, no una búsqueda de hiperparámetros guiada por los resultados test de arriba.
No se han ampliado las proyecciones LoRA ni entrenado todas las matrices densas.
