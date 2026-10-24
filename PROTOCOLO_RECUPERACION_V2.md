# Protocolo de recuperación v2 — Residencia fija de expertos en OLMoE-1B-7B

Versión 1.0 · 17 oct 2026 · Para ejecutar con un agente de código (Codex) bajo supervisión del autor.
Reglas permanentes del agente: `AGENTS_addendum.md`.

---

## 0. Objetivo y alcance

**Pregunta principal.** ¿Se puede recuperar la calidad de `OLMoE-1B-7B-0924-Instruct` cuando se restringe a K
expertos residentes por capa (K=32 y K=16 de 64; A=8 activos) y, si no se recupera, el límite es de cómputo de
adaptación o de capacidad?

**Preguntas secundarias.**
- Q1. ¿Dónde se pierde la masa del router y qué capas son sensibles? ¿Aporta algo la categoría frente a una máscara general?
- Q2. ¿Cuánto mejora la selección de máscara (sin entrenar) frente a la calibración actual de 24 ejemplos?
- Q3. ¿Cuánto repara una corrección capa a capa antes del entrenamiento extremo a extremo?
- Q4. ¿Cómo escala el KL con los tokens de destilación? ¿Se aplana?
- Q5. ¿La destilación on-policy elimina los bucles de generación?
- Q6. ¿Qué frontera calidad–memoria–transferencias se obtiene con A mayor, presupuesto de fallos y precisión híbrida, frente a un modelo completo cuantizado a igual memoria?

**Invariantes.**
- Teacher: `allenai/OLMoE-1B-7B-0924-Instruct@7f1c97f440f06ce36705e4f2b843edb5925f4498`, enrutamiento nativo (top-8 de 64, sin renormalizar).
- Pesos base de los expertos congelados. Se entrena solo: LoRA (atención y expertos), filas del router de los expertos residentes, ganancias por capa y compensadores pequeños.
- Las categorías se dan por oráculo (como en el paper). Categorías: `code`, `math`, `bio` y `general` (mezcla, para controles y como relleno).
- Política de enrutamiento restringido por defecto: `reselect` = top-8 entre los residentes con los mismos logits (softmax sobre los 64), sin renormalizar. Política alternativa para diagnóstico: `drop` = top-8 nativo intersectado con los residentes (activa menos de 8).

**Lo que este protocolo no hace:** no entrena el modelo completo, no cambia los archivos base, no afirma nada sobre escala mayor, no reutiliza el test v1.

---

## 1. Fases, dependencias y perfiles

| Fase | Responde | Depende de | Gate |
|---|---|---|---|
| F0 | Backend fiable, harness, datos v2, baselines iso-memoria, presupuesto de cómputo | — | G0 |
| F1 | Cobertura del router por capa y categoría, sensibilidad por capa | F0 | G1 |
| F2 | Mejor método de máscara y K por capa, sin entrenar | F1 | G2 |
| F3 | Reparación capa a capa (ganancia, router, compensador) | F2 | G3 |
| F4 | Ablaciones de recuperación y escalera de tokens (cómputo vs capacidad) | F3 | G4 |
| F5 | Destilación on-policy (DAgger/GKD) contra bucles | F4 | G5 |
| F6 | Frontera de Pareto: A mayor, presupuesto de fallos, precisión híbrida | F4 (F5 opcional) | G6 |
| F7 | Evaluación final sellada y actualización del paper | todas | — |

**Perfil A (mínimo viable, recomendado para empezar).** F0 → F1 → F2 → F3 → F4 con categoría piloto `code`,
K=32 y K=16, escalera hasta 10^7 tokens → F5 solo para `code` y K=32 → F7 reducido.

**Perfil B (completo).** Todo lo anterior, escalera hasta 10^8 tokens en K=32 (y en K=16 si la pendiente lo
justifica), categorías `math` y `bio` hasta 10^7, F6 completo.

Elige perfil tras F0, cuando exista `budget.json` con el throughput real medido.

**Orden de magnitud de cómputo (estimación NO medida; F0 la sustituye por datos reales).** Teacher: ~1.3×10^8
tokens en perfil B. Estudiante: ~2×10^8 tokens de entrenamiento en perfil B. A 0.5–3k tokens/s, del orden de
varios días de GPU en perfil B y de 1–2 días en perfil A.

---

## 2. Convenciones globales

### 2.1 Estructura de directorios

```
recovery_v2/            código nuevo (paquete)
  data/ eval/ analysis/ masks/ repair/ distill/ onpolicy/ pareto/
configs/                YAML por experimento (f0/ ... f7/)
data/v2/                calib/ distill_train/ dev/ test/ (sellado) burned.json manifests/
teacher/<categoria>/    caché de logits del teacher (shards + manifest)
masks/                  máscaras JSON (esquema en Apéndice A)
adapters/               LoRA, router, ganancias, compensadores por (categoria, K)
results/v2/<fase>/      report.json, tablas, figuras, DECISION.md
docs/repo_map.md        mapa del repo existente
prereg_v2.yaml          umbrales y métricas primarias (ver 2.5)
```

### 2.2 Datos y splits v2

| Split | Contenido | Tamaño | Uso |
|---|---|---|---|
| `calib` | texto plano por categoría | 200k tokens por categoría (+ `general`) | F1–F3 (nunca para evaluar) |
| `distill_train` | shards de tokens por categoría | piloto `code`: hasta 10^8; `math`, `bio`, `general`: hasta 10^7 | F4–F5 |
| `train_prompts` | prompts instruccionales por categoría | ≥ 5k por categoría | F5 |
| `dev` | 200 prompts por categoría con tests o respuesta + 100k tokens de texto plano por categoría | — | selección de métodos e hiperparámetros |
| `test` | 200 prompts por categoría | sellado | solo F7 |
| `burned.json` | test v1 + 3 prompts inspeccionados | — | solo diagnóstico |

**Fuentes sugeridas (verifica tamaños y licencias al descargarlas).**
- `code`: MBPP completo para dev/test (partir la partición `test` por semilla fija en dev y test sin solape); texto de código y prompts instruccionales de un corpus permisivo para entrenamiento.
- `math`: GSM8K `test` partido en dev/test; GSM8K `train` y un corpus de matemáticas abierto para entrenamiento.
- `bio`: SciQ `test` partido en dev/test; SciQ `train` y texto científico/biomédico abierto para entrenamiento.
- `general`: FineWeb-Edu (ya usado en el proyecto).

**Composición de `distill_train` por categoría.** 70% texto del dominio, 20% formato chat (prompt + respuesta) y 10% `general`. Se aplica la plantilla de chat del modelo Instruct. La pérdida va sobre todos los tokens en texto plano y solo sobre tokens del asistente + EOS en formato chat.

**Descontaminación.** Elimina de entrenamiento y calibración todo documento con coincidencia exacta o con solape de 13-gramas frente a `dev`, `test` y `burned`. Registra cuántos se eliminan en `data/v2/manifests/decontam.json`. Semilla fija de partición: `20261017`.

### 2.3 Métricas (definiciones)

Todas por categoría y agregadas, con IC bootstrap por documento.

| ID | Nombre | Definición |
|---|---|---|
| M1 | `kl_tf` | KL(pT‖pS) medio por token con prefijos del texto de referencia (teacher forcing), en dev texto plano y en dev formato chat |
| M2 | `ce_tf` | Entropía cruzada frente al token de referencia (compatibilidad con v1) |
| M3 | `kl_onpolicy` | KL(pT‖pS) sobre las trayectorias generadas por el propio estudiante (greedy y muestreo), puntuadas por el teacher |
| M4 | `teacher_nll_of_student` | −log pT(token del estudiante │ prefijo del estudiante), medio por token |
| M5 | `first_divergence` | Posición del primer token donde el greedy del estudiante difiere del greedy del teacher para el mismo prompt (censurado en `max_new`); mediana y curva de supervivencia |
| M6 | `repetition` | Fracción de trigramas repetidos (como el paper), longitud máxima de racha de un mismo token, `loop_rate` (fracción de respuestas con inicio de bucle: un n-grama con n ≤ 8 repetido ≥ 3 veces consecutivas), tasa de EOS antes del tope, longitud media |
| M7 | `task_acc` | `code`: pass@1 ejecutando tests; `math`: coincidencia numérica exacta de la respuesta final; `bio`: acierto de opción múltiple por generación de letra y por log-verosimilitud |
| M8 | `resources` | Pico asignado de PyTorch, pico de proceso (NVML), RSS pico muestreado, cargas de expertos, bytes H2D, tiempo de preparación, tok/s de decodificación (sin el primer token) |

El teacher mide la cota superior de M3–M7. Toda mejora se reporta también como fracción de la brecha teacher–restringido-sin-actualizar.

### 2.4 Ejecución segura de código generado

Evalúa `code` ejecutando los tests en un proceso aislado: directorio temporal, timeout duro (p. ej. 10 s), límite de memoria, sin red. En Windows, `resource` no existe: usa un Job Object (pywin32) o, mejor, ejecuta el sandbox en WSL2/Docker sin red. Tests unitarios obligatorios del sandbox: un bucle infinito es terminado, un intento de red falla, un `rm`/borrado fuera del temporal falla.

### 2.5 Estadística y preregistro

- Comparaciones pareadas: bootstrap por documento (10 000 remuestras, IC percentil 95%). Resultados binarios (tests): McNemar exacto. Semillas de entrenamiento: media ± desviación y puntos individuales (n ≥ 2 donde el presupuesto lo permita).
- Antes de ejecutar F1, crea `prereg_v2.yaml` con: métrica primaria de cada decisión, umbrales de cada gate (los valores por defecto están en cada fase), semillas y tamaños de muestra. Registra su SHA-256 en cada `report.json`. Cualquier cambio posterior se anota como enmienda con fecha y motivo.
- Los umbrales de este documento son valores por defecto razonables; el autor puede ajustarlos antes de F1, no después de ver resultados.

### 2.6 Reanudabilidad y reportes

Ver reglas 5 y 6 de `AGENTS_addendum.md`. Cada fase termina con `results/v2/<fase>/DECISION.md` (gate, números clave, opciones) y una entrada nueva en el registro de evidencia (`build_evidence.py` extendido con E7–E14, ver §11).

### 2.7 Hardware y presupuesto

F0 produce `results/v2/F0/budget.json` con el throughput medido de: prefill del teacher (batch grande), entrenamiento del estudiante (K=32 y K=16), generación batcheada del estudiante (batch 1, 8, 32, 64) y ancho de banda efectivo de carga de expertos (disco y RAM pineada). Todas las estimaciones de tiempo de las fases siguientes se recalculan con esos números.

---

## 3. F0 — Fundamentos: backend, harness, datos y baselines

### Tareas

**0.1 Auditoría del repositorio.** Genera `docs/repo_map.md`. Ejecuta la suite de tests existente y guarda el resultado como línea base.

**0.2 Datos v2.** Construye los splits de §2.2 con manifiestos (hash por shard, fuente, revisión, filtros) y la descontaminación. Escribe `burned.json`.

**0.3 Harness de evaluación (`recovery_v2/eval/`).** Implementa M1–M8, el sandbox (§2.4), el bootstrap pareado y McNemar. Tests unitarios con datos sintéticos: KL entre distribuciones idénticas = 0 (±1e-6); la métrica de repetición reproduce los números del paper sobre los textos archivados de la generación; el sandbox pasa los tres casos de §2.4.

**0.4 Equivalencia del backend con la referencia (puerta crítica).** El paper solo verificó equivalencia numérica en modelos pequeños. Compara el backend de caché exacta con el forward de referencia de Hugging Face sobre ≥ 100 secuencias de 512 tokens del conjunto `calib`:
- Logits: KL media(HF‖backend) y desacuerdo del top-1.
- Con la política restringida y una máscara trivial (los 64 expertos como residentes) el backend restringido debe coincidir con el nativo.
- Si el forward de referencia de HF no cabe en la GPU, usa offload de accelerate.

**0.5 Cargador de expertos (recomendado antes de medir velocidad).** Mide el ancho de banda efectivo de H2D en el backend actual. El paper implica ≈ 0.17 GB/s (98 GiB en ~584 s), muy por debajo de lo que permite PCIe con memoria pineada. Un baseline nativo lento por implementación debilita cualquier afirmación de aceleración. Implementa, como variante opcional del baseline, una caché en RAM pineada con prefetch asíncrono y reporta ambos baselines (el original y el optimizado). No sustituyas el original.

**0.6 Referencias del teacher.** Con generación batcheada, produce para cada prompt de `dev` y `test` la salida greedy del teacher (max 256 tokens) y una muestra a temperatura 0.7 con semilla fija. Guarda token IDs, motivo de parada y logprobs. Esto no filtra etiquetas del test al estudiante (son salidas del teacher, no del estudiante).

**0.7 Baselines sobre `dev` (todas las métricas M1–M8).**

| ID | Descripción |
|---|---|
| B0 | Teacher nativo (backend original y, si existe, el optimizado de 0.5) |
| B1 | Restringido sin actualizar, K=32 y K=16, máscara v1 (24 ejemplos), política `reselect` |
| B2 | Modelo completo cuantizado, iso-memoria: ≈ 4 bits (compárese con el pico de K=16, ~4.0 GiB) y 8 bits (compárese con K=32, ~7.0 GiB) |
| B3 | Máscara aleatoria con el mismo K por capa, sin actualizar (control de la máscara) |
| B4 | Máscara general compartida (selección sobre la mezcla de categorías) con el mismo K |

Para B2 usa lo que funcione en el entorno: bitsandbytes (NF4/int8) vía Hugging Face si soporta el apilado de expertos de la versión instalada, o GGUF (Q4_K_M/Q8_0) con llama.cpp si soporta la arquitectura OLMoE (verifícalo) y su herramienta de KL de `llama-perplexity`. Documenta la elección. Mide la memoria con NVML para que sea comparable entre pilas. Registra que el KL de llama.cpp usa su propio pipeline y no es directamente comparable con M1 salvo que se evalúen los mismos tokens y chunks.

**0.8 Microbenchmarks → `budget.json`** (ver §2.7). Incluye si el teacher cabe entero en la RTX 5060 Ti de 16 GB para el prefill por lotes (≈ 13.8 GB en bf16, previsiblemente justo) y cuál es el mayor batch de prefill que cabe.

### Gate G0
- (a) Equivalencia 0.4: KL media ≤ 1e-3 nats/token y top-1 ≥ 99.5%. Si falla, corrige el backend antes de seguir.
- (b) Tests unitarios y de sandbox en verde; suite histórica en verde.
- (c) Tabla B0–B4 completa con IC y `budget.json` presente.
- (d) **Barra a superar:** escribe en `DECISION.md` cuánto de la brecha cubre B2 a igual memoria. Si B2 ≥ 4 bits ya supera con holgura cualquier valor razonable de recuperación del restringido (lo esperable), el resto del proyecto se justifica por (i) modelos que no caben ni cuantizados, o (ii) la variante híbrida de F6. Es una conclusión legítima y se reporta.

---

## 4. F1 — Cobertura del router y sensibilidad por capa

**Objetivo.** Medir dónde se pierde información al restringir, antes de entrenar nada.

### Tareas

**1.1 Trazas de enrutamiento.** Forward nativo sobre `calib` (200k tokens por categoría + `general`) con hooks en cada bloque MoE. Por token y capa guarda:
- `idx[N,16,8]` (uint8), `w[N,16,8]` (fp16, pesos top-8 sin renormalizar), `n[N,16,8]` (fp16, ‖f_e(h)‖₂ de cada experto seleccionado), y la probabilidad completa del softmax en agregado (no por token).
- Estadísticos agregados por (capa, experto, categoría): conteo, Σw, Σ w·n, Σ n.
- Divide cada categoría en dos mitades: A (para seleccionar) y B (para evaluar). La cobertura se reporta siempre sobre B.

**1.2 Curvas de cobertura.** Para K ∈ {8, 12, 16, 24, 32, 40, 48, 64}, seleccionando por frecuencia sobre A y evaluando sobre B:
- `cov_count` = |S_t ∩ R| / 8.
- `cov_mass` = Σ_{e∈S_t∩R} w_e / Σ_{e∈S_t} w_e.
- `cov_outnorm` = Σ_{e∈S_t∩R} w_e·n_e / Σ_{e∈S_t} w_e·n_e (la más cercana al daño real).
- Reporta por capa, categoría y K. Mapas de calor capa × K y curvas por categoría.
- Entropía y Gini del uso de expertos por capa y categoría.

**1.3 Matriz cruzada de categorías.** Aplica la máscara de la categoría c a los tokens de la categoría c′ (incluye `general` y una máscara aleatoria de igual tamaño). La brecha entre diagonal y fuera de la diagonal es la prueba directa de si el condicionamiento por categoría aporta algo. Definición: `Δcat(K)` = `cov_outnorm`(máscara de la propia categoría) − `cov_outnorm`(máscara `general`) sobre los tokens de la categoría.

**1.4 Sensibilidad por capa (experimento causal barato).** Para cada capa ℓ, restringe solo esa capa a K=32 (resto nativo) y mide M1 en 50k tokens de `dev`. Repite con K=16. Resultado: `S_ℓ(K)` y la suma de contribuciones frente al KL con todas las capas restringidas (¿es aditivo?). Calcula la correlación entre `S_ℓ` y la pérdida de `cov_outnorm` por capa.

**1.5 Figuras y tablas** reproducibles desde `report.json`.

### Gate G1 (informativo; define decisiones)
- **D1.1** Si max_ℓ S_ℓ / min_ℓ S_ℓ ≥ 2 (con S_ℓ por encima del ruido), se prueba K no uniforme en F2.
- **D1.2** Si Δcat < 0.02 (2 puntos de `cov_outnorm`) tanto en K=32 como en K=16, anótalo: las máscaras por categoría se parecen a la general. El experimento sigue (el daño puede deberse a la restricción y no a la categoría), pero el paper debe reformular su premisa y F6 (híbrido) sube de prioridad.

---

## 5. F2 — Selección de máscara y K por capa (sin entrenar)

**Objetivo.** Quitar de la ecuación la mala calibración (24 ejemplos) y comprobar si hay margen sin gastar entrenamiento.

### Métodos de máscara (todos anidados: la de K=16 ⊂ la de K=32)

| ID | Criterio de puntuación del experto e en la capa ℓ y categoría c |
|---|---|
| `v1_freq` | La máscara histórica del paper (24 ejemplos de calibración). Se carga, no se recalcula |
| `freq` | Frecuencia de selección sobre `calib` |
| `mass` | Σ w |
| `reap` | Σ w·‖f_e(h)‖₂ (frecuencia × peso del router × norma de salida) |
| `recon_greedy` | Eliminación hacia atrás que minimiza el error de reconstrucción de la capa bajo la política `reselect` (ver abajo) |
| `random` | Aleatoria, mismo K por capa (control; 3 semillas) |
| `general` | Mismo criterio que el mejor de los anteriores pero sobre la mezcla de categorías |

**`recon_greedy`.** Por capa, con N_recon = 16 384 tokens de `calib` (mitad A): precalcula las salidas de los 64 expertos O[t,e,:] (≈ 4.3 GB bf16 por capa; procesa una capa cada vez en GPU) y los pesos del router W[t,e]. Parte de R = 64 expertos y elimina iterativamente el que menos aumenta ‖m_nativo − m_R‖² / ‖m_nativo‖² con m_R calculada con top-8 entre los residentes (`reselect`). El orden de eliminación da todas las máscaras anidadas desde 64 hasta 8, y la curva de error e_ℓ(K).

### Tamaño de muestra de calibración

Para `reap` y `recon_greedy`, evalúa n_cal ∈ {24 ejemplos (como v1), 2^14, 2^16, 2^18 tokens} por categoría. Es la curva que responde si el problema de calibrar era el tamaño.

### K no uniforme (si D1.1 se cumple)

Con presupuesto Σ_ℓ K_ℓ = 16·K̄ (K̄ ∈ {32, 16}), 8 ≤ K_ℓ ≤ 64, asigna de forma voraz minimizando Σ_ℓ s_ℓ · e_ℓ(K_ℓ), donde s_ℓ = KL de la restricción de solo esa capa (F1.4) dividido por su error relativo e_ℓ(K̄) (aproximación lineal, heurística). Todos los expertos pesan lo mismo, así que la memoria total es igual; verifícalo midiendo el pico. Valida con el KL real de extremo a extremo; no te fíes de la predicción.

### Evaluación (sin entrenar)
M1 (texto plano y chat) en `dev` para cada (método, K, categoría, n_cal, uniforme/no uniforme), M2 y los recursos M8. Informa la `cov_outnorm` de B también como diagnóstico.

### Gate G2
- Adopta el método ganador si reduce M1 ≥ 15% relativo frente a `v1_freq` con IC que excluye cero; si no, conserva `v1_freq` por continuidad y anótalo.
- Selecciona **como máximo 2 configuraciones de máscara por K** para F3–F4 (la mejor y la mejor no uniforme, si procede). Registra el K por capa y los hashes.
- Control obligatorio en la tabla: `random` y `general` frente a la máscara por categoría ganadora, con IC. Si `general` ≈ por categoría, F1.3 y F2 convergen en la misma conclusión.

---

## 6. F3 — Reparación capa a capa

**Objetivo.** Dar a F4 un punto de partida mucho mejor que "restringir y entrenar desde cero".

### Parametrización por capa (todo lo nuevo permanece residente y se contabiliza en `M_adapt`)

Salida restringida reparada:
`m̂_ℓ(h) = g_ℓ · Σ_{e∈A} w̃_e f_e(h) + c_ℓ(h)`, donde:
- `g_ℓ`: ganancia escalar (init por mínimos cuadrados cerrados: g = ⟨m_nat, m_res⟩ / ⟨m_res, m_res⟩ sobre `calib`). Variante por canal (2048 parámetros) como ablación.
- `w̃`: pesos del router, con las filas de los residentes entrenables (K×2048 parámetros por capa, FP32), las demás fijas.
- `c_ℓ(h) = W2 σ(W1 h)`: compensador pequeño siempre residente, ancho oculto 256, `W2` inicializada a 0. ≈ 1M parámetros por capa (~32 MB en total en bf16).

### Procedimiento
1. Cachea las **entradas** h_ℓ de cada bloque MoE (nativo) de `calib`, tokens por categoría configurable (por defecto 100k + 100k de `general`; ≈ 4 KB/token/capa en bf16, ≈ 25 GB para 4 categorías y 16 capas; ajusta al disco disponible y documenta).
2. Por capa: carga sus 64 expertos en GPU (~0.8 GB), calcula la salida nativa y la restringida sobre el lote y minimiza el error relativo MSE (Adam; lr 1e-3 para ganancia y compensador, 1e-4 para el router; 2–3 épocas; 90% entrenamiento / 10% evaluación de la capa).
3. Entrada = activaciones del modelo nativo (capas independientes). Es una aproximación: el error se compone entre capas, por eso F4 cierra el lazo con extremo a extremo.
4. Ablaciones (K=32, una categoría piloto): solo ganancia; ganancia + router; ganancia + compensador; las tres. Por categoría y K.
5. Evalúa el modelo restringido y reparado con M1 en `dev` (extremo a extremo) y M8.

### Gate G3
Usa la reparación como inicialización en F4 si M1 baja ≥ 10% relativo frente a "sin reparar" (IC que excluye cero) en K=32. Si no, desactiva la rama 'repair' de F4 (A4). Reporta cuánta memoria suma `M_adapt` (debe ser ≪ 1 GiB; si no, justifícalo).

---

## 7. F4 — Destilación a escala y escalera de tokens

**Objetivo.** Separar "falta cómputo de adaptación" de "falta capacidad" y fijar la receta.

### 7.1 Caché del teacher (`recovery_v2/distill/teacher_cache.py`)

- Prefill por lotes del teacher sobre los tokens de `distill_train` (secuencias de 1024 tokens empaquetadas con EOS). Nunca decodificación autoregresiva con el teacher: con la caché de expertos en disco es inviable (~0.5 tok/s); el prefill batcheado amortiza la carga de expertos.
- Por token se guardan los **top-k logprobs (k=32) con índices en `uint16`** (el vocabulario cabe en 16 bits) y el logaritmo de la masa de cola `tail_lp = log(1 − Σ p_topk)`. ≈ 4 B por entrada → ≈ 13 GB para 10^8 tokens.
- Shards con manifiesto (SHA-256, revisión del teacher, política de enrutamiento, k, semilla). El caché se comparte entre todos los estudiantes y arms.
- Verificación: para 50 secuencias, la caché reproduce los logprobs de un forward directo del teacher (diferencia máxima ≤ 5e-3 nats) y la masa de cola es coherente.
- Almacén opcional de estados ocultos para la ablación A5 (capas 4, 8, 12; ≤ 2×10^6 tokens; ~4 KB/token/capa).

**Pérdida principal (KL con top-k + cola):**

```python
def kl_topk_tail(t_idx, t_lp, t_tail_lp, s_logits):
    # t_idx [B,L,k] int64 | t_lp [B,L,k] logprob teacher (vocab completo) | t_tail_lp [B,L]
    s_lp = torch.log_softmax(s_logits.float(), dim=-1)
    s_top = s_lp.gather(-1, t_idx)
    s_tail = log1mexp(torch.logsumexp(s_top, dim=-1))      # log(1 - sum p_S(top)), estable
    kl = (t_lp.exp() * (t_lp - s_top)).sum(-1) \
       + t_tail_lp.exp() * (t_tail_lp - s_tail)
    return kl                                              # [B,L]; promediar sobre tokens no enmascarados
```

Procesa los logits en trozos de tokens si no cabe (4×1024×50304 en fp32 ≈ 0.8 GB).

### 7.2 Entrenamiento del estudiante (`recovery_v2/distill/train_distill.py`)

```yaml
name: f4_a3_allproj_r32_percat
category: code
K: 32
mask: masks/<ganadora_F2>_code_K32.json
init: null            # o adapters/repair/code_K32.safetensors (arm A4)
lora:
  attention: {enabled: true, rank: 16, targets: [q_proj, k_proj, v_proj, o_proj]}
  experts:   {enabled: true, rank: 32, targets: [gate_proj, up_proj, down_proj]}
  scope: per_category_and_K        # alternativa: shared_across_categories
router: {train: true, rows: resident_only, lr: 1.0e-4}
loss: {kl: 1.0, ce: 0.0, hidden: 0.0, hidden_layers: [4, 8, 12]}
optim: {lr: 2.0e-4, warmup_frac: 0.03, schedule: cosine, wd: 0.0, betas: [0.9, 0.95], clip: 1.0}
data: {teacher_cache: teacher/code, seq_len: 1024, tokens: 1000000, order_seed: 1337}
batch: {tokens_per_step: 65536, micro_tokens: 4096, grad_checkpointing: true}
eval: {every_tokens: 250000, dev: data/v2/dev}
seed: 1337
```

Requisitos de implementación:
- Base en bf16, parámetros entrenables y router en FP32 (autocast). LoRA de expertos aplicada solo a los tokens enrutados a cada experto, con tokens ordenados por experto y matmul por grupo; **sin bucles Python por token**.
- Atención y router comparten memoria con los pesos comunes; los adaptadores se cuentan en `M_adapt` y deben estar residentes en la GPU durante la generación.
- Monitoriza el uso de expertos dentro del conjunto residente (entropía) por capa; no añadas pérdida de balanceo por defecto.
- Checkpoint con estado del optimizador cada 10^6 tokens, `--resume`, orden de datos determinista por `order_seed`.
- Registra tokens vistos, KL, norma de gradiente y throughput en JSONL.
- Si el estudiante no cabe en la 3060 para entrenar, entrena en la 5060 Ti y evalúa memoria en la 3060.

### 7.3 Ablaciones (categoría `code`, K=32, 10^6 tokens, 2 semillas)

| Arm | Cambio respecto al anterior |
|---|---|
| A0 | Réplica del paper: LoRA rango 8 solo en `down_proj` de expertos + router, pérdida 0.5·CE + 0.5·KL |
| A1 | Pérdida solo KL (auto-destilación pura, sin CE contra referencias) |
| A2 | A1 + LoRA en atención (rango 16) |
| A3 | A2 + LoRA en `gate/up/down` de expertos, rango 32 |
| A4 | A3 inicializado desde F3 (solo si G3 pasó) |
| A5 | A3 + pérdida de estados ocultos (λ pequeño, capas 4/8/12; subconjunto de tokens) |
| A6 | A3 con adaptador compartido entre categorías (entrenado con `code` + `math` mezclados) frente a uno por categoría (comparación pareada en `code` y `math`) |

Selecciona la receta por M1 de `dev` (primaria) y, como desempate, `loop_rate` y M3 en 50 prompts de `dev`. Mide también el KL de cada arm en `dev` de otras categorías para detectar interferencia.

### 7.4 Escalera de tokens

Receta ganadora, categoría `code`. Cada rung es una **ejecución independiente con su propio coseno** (no recortes checkpoints de un run largo: el calendario de lr importa).

| Rung (tokens) | Semillas K=32 | Semillas K=16 |
|---|---|---|
| 1×10^6 | 2 | 2 |
| 3×10^6 | 2 | 2 |
| 1×10^7 | 1 (+1 si hay presupuesto) | 1 |
| 3×10^7 | 1 | 1 |
| 1×10^8 | 1 | solo si la regla de abajo lo justifica |

Perfil A: llega hasta 10^7. Perfil B: completa. Para K=16, solo sube a 10^8 si la pendiente entre 3×10^7 y 10^7 de K=32 supera el umbral de "falta cómputo".

**Análisis.**
- Gráfica log-log de M1 frente a tokens por K, con los IC y los puntos de cada semilla.
- Pendiente log-log entre los últimos dos rungs (criterio no paramétrico) y, si hay ≥ 4 rungs, ajuste `KL(D) = c + a·D^(−b)` con IC bootstrap sobre semillas y documentos. Reporta `c` (suelo extrapolado) y la proyección a 10^9.
- En cada rung final mide también M3–M7 en un subconjunto de `dev` (100 prompts por categoría).
- Si K=32 pasa un umbral razonable en `code`, repite el rung de 10^7 en `math` y `bio` (perfil B) y en K=16.

### Gate G4 (clasificación)
- **Limitado por cómputo:** pendiente log-log ≤ −0.15 entre los dos últimos rungs *y* suelo `c` extrapolado por debajo de τ_KL (por defecto τ_KL = 0.10 nats/token). → continúa a F5 y a rungs mayores.
- **Limitado por capacidad:** pendiente > −0.05 o `c` > τ_KL. → continúa a F6 (híbridos) y reporta el límite como resultado del paper.
- **Ambiguo:** ni lo uno ni lo otro. Añade un rung y declara la incertidumbre.
- Independientemente: reporta cuánto cubre cada rung de la brecha entre B1 (sin actualizar) y B0 (teacher), y compáralo con B2 a igual memoria.

---

## 8. F5 — Destilación on-policy (DAgger/GKD) contra los bucles

**Objetivo.** Entrenar sobre los estados que el estudiante realmente visita, para cerrar el sesgo de exposición que explica los bucles (M6).

### 8.1 Generación batcheada del estudiante
El decode actual es de un solo flujo (~5.4–5.9 tok/s). Implementa generación batcheada con KV cache y padding para el modelo restringido, con las mismas máscaras y adaptadores.

Test de equivalencia: con greedy, la salida batcheada coincide token a token con la de un solo flujo en ≥ 95% de 100 prompts (las diferencias se deben a ruido numérico de bf16; reporta el resto). Mide el throughput para batch ∈ {1, 8, 32, 64}.

### 8.2 Bucle de rondas (R = 3–5)
Para cada ronda r, categoría y K:
1. **Rollouts.** 2 000 prompts de `train_prompts` × 2 muestras (una greedy y una a temperatura 1.0), hasta 256 tokens nuevos con parada EOS.
2. **Puntuación del teacher.** Prefill del teacher sobre prompt + respuesta del estudiante, top-k=32 + cola (reutiliza `teacher_cache.py`). Guarda M3 y M4 de esas trayectorias *antes* de entrenar.
3. **Entrenamiento.** Retoma desde el checkpoint de la ronda anterior. Mezcla por lote: ρ de trayectorias on-policy agregadas de todas las rondas (con peso mayor a las más recientes) y (1−ρ) de datos off-policy de F4 (evita la deriva). Pérdida: KL hacia delante top-k + cola sobre los tokens generados por el estudiante.
4. **Evaluación de la ronda.** M1 (que no debe empeorar), M3–M7 sobre `dev`, y el desglose por posición (KL frente a longitud del prefijo).

**Arms (K=32, `code`):** ρ ∈ {0.25, 0.5, 1.0}; con y sin muestreo a T=1. **Ablación opcional:** JSD generalizada (β=0.5) en lugar del KL hacia delante.

### 8.3 Diagnóstico de bucles
Registra en cada ronda: `loop_rate`, longitud de racha máxima, tasa de EOS, la probabilidad de EOS del estudiante frente a la del teacher en los mismos prefijos, y la masa del router dentro del conjunto residente alrededor del inicio del bucle (valor medio sobre capas, ±20 tokens).

### Gate G5
- Éxito: `loop_rate` ≤ el del teacher + 5 puntos porcentuales, M3 ≥ 30% mejor que el mejor de F4 y M1 no empeora más de lo que indica el IC.
- Plateau: dos rondas consecutivas sin mejora de M3 (IC) → detén y reporta el suelo.
- Si M1 baja pero los bucles persisten, declara que el sesgo de exposición no explica el fallo (o no basta) y registra las hipótesis restantes (capacidad, escala de la mezcla, decodificación).

---

## 9. F6 — Frontera de Pareto: A mayor, presupuesto de fallos, precisión híbrida

**Objetivo.** Sustituir el "todo o nada" (0 cargas frente a miles) por una frontera, y probar si cuantizar los expertos fuera del conjunto residente supera a descartarlos.

### 9.1 Aumentar A dentro del conjunto residente
A ∈ {8, 10, 12, 16} con K fijo, primero solo evaluación con los adaptadores de F4 (sin reentrenar) y después 10^6–10^7 tokens de reentrenamiento a la A elegida. Cuesta cómputo, no memoria ni transferencias. Reporta M1, M3–M7, tok/s y memoria. Indica que A>8 es una desviación de la configuración nativa (A=8) y que no se renormalizan los pesos.

### 9.2 Presupuesto de fallos
Cuando el router nativo elige un experto no residente con peso w_e:
- **P1 (umbral):** si w_e ≥ τ, se carga desde RAM/disco con la caché exacta; si no, se aplica `reselect`.
- **P2 (presupuesto):** como P1, pero con un máximo de B cargas por respuesta (por orden de aparición).
- Reserva S ranuras de desbordamiento por capa (p. ej. 2) con LRU explícito, contabilizadas en la memoria. El experto cargado es el nativo (sin LoRA).
- Barrido de τ para obtener cargas por token ∈ {0, 0.01, 0.03, 0.1, 0.3, 1} (la fracción de las ~128 selecciones por token que son fallos).
- Reporta M1, M3–M7, cargas, bytes H2D, tok/s. Compáralo con el baseline nativo original y con el optimizado de F0.5.

### 9.3 Precisión híbrida (quality-first, con fake-quant)
Para cada capa, los K_hi expertos de la categoría quedan en alta precisión (bf16 o int8) y los demás residentes en 4 bits (RTN por grupos de 128 como línea base simple; GPTQ/AWQ opcionales). Primero **fake-quant** (cuantizar y descuantizar en bf16) para medir calidad; la memoria se calcula analíticamente:
`M_exp ≈ Σ_ℓ [K_hi·s_hi + (64 − K_hi)·s_lo]`, con s_hi ≈ 12.6 MB (bf16) o 6.3 MB (int8), s_lo ≈ 3.3 MB (int4 con escalas); verifica los tamaños reales.
- Presupuestos de memoria {4.0, 5.5, 7.0} GiB (pico asignado), para comparar con B2 (cuantización uniforme) y con K=32/K=16 recuperados.
- Implementa kernels o módulos cuantizados reales **solo si** el fake-quant supera a B2 a igual memoria con IC que excluya cero. Si no lo supera, documenta el negativo y no sigas.

### 9.4 Salida de F6
Una tabla y una gráfica de Pareto (x: pico de GPU en GiB y cargas por token; y: M1, M3 y M7) con todos los puntos: B0, B2, K=32/16 recuperado, +A, +presupuesto, +híbrido. `DECISION.md` indica qué puntos dominan a cuáles.

### Gate G6
Solo informativo: identifica la frontera y señala qué puntos entran en la evaluación final F7 (máximo 6 configuraciones, congeladas en §10.1).

---

## 10. F7 — Evaluación final sellada y actualización del paper

### 10.1 Congelación
Crea `results/v2/FREEZE.json` con: configuraciones finales (máximo 6), hashes de máscaras, adaptadores, semillas de decodificación y el commit. Después de esto no se cambia nada.

### 10.2 Evaluación
`recovery_v2/eval/final_eval.py` (único lector de `data/v2/test`): 200 prompts por categoría (según §2.2), decodificación greedy y muestreo (T=0.7) con 3 semillas, M1–M8 completas, IC por bootstrap pareado y McNemar frente al teacher y frente a B2. Código ejecutado en sandbox. Sin reintentos ni segunda pasada.

### 10.3 Reporte
- Tablas y figuras reconstruibles desde `report.json`; extiende `build_evidence.py` y el JSON de evidencia con las entradas E7–E14 (ver abajo).
- Lista de afirmaciones permitidas según el resultado (§11) y lista de límites que se conservan: una sola máquina, una arquitectura (OLMoE-1B-7B), categorías de oráculo, hardware de consumo, etc.
- Solo entonces, edita el paper.

---

## 11. Árbol de decisión y afirmaciones permitidas

| Resultado | Afirmación que sí se puede hacer | Qué no se puede afirmar |
|---|---|---|
| G4 limitado por cómputo y G5 con éxito | "En K=… la calidad se recupera hasta X% del teacher en M7 con la receta R y N tokens de destilación" | Que generalice a otros modelos o escalas |
| G4 limitado por capacidad | "A K=16 el suelo de KL es ≈ c con esta receta; la restricción dura no es viable en este modelo" | Que ninguna receta lo logre |
| Δcat ≈ 0 (D1.2) | "En OLMoE la categoría no aporta frente a una máscara general" | Que la especialización no exista en otros modelos |
| F6 híbrido > B2 iso-memoria | "Usar la categoría para decidir precisión supera a la cuantización uniforme a igual memoria" | Latencia real hasta tener kernels reales |
| B2 domina todo (esperable) | "A 7B, cuantizar es mejor que descartar; la residencia solo se justifica en regímenes donde ni siquiera cabe cuantizado" | Que la residencia no sea útil a mayor escala |
| M1 baja pero los bucles persisten tras F5 | "El KL en teacher forcing no predice la calidad de generación aquí" | Una causa concreta sin más controles |

**Entradas del registro de evidencia.**

| ID | Fase | Familia de resultados |
|---|---|---|
| E7 | F0 | Equivalencia de backend, baselines B0–B4 |
| E8 | F1 | Cobertura y sensibilidad por capa |
| E9 | F2 | Máscaras y K no uniforme |
| E10 | F3 | Reparación capa a capa |
| E11 | F4 | Ablaciones y escalera de tokens |
| E12 | F5 | Destilación on-policy |
| E13 | F6 | Frontera de Pareto e híbrido |
| E14 | F7 | Evaluación final sellada |

---

## 12. Prompts para Codex (uno por sesión)

Antes de cualquier fase, asegúrate de que `AGENTS_addendum.md` esté incorporado a `AGENTS.md` y que `PROTOCOLO_RECUPERACION_V2.md` esté en la raíz.

**Plantilla.**
> Lee `AGENTS.md` y `PROTOCOLO_RECUPERACION_V2.md`. Ejecuta **solo la fase Fx** (sección §N) siguiendo el orden de las tareas. Respeta el aislamiento (`recovery_v2/`, `results/v2/Fx/`). Escribe tests antes de ejecutar los experimentos. Si un gate falla, escribe `results/v2/Fx/DECISION.md` y detente. Al terminar: resumen con lo ejecutado, lo no ejecutado y el resultado del gate. No inicies la fase siguiente.

**F0 (primera sesión).**
> Lee `AGENTS.md` y el protocolo. Ejecuta F0 (§3). Empieza por `docs/repo_map.md` y la suite de tests existente. No modifiques `results/` ni `evidence/`. Construye los datos v2 con descontaminación, el harness (M1–M8 + sandbox) con tests unitarios, la verificación de equivalencia del backend con Hugging Face (0.4) y los baselines B0–B4. Si el gate G0(a) falla, detente y propón la corrección. Mide los microbenchmarks y escribe `budget.json`.

**F1.**
> Ejecuta F1 (§4). Crea antes `prereg_v2.yaml` con los umbrales por defecto del protocolo y registra su hash. Genera las trazas de enrutamiento, las curvas de cobertura (mitad B para evaluar), la matriz cruzada de categorías y la sensibilidad por capa. Escribe `DECISION.md` con D1.1 y D1.2.

**F2.**
> Ejecuta F2 (§5). Implementa `freq`, `mass`, `reap`, `recon_greedy`, `random` y `general`, con máscaras anidadas y el barrido de n_cal. Si D1.1 lo indicó, implementa el K no uniforme y valida con KL real. Selecciona ≤ 2 máscaras por K según G2.

**F3.**
> Ejecuta F3 (§6). Cachea las activaciones de entrada por capa, entrena ganancia, filas del router y compensador por capa con MSE relativo y evalúa extremo a extremo. Aplica G3.

**F4.**
> Ejecuta F4 (§7). Primero `teacher_cache.py` con su verificación. Luego las ablaciones A0–A6 (1M tokens, 2 semillas) y, si el presupuesto lo permite, la escalera de tokens para K=32 y K=16 en `code`. Cada rung es un run independiente. Genera el ajuste log-log y la clasificación de G4. Lanza los runs largos en segundo plano con `--resume`.

**F5.**
> Ejecuta F5 (§8). Implementa y verifica la generación batcheada, luego el bucle de rondas con puntuación del teacher y entrenamiento mixto on/off-policy. Registra el diagnóstico de bucles en cada ronda. Aplica G5.

**F6.**
> Ejecuta F6 (§9): barrido de A, presupuesto de fallos con ranuras de desbordamiento y precisión híbrida con fake-quant. Genera la frontera de Pareto y `DECISION.md`.

**F7.**
> Ejecuta F7 (§10). Crea `FREEZE.json`, luego ejecuta `final_eval.py` una sola vez. Actualiza `build_evidence.py` y el JSON de evidencia con E7–E14. No edites el paper: entrega una lista de afirmaciones permitidas según §11.

---

## Apéndice A — Esquemas

**Máscara (`masks/*.json`).**
```json
{
  "schema": "mask_v2",
  "model_revision": "7f1c97f440f06ce36705e4f2b843edb5925f4498",
  "category": "code",
  "method": "recon_greedy",
  "policy": "reselect",
  "calib": {"tokens": 262144, "manifest_sha256": "..."},
  "K_per_layer": [32, 32, 32, "... 16 valores"],
  "resident": {"0": [0, 3, 5], "1": [2, 4, 9]},
  "nested_from": "masks/recon_greedy_code_K32.json",
  "sha256": "..."
}
```

**`report.json` (mínimo).**
```json
{
  "phase": "F4", "experiment": "f4_a3_allproj_r32_percat", "status": "complete",
  "git_commit": "...", "config_sha256": "...", "prereg_sha256": "...",
  "seeds": [1337], "hardware": {"gpu": "RTX 3060 12GB", "driver": "...", "ram_gb": 16},
  "versions": {"torch": "2.11.0+cu128", "transformers": "..."},
  "inputs": {"teacher_cache_sha256": "...", "mask_sha256": "..."},
  "outputs": {"adapter_sha256": "..."},
  "metrics": {"kl_tf": {"mean": 0.0, "ci95": [0.0, 0.0], "n_docs": 200}},
  "resources": {"peak_alloc_gib": 0.0, "peak_nvml_gib": 0.0, "h2d_gib": 0.0, "loads": 0},
  "notes": "incidencias, runs fallidos, desviaciones del protocolo"
}
```

## Apéndice B — Riesgos conocidos

- **Fuga entre splits:** la descontaminación y el sellado del test son obligatorios; si se detecta solape después, invalida el test y crea uno nuevo.
- **Rendimiento de la LoRA en expertos:** una implementación ingenua con bucles Python por token puede costar un orden de magnitud de throughput; usa agrupación por experto.
- **Tokenización y plantilla:** usa el tokenizador y la plantilla de chat de la revisión fija; verifica EOS y padding en la generación batcheada.
- **Caché del SO:** el paper no vació la caché de página. Reporta el estado (frío/caliente) de cada medida o aleatoriza el orden.
- **Comparabilidad entre pilas (HF, llama.cpp, backend propio):** declara qué se mide en cada una y no mezcles KL de pipelines distintos sin control.
- **Aditividad de la sensibilidad por capa (F1.4):** si no es aditiva, la asignación no uniforme de F2 es solo una heurística; compruébalo antes de confiar en ella.
- **Teacher débil en tareas:** OLMoE-1B-7B-Instruct falla incluso el ejemplo de matemáticas del paper. M7 se interpreta relativo al teacher, no en términos absolutos.
