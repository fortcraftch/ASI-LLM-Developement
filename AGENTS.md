# Addendum para AGENTS.md — Protocolo de recuperación v2

> Copia este contenido al final de `AGENTS.md` en la raíz del repositorio (créalo si no existe).
> El plan completo está en `PROTOCOLO_RECUPERACION_V2.md`. Aquí solo van las reglas que se aplican siempre.

## Contexto

Estudio exploratorio de residencia de expertos por categoría en MoE. Esta fase intenta recuperar la calidad de
`allenai/OLMoE-1B-7B-0924-Instruct` (revisión `7f1c97f440f06ce36705e4f2b843edb5925f4498`) cuando se restringe a
K=32 o K=16 expertos residentes por capa (de 64), con A=8 activos por token y pesos top-A sin renormalizar.
Los pesos base de los expertos permanecen congelados: solo se entrenan LoRA, filas del router, ganancias y
expertos compensadores pequeños.

## Reglas inviolables

1. **Lee antes de escribir.** Antes de tocar código crea o actualiza `docs/repo_map.md` con los módulos reales
   (backend de caché de expertos, modelo restringido, scripts de recovery y generation, `build_evidence.py`,
   formato de `evidence/asi_evidence.json`). No inventes nombres de módulos: mapea los existentes.
2. **Aislamiento.** Código nuevo en `recovery_v2/`, configuraciones en `configs/`, resultados en `results/v2/<fase>/`.
   No modifiques ni borres nada de `results/` histórico ni de `evidence/`. En código existente solo se permiten
   hooks mínimos y retrocompatibles, y la suite de tests existente debe seguir pasando antes y después.
3. **Test sellado.** Solo `recovery_v2/eval/final_eval.py` puede leer `data/v2/test/`, y se niega a ejecutarse si
   no existe `results/v2/FREEZE.json` con los hashes de todas las configuraciones finales.
4. **Prohibido seleccionar con el test.** Ni hiperparámetros, ni checkpoints, ni métodos. Los ejemplos listados en
   `data/v2/burned.json` (test v1 y los 3 prompts de generación ya inspeccionados) solo sirven como diagnóstico.
5. **Reanudable e idempotente.** Todo script largo acepta `--resume`, escribe de forma atómica (archivo temporal y
   renombrado), guarda checkpoints con estado del optimizador y orden de datos, y mantiene un `STATUS.json`.
   Un fallo del proceso nunca debe obligar a repetir horas de trabajo. Lanza trabajos largos en segundo plano
   (nohup, tmux o `Start-Process`) y haz polling del log; no bloquees la sesión.
6. **Cada experimento escribe `report.json`** con: commit de git, hash de la configuración, semillas, versiones
   (torch, transformers, CUDA, driver), hardware, hashes SHA-256 de entradas y salidas, métricas con IC bootstrap
   por documento, y `status` ∈ {complete, partial, failed}. Los runs fallidos o parciales se conservan y se
   etiquetan; nunca se reetiquetan como completos.
7. **Gates.** Si un gate del protocolo falla, escribe `results/v2/<fase>/DECISION.md` (qué falló, números, opciones)
   y detente. No avances a la fase siguiente sin instrucción explícita.
8. **No toques el paper** (`.tex`, `.md` del manuscrito, figuras publicadas) hasta F7.
9. **Entorno.** El código de salida `0xC0000005` documentado en el paper sugiere Windows. Por tanto: no dependas de
   Triton ni de `torch.compile`, no uses `resource` ni `fork`, usa multiprocessing con `spawn`, y ejecuta el código
   generado por el modelo dentro de un sandbox (ver protocolo §2.4).
10. **Memoria.** Mide siempre pico de memoria asignada de PyTorch, pico de proceso por NVML, RSS muestreado y bytes
    H2D. Las medidas de memoria y velocidad se reportan en la RTX 3060 de 12 GB; el entrenamiento y el cálculo de
    teacher pueden ir en la RTX 5060 Ti de 16 GB si el benchmark de F0 lo confirma.
11. **Estadística.** Remuestreo por documento o prompt, nunca por token. Con varias comparaciones, describe; no
    afirmes significación sin corrección (Holm).
12. **Una fase por sesión.** Termina con un resumen: qué se ejecutó, qué no y por qué, y qué decide el gate.
