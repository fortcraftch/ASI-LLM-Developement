# Comparación de ejecución GPT-2 / banco denso

## Diferencia encontrada

El número de parámetros no determina el tiempo de ejecución. El banco comparte
la arquitectura MLA de ASI: proyecciones latentes, RoPE, RMSNorm y atención con
8 cabezas de dimensión QK 192 y V 128. GPT-2 usa atención convencional, con
12 cabezas de dimensión 64 y SDPA. Embeddings y dimensiones de FFN también
distribuyen de forma distinta sus parámetros y operaciones.

Además, la ruta de inferencia histórica MLA usa factoración absorbida, varias
operaciones einsum y cachés, aunque la prueba temporal recalcula todo el prefijo.
GPT-2 ya usa SDPA causal. Ambos proyectaban logits para todo el prefijo.
Esto permite un control de implementación sin entrenar ni cambiar pesos.

## Protocolo iniciado el 23/10/2026

Script aislado `recovery_v2/dense_inference_audit.py`; resultados en
`results/v2/F0/dense_inference_20261023`. No modifica los modelos de producción.

- Banco: baseline, último logit solamente, MLA directa SDPA + último logit.
- GPT-2: baseline y último logit solamente; ya usa SDPA.
- Ocho prompts históricos, 64 tokens, batch 1, pesos FP32/autocast BF16,
  greedy, tres repeticiones, cuatro threads CPU, calentamiento por variante.
- Orden aleatorio de las variantes y repeticiones dentro de cada prompt.
- Todas las variantes recalculan el prefijo completo; ninguna recibe ventaja
  por usar generación incremental frente a otra que no la usa.
- Se conserva explícitamente el softmax_scale de inferencia MLA. No se cambia
  el modelo a train para reutilizar una rama con semántica potencialmente distinta.

Logits de control: prefijo inicial y prefijos con 16/48 tokens del baseline,
en FP32 y BF16. Se registra error máximo y allclose (rtol/atol 1e-4).
Generación: igualdad exacta de tokens en una respuesta fija y otra con EOS,
por prompt. Es una comprobación diagnóstica, no una garantía para todo input.

Solo una variante con equivalencia FP32 y tokens idénticos en todos los casos
puede presentarse como optimización equivalente en esta prueba. Las diferencias
BF16 también deben explicitarse aunque no cambien el argmax. Una variante que
cambia respuestas se conserva como intervención numérica separada; no sustituye
las cifras históricas del paper ni se atribuye su velocidad a la misma salida.

Los tiempos históricos de ASI y DeepSeek no se mezclarán con rutas nuevas como
si fueran una comparación simultánea y uniformemente optimizada. Primero se
presentará la ablación GPT-2/banco; extender a todos exige verificar sus rutas.

No hay entrenamiento, selección con test sellado ni avance de fase F0. Si se
interrumpe, --resume omite familia/prompts completos. No lanzar una segunda
instancia mientras exista el PID indicado en STATUS.json.

La memoria del script es pico asignado de CUDA y RSS al final de cada ejecución;
no incluye muestreo RSS/NVML completo. Por ello no sustituye las medidas de
memoria anteriores ni demuestra mejoras de memoria del sistema. Las pruebas
unitarias se ejecutaron durante el primer prompt: repetir sus tiempos aislados
antes de publicar para evitar interferencia de esa carga auxiliar.

## Resultado verificado (23 octubre)
16/16 casos y repetición aislada de bank/technology completos. FP32 allclose y
salidas exactas (fixed/EOS y tres repeticiones) pasan para cada variante en los
ocho prompts por familia. BF16 no pasa allclose en variantes optimizadas;
no es equivalencia numérica universal. Medianas decode: banco 30.35 -> 41.89
tokens/s con SDPA (1.38x), GPT2 85.42 baseline; last-logit aislado no aporta una
mejora clara en estos prefijos. IC por prompt en summary.json, sin significación.
Generador: recovery_v2/dense_inference_report.py. Evidencia:
results/v2/F0/dense_inference_20261023_analysis/. La gráfica es una ablación
separada; no sustituye memoria ni tiempos históricos de las cuatro familias.
Overleaf actualizado y verificado: 0 errores, 0 warnings; diez tablas y archivos TeX auxiliares preservados. PDF output/pdf/asi_paper_dense_20261023.pdf, revisión visual páginas 10-11. Gate diagnóstico aprobado; no avance de recuperación ni acceso al test sellado.

## Extensión a las cuatro familias (23 octubre)
Tanda nueva: results/v2/F0/four_model_inference_20261023.
Ejecutor: python -m recovery_v2.four_model_inference_audit --output results/v2/F0/four_model_inference_20261023 --resume
32 casos: ASI categorizado, DeepSeek libre, banco y GPT2, ocho prompts cada uno.
Mismas 64 salidas, tres repeticiones, batch1, FP32/BF16 y cuatro threads.
ASI conserva una pareja residente por capa y su routing categorizado;
DeepSeek conserva todos los expertos y su top2 libre. La optimización modifica
atención/proyección, no selecciones, pesos ni normalización del router.
Baseline y last para todos; SDPA+last para las tres familias MLA (GPT2 ya SDPA).
Se registran picos CUDA, RSS muestreado cada 50ms, NVML por proceso si el driver
lo expone (null es desconocido, no cero) y transferencias de expertos en decode.
La preparación y el clasificador quedan fuera de la región temporal.
Antes de la tanda: 229 tests unittest pasaron. pytest no está instalado.
Gate: FP32 allclose y tokens exactos fixed/EOS/tres repeticiones en todos los
prompts. Ante fallo se conserva report y DECISION.md y no se promueve la ruta.
La habilitación de optimizaciones para uso normal y actualización del paper
quedan pendientes de este gate. No se sustituirán tablas manuales ni se publicarán
resultados parciales como conclusión. Históricos conservados.

## Control GPT2 MATH completado
8 prompts, 3 repeticiones, 64 tokens: SDPA automático 85.5775 tokens/s frente
MATH 53.7660; TTFT 11.8508 frente 18.6408 ms. FP32 allclose 8/8, tokens fixed/EOS
idénticos 5/8. No es equivalencia de outputs ni reproducción de MLA. El usuario
autorizó esta comparación numérica separada. summary.json conserva bootstrap
por prompt, memoria y huella de entrada. La figura mantiene barras históricas y
añade marcadores explícitos para ambas ramas de la nueva sesión; tablas intactas.
ASI precision: FP32 original/SDPA coinciden en el caso diagnóstico; BF16 empata
help/develop en el prefijo divergente. No demuestra equivalencia universal.

## Tanda SDPA común autorizada
results/v2/F0/sdpa_four_models_20261023, recovery_v2/sdpa_four_model_audit.py.
Ocho prompts por familia, 64 tokens, tres repeticiones, batch1, cuatro threads,
pesos FP32/autocast BF16, greedy full-prefix sin KV incremental. Ruta principal:
SDPA+proyección de último token en las cuatro familias (GPT2 ya usa SDPA).
Baseline nativo emparejado retenido; no se mezclan tiempos con histórico.
ASI mantiene pareja categorizada residente y DeepSeek selección libre top2.
Mismos controles FP32/BF16 y tokens fixed/EOS/repeticiones. Diferencias se
registran sin interrumpir esta tanda expresamente autorizada. La publicación
queda bloqueada para decisión del usuario si hay cambios. SDPA no implica
el mismo kernel para distintas formas. No se cambia arquitectura ni pesos.
No tests concurrentes con tiempos; suite anterior 229/229. No test sellado.
