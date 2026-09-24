# ASI: investigación sobre expertos, routing y memoria

Estudiamos cómo ejecutar LLM mayores que la memoria disponible manteniendo pocos
expertos en GPU y evitando transferencias en cada token generado. El objetivo no
es acelerar un modelo que ya cabe completo en GPU. ASI es el nombre del prototipo,
no una afirmación de superinteligencia.

Hay dos líneas: **entrenar expertos con pools definidos** y **categorizar expertos
ya entrenados**. Comparten instrumentación y gestión de memoria, pero conservan
sus arquitecturas y checkpoints.

Empieza por el [mapa del TFG y protocolo](docs/PLAN.md), consulta los
[comandos y migración](docs/USAGE.md) y los [resultados interpretados](docs/FINDINGS.md).
Se conserva el [documento original](docs/TFG_original.txt).

```text
asi/                    Código importable; entrada: python -m asi
  data/                 Clasificación, shards y manifiestos
  models/               Modelo por pools y adaptador del original
  runtime/              Clasificador, caché y generación
  analysis/             Rutas y clasificación de expertos
  experiments/          Entrenamiento, auditoría y comparación posthoc
  quantization.py       Exportación offline INT8
  plan.py               Consulta del mapa de investigación
configs/                Configuraciones
examples/               Prompts de diagnóstico
tests/                  Pruebas de corrección
docs/                   Plan, uso y conclusiones
legacy/                 Utilidades anteriores fuera del flujo actual
fineweb_edu_specialized_pipeline/specialized_fineweb/  Dataset existente
results/                Checkpoints e informes locales, excluidos de Git
```

Desde la raíz, con el entorno activado:

```powershell
python -m asi --help
python -m asi plan
python -m asi plan 2.2
python -m unittest discover -s tests -v
```

El refactor conserva pesos, formatos y dataset. Las rutas antiguas de scripts se
sustituyen por los comandos de la guía. La arquitectura externa del checkpoint
de ocho expertos continúa siendo una dependencia explícita.

Existe caché **RAM → GPU** con precarga y carga bajo demanda, y un predictor
estadístico de uso aprendido de sesiones. `python -m asi cache-study` lo compara
con LRU, popularidad y precarga temática con el router original intacto.
`python -m asi decode-study` mide generación incremental con un límite estricto
de dos expertos enrutados por capa. `python -m asi export-store` prepara expertos
para carga desde disco con una caché RAM limitada, sin cargar todos sus pesos al
iniciar la inferencia. Backbone, expertos compartidos y KV siguen ocupando GPU.
La compresión ejecutable dentro de GPU sigue pendiente.
La existencia del predictor no demuestra una mejora: consulta sus resultados.
