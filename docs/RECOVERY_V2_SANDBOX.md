# Comprobación del sandbox de F0 en homelab

## Ejecución verificada: 18/10/2026

Conexión SSH operativa; paquete aislado en `/home/marc/sandbox_f0_20261018`.
Informe recuperado en `results/v2/F0/sandbox_homelab_20261018/report.json`:
las tres comprobaciones pasan (escritura fuera del temporal bloqueada, conexión
de red bloqueada, bucle iniciado y terminado por timeout, contenedores retirados).
Imagen fijada: `python@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634`.
No se ejecutaron soluciones de BigCodeBench ni se utilizó GPU. Falta preparar
dependencias y repetir este control sobre la imagen definitiva de referencias.

Validación inicial de referencias iniciada en `/home/marc/reference_f0_20261018`:
253 tareas cuyos imports estáticos pertenecen a la biblioteca estándar; imagen
Python anterior, un contenedor por tarea, 30 segundos máximo, sin GPU. Proceso
observado PID 3947. Consultar `results/STATUS.json` y `results/report.json` remotos
antes de reanudar. Este subconjunto sirve para validar infraestructura y referencias,
no representa la suite completa ni autoriza eliminar fallos silenciosamente.
Primeras 31 tareas: 29 pasan; hay una referencia que espera DNS externo y otra
con fallo de contenido de archivo. Conservar y clasificar los fallos; no habilitar
red para hacer pasar pruebas que la requieren.

### Corrección del cargador (18/10)

La revisión del piloto detectó que `exec` en un diccionario no creaba un módulo
importable. Afecta como mínimo a BigCodeBench/205 y /779; no interpretar esos
errores como referencias incorrectas. Cargador local corregido: archivo temporal,
registro en `sys.modules` e importación convencional, con multiprocessing spawn.
Prueba sintética de importación y Pool spawn aprobada fuera del sandbox de Windows
(este bloquea las tuberías). El runner remoto activo conserva su versión inicial;
sus resultados son diagnósticos y requieren revalidación del cargador corregido.

El equipo local no tiene Docker/WSL disponibles. Este control no entrena modelos
ni necesita GPU. Tampoco ejecuta respuestas del dataset: usa tres pruebas controladas.

Desde la raíz del proyecto y con el entorno Python activado:

```bash
docker pull python:3.10-slim
IMAGE=$(docker image inspect python:3.10-slim --format '{{index .RepoDigests 0}}')
python -m recovery_v2.eval.sandbox_probe \
  --image "$IMAGE" \
  --output results/v2/F0/sandbox_homelab_attempt1
```

El tag solo sirve para obtener la imagen; el comprobador recibe y registra su
digest inmutable. No montar directorios del host ni pasar credenciales al contenedor.
El programa impone usuario sin privilegios, red deshabilitada, raíz de solo lectura,
tmpfs temporal, límites de memoria/CPU/procesos y eliminación del contenedor.

Devolver la carpeta completa `results/v2/F0/sandbox_homelab_attempt1`. Debe contener
`report.json` con `passed: true` y las tres comprobaciones aprobadas. Si falla,
conservar la carpeta y usar otra para el siguiente intento. No dar G0 por aprobado
solo porque Docker arranca o este control termina: falta ejecutar las referencias
con una imagen que contenga las dependencias reales de BigCodeBench y repetir
el control para esa imagen. Este control es una prueba de aislamiento, no una
garantía absoluta frente a vulnerabilidades del runtime.
