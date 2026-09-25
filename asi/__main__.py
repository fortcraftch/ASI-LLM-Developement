"""Single entry point. Heavy dependencies are loaded only for the chosen command."""
import argparse
import importlib
import sys


COMMANDS = {
    'graphs': ('asi.analysis.plots', 'main', 'Explicar y graficar pruebas ya ejecutadas, sin cargar modelos'),
    'comparison': ('asi.experiments.comparison', 'main', 'Preparar y ejecutar por separado la comparación de tres modelos'),
    'data': {
        'prepare': ('asi.data.prepare', 'main', 'Clasificar documentos y crear shards'),
        'pools': ('asi.data.pools', 'main', 'Construir el manifiesto de pools'),
        'inspect': ('asi.data.inspect', 'main', 'Consultar la distribución del dataset'),
    },
    'train': ('asi.experiments.train', 'main', 'Entrenar expertos con pools definidos'),
    'audit': ('asi.experiments.audit', 'main', 'Auditar rutas y memoria del modelo por pools'),
    'cache-study': ('asi.experiments.cache_study', 'main', 'Entrenar predictor y comparar cachés en sesiones separadas'),
    'decode-study': ('asi.experiments.decode_study', 'main', 'Medir swapping por token con residencia limitada por capa'),
    'fixed-study': ('asi.experiments.fixed_study', 'main', 'Medir degradación con expertos fijos N sobre N sin E'),
    'export-store': ('asi.runtime.storage', 'main', 'Exportar backbone y expertos para carga SSD/RAM/GPU'),
    'posthoc': {
        'calibrate': ('asi.experiments.posthoc', 'main', 'Clasificar y comparar expertos ya entrenados'),
    },
    'chat': {
        'domain': ('asi.runtime.generation', 'domain_main', 'Conversar con el modelo entrenado por pools'),
        'existing': ('asi.runtime.generation', 'existing_main', 'Conversar con el modelo original categorizado'),
    },
    'quantize': ('asi.quantization', 'main', 'Exportar expertos INT8; no activa inferencia comprimida'),
    'plan': ('asi.plan', 'main', 'Consultar la correspondencia con los pasos del TFG'),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = COMMANDS
    prefix = 'python -m asi'
    while isinstance(commands, dict):
        parser = argparse.ArgumentParser(prog=prefix)
        parser.add_argument('command', choices=commands, help=' | '.join(
            f'{key}: {value[2] if isinstance(value, tuple) else "subcomandos"}'
            for key, value in commands.items()))
        # Parse only the command so that leaf parsers own all their flags.
        selected = parser.parse_args(args[:1]).command
        args = args[1:]
        prefix += ' ' + selected
        commands = commands[selected]
    module, function, _ = commands
    previous = sys.argv
    try:
        sys.argv = [prefix, *args]
        getattr(importlib.import_module(module), function)()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
