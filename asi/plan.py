"""Display the research map without importing torch or loading a model."""
import argparse
import json
from asi import ROOT


def load_plan():
    return json.loads((ROOT / 'docs/plan.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', nargs='?', help='Por ejemplo: 1.1 o 2.2')
    args = parser.parse_args()
    steps = load_plan()
    if args.step and args.step not in {step['id'] for step in steps}:
        parser.error('Paso desconocido: ' + args.step)
    for step in steps:
        if args.step and args.step != step['id']:
            continue
        print(f"{step['id']} — {step['title']} [{step['status']}]")
        print('  Código: ' + (', '.join(step['files']) or 'Sin implementación'))
        print('  Disponible: ' + step['available'])
        print('  Pendiente: ' + step['pending'])
