"""Run the existing test baseline without launching a model experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/v2/F0'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if args.resume and previous.get('baseline_returncode') == 0:
            print('Baseline already passed; G0 still pending.')
            return
        raise SystemExit('Existing attempt preserved. Inspect it before scheduling a new attempt.')
    def command(*argv):
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    versions = {}
    for package in ('torch', 'transformers', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    inputs = {str(p.relative_to(root)): digest(p)
              for directory in ('asi', 'tests') for p in sorted((root / directory).rglob('*.py'))}
    report = {'status': 'partial', 'phase': 'F0', 'gate': 'pending',
              'started': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'versions': versions,
              'git': command('git', 'rev-parse', 'HEAD'),
              'worktree': command('git', 'status', '--short'), 'inputs_sha256': inputs,
              'protocol_sha256': digest(root / 'PROTOCOLO_RECUPERACION_V2.md'),
              'metrics': None, 'confidence_intervals': None,
              'note': 'Software baseline only; no quality or hardware benchmark yet.'}
    save(report_path, report)
    save(out / 'STATUS.json', {'status': 'partial', 'stage': 'existing_unit_tests', 'pid': os.getpid()})
    with (out / 'baseline.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT)
    report['baseline_returncode'] = result.returncode
    report['outputs_sha256'] = {'baseline.log': digest(out / 'baseline.log')}
    report['finished'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'partial' if result.returncode == 0 else 'failed'
    save(report_path, report)
    save(out / 'STATUS.json', {'status': report['status'], 'stage': 'baseline_finished',
                             'baseline_returncode': result.returncode, 'gate': 'pending'})
    if result.returncode:
        (out / 'DECISION.md').write_text('Baseline failed. Inspect baseline.log; do not advance beyond F0.\n')


if __name__ == '__main__':
    main()
