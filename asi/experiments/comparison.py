"""Prepare a frozen three-model experiment; preparation never loads model weights."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from asi import ROOT


ROLES = ('original', 'classified', 'domain')
METRICS = {
    'quality': ['nll', 'perplexity', 'next_token_accuracy', 'tokens'],
    'transfer': ['hits', 'misses', 'evictions', 'host_to_device_bytes', 'prefetch_loads'],
    'timing': ['classification_seconds', 'prepare_seconds', 'prefill_seconds', 'decode_seconds'],
    'memory': ['expert_weight_bytes_by_device', 'backbone_and_shared_weight_bytes',
               'buffer_bytes', 'cuda_allocated_bytes', 'cuda_reserved_bytes', 'process_rss_bytes'],
}
PROMPTS = [
    ('expert_00_programming', ['Explain Python generators with a short example.', 'How do you test a generator?', 'Explain the difference between yield and return.']),
    ('expert_01_systems', ['Explain database indexes.', 'How does a Linux process use virtual memory?', 'Explain a network timeout and retry.']),
    ('expert_02_ai', ['Explain attention in a transformer.', 'What is overfitting?', 'Explain the purpose of a validation dataset.']),
    ('expert_03_math', ['Derive the derivative of x squared.', 'Explain conditional probability.', 'Solve the equation 2x + 3 = 11.']),
    ('expert_04_physical_sciences', ['Explain conservation of energy.', 'What determines the acceleration of a falling object?', 'Explain electric potential.']),
    ('expert_05_life_sciences', ['Explain DNA transcription.', 'What do ribosomes do?', 'Explain natural selection.']),
    ('expert_06_engineering', ['Explain feedback control.', 'How does a bridge distribute a load?', 'Explain the role of a sensor in a control loop.']),
    ('expert_07_humanities_social', ['Explain opportunity cost.', 'Compare two approaches to historical evidence.', 'Explain the difference between correlation and causation in social research.']),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(path):
    return (ROOT / path).resolve() if path else None


def validate_config(config):
    if config.get('schema') != 1 or set(config.get('models', {})) != set(ROLES):
        raise ValueError('Expected schema 1 and original/classified/domain model slots')
    for model in config['models'].values():
        if model['adapter'] not in ('native', 'domain'):
            raise ValueError('Unknown model adapter')
    if config['quality']['split'] not in ('val', 'test'):
        raise ValueError('Evaluation must not use train shards')
    values = (config['quality']['lengths'] + config['capacities_per_layer'] +
              config['generation']['new_tokens'] + [config['repeats'],
              config['quality']['windows_per_class'], config['generation']['max_prompt_tokens']])
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('Lengths, capacities, repeats and sample counts must be positive integers')
    for sequence in (config['quality']['lengths'], config['capacities_per_layer'],
                     config['generation']['new_tokens'], config['cache_starts']):
        if not sequence or len(sequence) != len(set(sequence)):
            raise ValueError('Experimental axes must be nonempty and contain no duplicate values')
    if not config['cache_starts'] or set(config['cache_starts']) - {'cold', 'warm'}:
        raise ValueError('Cache starts must be cold/warm')


def session_workloads(names):
    """Authored stress inputs, not a scored knowledge benchmark or training data."""
    topics = [(name, prompts) for name, prompts in PROMPTS if name in names]
    if len(topics) < 2:
        raise ValueError('Conversation stress tests require at least two known domains')
    result = []
    for i, (a, prompts) in enumerate(topics):
        b, other = topics[(i + 1) % len(topics)]
        patterns = {'stable': [a]*6, 'switch': [a]*3+[b]*3,
                    'alternating': [a,b]*3, 'return': [a,a,b,b,a,a],
                    'mixed': [a,b]*3}
        lookup = {a: prompts, b: other}
        for scenario, labels in patterns.items():
            turns = []
            for turn, label in enumerate(labels):
                prompt = lookup[label][turn % 3]
                if scenario == 'mixed':
                    second = b if label == a else a
                    prompt += ' Also address this separate topic: ' + lookup[second][turn % 3]
                turns.append({'prompt': prompt, 'label': label,
                              'label_source': 'authored_primary_topic',
                              'all_labels': [label, b if label == a else a] if scenario == 'mixed' else [label]})
            result.append({'id': f'{scenario}_{i:02}', 'split': 'test', 'scenario': scenario, 'turns': turns})
    return result


def variants(role):
    base = [{'policy': 'resident', 'label_source': 'none', 'candidates': None}]
    if role == 'original':
        return base + [{'policy': 'exact', 'label_source': 'none', 'candidates': None}]
    for source in ('oracle', 'predicted'):
        base.append({'policy': 'fixed', 'label_source': source, 'candidates': 2})
        for count in (2, 4, 8):
            base.append({'policy': 'restrict', 'label_source': source, 'candidates': count})
    # Exact routing separates cache overhead from quality changes.
    base.append({'policy': 'exact', 'label_source': 'none', 'candidates': None})
    if role == 'classified':
        base.append({'policy': 'fixed', 'label_source': 'global', 'candidates': 2})
    return base


def build_jobs(config):
    jobs = []
    for role in ROLES:
        for variant in variants(role):
            capacities = [None] if variant['policy'] == 'resident' else config['capacities_per_layer']
            for capacity in capacities:
                # Fixed N/N has no benefit from spare slots. Keep the strict N=2 arm.
                if variant['policy'] == 'fixed' and capacity != 2:
                    continue
                for phase in ('quality', 'generation'):
                    lengths = config['quality']['lengths'] if phase == 'quality' else config['generation']['new_tokens']
                    # Quality repeats do not create new statistical observations.
                    repeats = range(1) if phase == 'quality' else range(config['repeats'])
                    starts = ['cold'] if phase == 'quality' else config['cache_starts']
                    for length, repeat, start in itertools.product(lengths, repeats, starts):
                        jobs.append({'role': role, **variant, 'capacity': capacity, 'phase': phase,
                                     'length': length, 'repeat': repeat, 'cache_start': start})
    random.Random(config['seed']).shuffle(jobs)
    for i, job in enumerate(jobs):
        job['id'] = f'job_{i:05}'
    return jobs


def prepare(config, output):
    validate_config(config)
    # Imported only to inspect token shards. No model/classifier is instantiated.
    from asi.experiments.posthoc import sample_windows
    import numpy as np
    manifest_path = resolve(config['pool_manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = resolve(config['data_root'])
    windows, coverage = [], {}
    for length in config['quality']['lengths']:
        selected, counts = sample_windows(root, manifest, config['quality']['split'], length,
                                           config['quality']['windows_per_class'], config['seed'])
        coverage[str(length)] = counts
        for window in selected:
            arr = np.load(window['shard'], mmap_mode='r')
            window['tokens_sha256'] = hashlib.sha256(arr[window['start']:window['start']+window['length']].tobytes()).hexdigest()
            arr._mmap.close()
            window['id'] = f'w_{len(windows):06}'
            windows.append(window)
    if not windows:
        raise ValueError('No held-out windows available; no empty experiment will be prepared')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payloads = {'config.json': config, 'windows.json': windows,
                'sessions.json': session_workloads(manifest['pools']), 'jobs.json': build_jobs(config)}
    for name, value in payloads.items():
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    lock = {'schema': 1, 'state': 'prepared_not_executed', 'pool_manifest_sha256': digest(manifest_path),
            'files': {name: digest(output/name) for name in payloads}, 'coverage': coverage,
            'metrics': METRICS, 'calibration_split': 'train',
            'warnings': ['Missing domains are not replaced by train data.',
                         'Different context lengths can overlap: analyze lengths separately.',
                         'Windows are not independent documents; no document-level confidence claims.',
                         'Authored sessions measure swapping, not general task competence.']}
    (output/'manifest.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    return lock


def verify_bundle(directory):
    directory = Path(directory)
    lock = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in lock['files'].items():
        if digest(directory/name) != expected:
            raise ValueError('Frozen experiment changed: ' + name + '; prepare a new bundle')
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    validate_config(config)
    if digest(resolve(config['pool_manifest'])) != lock['pool_manifest_sha256']:
        raise ValueError('Pool manifest changed after preparation')
    return config, lock


def bind_models(directory, new_config, output):
    """Register finished checkpoints without resampling or changing the protocol."""
    config, lock = verify_bundle(directory)
    validate_config(new_config)
    if {k:v for k,v in config.items() if k != 'models'} != {k:v for k,v in new_config.items() if k != 'models'}:
        raise ValueError('Binding may change model slots only; prepare a new protocol for other changes')
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in lock['files']:
        if name == 'config.json':
            (output/name).write_text(json.dumps(new_config, indent=2, ensure_ascii=False), encoding='utf-8')
        else:
            (output/name).write_bytes((directory/name).read_bytes())
    lock['files'] = {name: digest(output/name) for name in lock['files']}
    lock['parent_manifest_sha256'] = digest(directory/'manifest.json')
    lock['state'] = 'models_registered_not_executed'
    (output/'manifest.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')


def readiness(config):
    """No torch.load, CUDA calls, inference, calibration or implicit execution."""
    blockers, warnings = [], []
    for role, spec in config['models'].items():
        required = ['checkpoint'] + (['architecture'] if spec['adapter'] == 'native' else [])
        if role == 'classified':
            required += ['mapping']
        for field in required:
            path = resolve(spec.get(field))
            if path is None or not path.is_file():
                blockers.append(f'{role}.{field}: missing')
        if not spec.get('training'):
            warnings.append(f'{role}: training budget/provenance unknown; no causal training comparison')
    a, b = (config['models'][role].get('checkpoint') for role in ('original', 'classified'))
    if a and b and resolve(a) != resolve(b):
        warnings.append('Original and classified use different checkpoint paths; verify identical weights or report training confounding')
    return {'ready_paths': not blockers, 'blockers': blockers, 'warnings': warnings,
            'note': 'Paths only. Runtime validates weights, mapping provenance and model capacity before scoring.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--config', type=Path, default=ROOT/'configs/comparison_suite.json')
    prep.add_argument('--output', type=Path, required=True)
    bind = sub.add_parser('bind-models', help='Register future checkpoints without changing frozen inputs')
    bind.add_argument('--suite', type=Path, required=True)
    bind.add_argument('--config', type=Path, required=True)
    bind.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('check')
    check.add_argument('--suite', type=Path, required=True)
    report = sub.add_parser('summarize', help='Offline summaries of completed jobs; never executes models')
    report.add_argument('--suite', type=Path, required=True)
    report.add_argument('--results', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run-job', help='Explicit future execution of one isolated job')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--job', required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        config = json.loads(args.config.read_text(encoding='utf-8'))
        lock = prepare(config, args.output)
        print(json.dumps({'state': lock['state'], 'coverage': lock['coverage'], **readiness(config)}, indent=2))
    elif args.action == 'bind-models':
        config = json.loads(args.config.read_text(encoding='utf-8'))
        bind_models(args.suite, config, args.output)
        print(json.dumps(readiness(config), indent=2))
    else:
        config, _ = verify_bundle(args.suite)
        if args.action == 'check':
            print(json.dumps(readiness(config), indent=2))
        elif args.action == 'summarize':
            from asi.experiments.comparison_report import summarize
            summarize(args.suite, args.results, args.output)
        else:
            from asi.experiments.comparison_run import run_job
            run_job(args.suite, args.job, args.output)


if __name__ == '__main__':
    main()
