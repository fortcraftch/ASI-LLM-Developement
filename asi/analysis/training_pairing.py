"""Evidence of actual initialization and CPU batches for paired training."""
import hashlib
import json
from pathlib import Path


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ('asi/models/domain.py', 'asi/experiments/train.py', 'asi/analysis/training_pairing.py')
    # Source identity is portable across Git LF/CRLF checkouts.
    return {path: hashlib.sha256((root/path).read_text(encoding='utf-8').encode()).hexdigest()
            for path in paths}


def update_tensor(hasher, tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    hasher.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())


def initialization_evidence(model):
    hasher = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        hasher.update(name.encode())
        update_tensor(hasher, parameter)
        count += parameter.numel()
    return {'schema': 1, 'initial_parameters_sha256': hasher.hexdigest(),
            'parameter_count': count, 'implementation_sha256': implementation_hashes()}


def step_hasher(step, pool):
    return hashlib.sha256(json.dumps({'step': step, 'pool': pool}, sort_keys=True).encode())


def compare_runs(paths):
    """Compare observed evidence, without inferring unlogged or untrained steps."""
    runs = []
    for path in map(Path, paths):
        metadata = json.loads((path/'run_metadata.json').read_text(encoding='utf-8'))
        if not metadata.get('pairing_evidence'):
            raise ValueError('Missing pairing evidence: ' + str(path))
        rows = {}
        for line in (path/'log.txt').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if 'pool' not in row:
                continue
            if 'input_sha256' not in row:
                raise ValueError('Untracked training step: ' + str(path))
            step = row['step']
            if step in rows and (rows[step]['input_sha256'], rows[step]['pool']) != (row['input_sha256'], row['pool']):
                raise ValueError('Conflicting replayed step: ' + str(path))
            rows[step] = row
        expected = metadata['settings']['max_steps']
        if any(type(step) is not int or not 0 <= step < expected for step in rows):
            raise ValueError('Step outside frozen schedule')
        runs.append((str(path), metadata, rows))
    if len(runs) < 2:
        raise ValueError('Provide at least two runs')
    _, baseline, base_rows = runs[0]
    base_config = {k: v for k, v in baseline['model'].items() if k != 'routing_mode'}
    comparisons = []
    for path, metadata, rows in runs[1:]:
        common = sorted(set(base_rows) & set(rows))
        checks = {
            'initialization_and_implementation': metadata['pairing_evidence'] == baseline['pairing_evidence'],
            'training_settings': metadata['settings'] == baseline['settings'],
            'architecture_except_routing_policy': {k:v for k,v in metadata['model'].items() if k != 'routing_mode'} == base_config,
            'backend': metadata['moe_backend'] == baseline['moe_backend'],
            'dataset_and_pool_identity': all(metadata['recipe'].get(k) == baseline['recipe'].get(k)
                                            for k in ('source_manifest_sha256', 'pool_manifest_sha256', 'pool_order')),
            'same_observed_batches': bool(common) and all(rows[s]['input_sha256'] == base_rows[s]['input_sha256'] and
                                                         rows[s]['pool'] == base_rows[s]['pool'] for s in common),
        }
        comparisons.append(dict(run=path, checks=checks, shared_steps=len(common),
                                unmatched_baseline_steps=sorted(set(base_rows)-set(rows)),
                                unmatched_run_steps=sorted(set(rows)-set(base_rows))))
    return dict(schema=1, baseline=runs[0][0], comparisons=comparisons,
                observed_pairing_matches=all(all(c['checks'].values()) for c in comparisons),
                logged_schedule_complete=all(set(rows) == set(range(meta['settings']['max_steps'])) for _,meta,rows in runs),
                limitations=['Verifies initialization and logged CPU token batches, not trained model quality.',
                             'Logs do not validate checkpoint contents or unlogged steps; final checkpoints need separate validation.',
                             'Floating point training can differ across devices despite identical initialization and data.'])


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = compare_runs(args.runs)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)
    if not report['observed_pairing_matches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
