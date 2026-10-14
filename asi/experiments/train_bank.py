"""Train the independent dense specialist bank sequentially, with resumable runs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from asi.models.design import validate_recipe, count_parameters
from asi.data.identity import manifest_matches
from asi.experiments.train_restart import atomic_json
from asi.taxonomy import digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan_bank(config_dir, batch_size):
    manifest_path = config_dir / 'expert_pools.json'
    names = list(read(manifest_path)['pools'])
    jobs = []
    for name in names:
        path = config_dir / ('router_' + name + '.json')
        recipe = read(path)
        config = validate_recipe(recipe, names)
        if recipe['family'] != 'router' or recipe['pool'] != name:
            raise ValueError('Recipe is not the expected specialist: ' + name)
        if not manifest_matches(manifest_path, recipe['pool_manifest_sha256']):
            raise ValueError('Pool manifest changed: ' + name)
        counts = count_parameters(recipe['model'])
        if counts['total'] > recipe['parameter_budget']:
            raise ValueError('Specialist exceeds parameter budget: ' + name)
        training = recipe['training']
        if training['total_batch_size'] % (batch_size * training['seq_len']):
            raise ValueError('Batch does not divide the token budget')
        jobs.append({'pool': name, 'recipe_file': path.name, 'recipe': recipe,
                     'parameters': counts['total'], 'steps': training['max_steps'],
                     'tokens': training['max_steps'] * training['total_batch_size']})
    budgets = {j['recipe']['collection_parameter_budget'] for j in jobs}
    if len(budgets) != 1 or sum(j['parameters'] for j in jobs) > next(iter(budgets)):
        raise ValueError('Collection exceeds its common budget')
    return {'pool_order': names, 'jobs': jobs, 'parameters': sum(j['parameters'] for j in jobs),
            'tokens': sum(j['tokens'] for j in jobs), 'batch_size': batch_size,
            'pool_manifest_identity': digest(read(manifest_path))}


def latest_checkpoint(directory):
    candidates = [p for p in directory.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt', p.name)]
    return max(candidates, key=lambda p: int(p.stem.split('_')[1])) if candidates else None


def validate_checkpoint(path, job):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('training_recipe') != job['recipe']:
        raise ValueError('Checkpoint belongs to a different recipe: ' + str(path))
    return int(checkpoint['step']) + 1


def audit_main():
    """Verify a copied bank against frozen recipes without changing its files."""
    import hashlib
    import torch
    from asi.runtime.generation import load_model, validate_pool_identity
    p = argparse.ArgumentParser(description=audit_main.__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--config-dir', type=Path, default=Path('configs/experiment_124m_v1'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new report path to preserve previous evidence')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    saved = read(a.bank.parent / 'plan.json')
    expected = plan_bank(a.config_dir, saved['batch_size'])
    if any(saved.get(k) != v for k, v in expected.items()):
        p.error('Copied plan differs from frozen recipes')
    bank = read(a.bank)
    if set(bank) != set(expected['pool_order']):
        p.error('Bank pool keys differ from frozen recipes')
    report = {'complete': False, 'models': [], 'errors': [],
              'bank': str(a.bank.resolve()), 'parameters_expected': expected['parameters'],
              'limitations': ['Integrity and training completion only; not a quality benchmark.']}
    seen = set()
    torch.set_num_threads(4)
    for job in expected['jobs']:
        row = {'pool': job['pool'], 'expected_steps': job['steps']}
        try:
            value = bank[job['pool']]
            if not value:
                raise ValueError('No registered checkpoint')
            path = (a.bank.parent / value).resolve()
            if path in seen:
                raise ValueError('Checkpoint reused for multiple pools')
            seen.add(path)
            row['checkpoint'] = str(path)
            row['completed_steps'] = validate_checkpoint(path, job)
            if row['completed_steps'] < job['steps']:
                raise ValueError('Training is incomplete')
            with path.open('rb') as stream:
                fingerprint = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    fingerprint.update(chunk)
                row['sha256'] = fingerprint.hexdigest()
            model, metadata = load_model(path)
            validate_pool_identity(model, metadata, expected['pool_order'])
            row['parameters'] = sum(v.numel() for v in model.parameters())
            row['weight_bytes'] = sum(v.numel() * v.element_size() for v in model.parameters())
            row['checkpoint_bytes'] = path.stat().st_size
            row['val_loss'] = metadata.get('val_loss')
            if model.config.n_dense_layers != model.config.n_layer or row['parameters'] != job['parameters']:
                raise ValueError('Architecture or parameter count mismatch')
            if not all(torch.isfinite(v).all().item() for v in model.parameters()):
                raise ValueError('Nonfinite model weights')
            del model
            row['status'] = 'verified'
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            row['status'] = 'failed'
            row['error'] = str(exc)
            report['errors'].append(job['pool'] + ': ' + str(exc))
        report['models'].append(row)
        atomic_json(a.output, report)
        print(json.dumps(row), flush=True)
    report['complete'] = not report['errors']
    atomic_json(a.output, report)
    if not report['complete']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-dir', type=Path, default=Path('configs/experiment_124m_v1'))
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/router_124m_seed1337'))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--save-interval', type=int, default=25)
    p.add_argument('--keep-checkpoints', type=int, default=3)
    p.add_argument('--max-restarts', type=int, default=5)
    p.add_argument('--restart-delay', type=float, default=15)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.batch_size, a.save_interval, a.keep_checkpoints) < 1 or a.max_restarts < 0 or a.restart_delay < 0:
        p.error('Invalid batch/restart/retention limits')
    plan = plan_bank(a.config_dir, a.batch_size)
    plan['seed'] = a.seed
    plan['checkpoint_policy'] = {'save_interval': a.save_interval, 'keep': a.keep_checkpoints}
    if not a.dry_run:
        for job in plan['jobs']:
            if not manifest_matches(a.data_root / 'manifest.json', job['recipe']['source_manifest_sha256']):
                p.error('Dataset does not match the frozen recipes')
    plan_path = a.output / 'plan.json'
    if plan_path.exists():
        if not a.resume and not a.dry_run:
            p.error('Existing bank: pass --resume')
        if read(plan_path) != plan:
            p.error('Existing bank has a different plan; use a new output')
    elif a.output.exists() and any(a.output.iterdir()):
        p.error('Output is not empty and has no validated plan')
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(plan_path, plan)
    bank_path = a.output / 'router_bank.json'
    bank = dict.fromkeys(plan['pool_order'])
    if bank_path.exists():
        bank = read(bank_path)
        if set(bank) != set(plan['pool_order']):
            p.error('Bank pool mismatch')
    else:
        atomic_json(bank_path, bank)
    for job in plan['jobs']:
        directory = a.output / job['pool']
        checkpoint = latest_checkpoint(directory)
        completed_steps = validate_checkpoint(checkpoint, job) if checkpoint else 0
        command = [sys.executable, '-u', '-m', 'asi', 'train',
                   '--recipe', str(a.config_dir / job['recipe_file']),
                   '--pool-manifest', str(a.config_dir / 'expert_pools.json'),
                   '--data-root', str(a.data_root), '--log-dir', str(directory),
                   '--batch-size', str(a.batch_size), '--device', a.device, '--seed', str(a.seed),
                   '--auto-restart', '--save-interval', str(a.save_interval),
                   '--keep-checkpoints', str(a.keep_checkpoints), '--max-restarts', str(a.max_restarts),
                   '--restart-delay', str(a.restart_delay)]
        if checkpoint:
            command += ['--resume', str(checkpoint)]
        print(f"{job['pool']}: {job['parameters']:,} parameters; {completed_steps}/{job['steps']} updates; {job['tokens']:,} tokens", flush=True)
        if a.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        if completed_steps < job['steps']:
            # Existing training supervisor isolates CUDA errors and prunes only this pool's checkpoints.
            subprocess.run(command, check=True)
            checkpoint = latest_checkpoint(directory)
            if checkpoint is None or validate_checkpoint(checkpoint, job) < job['steps']:
                raise RuntimeError('Training exited without its final checkpoint: ' + job['pool'])
        bank[job['pool']] = os.path.relpath(checkpoint, bank_path.parent).replace(os.sep, '/')
        atomic_json(bank_path, bank)
    print('Prepared' if a.dry_run else 'Completed', bank_path, flush=True)


if __name__ == '__main__':
    main()
