"""Resume the frozen bank evaluation at task boundaries; never overwrite attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from asi.experiments.recovery_protocol import durable_save
from asi.models.original import file_sha256

TASKS = 'hellaswag lambada piqa openbookqa arc_easy arc_challenge winogrande boolq rte wsc triviaqa arithmetic storycloze webquestions copa race mmlu gsm8k truthfulqa'.split()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    status_path = a.output / 'STATUS.json'
    if status_path.exists() and not a.resume:
        p.error('Existing run: pass --resume')
    bank = Path('results/bank_active_matched_seed1337/router_bank.json')
    audit = a.output / 'audit.json'
    manifest = Path('configs/bank_active_matched_v1/expert_pools.json')
    evaluator = Path('../Creating-GPT-2-From-0/eval.py')
    checked = json.loads(audit.read_text())
    if not checked['complete']:
        p.error('Complete bank audit required')
    config = dict(bank=str(bank), audit_sha256=file_sha256(audit),
                  manifest_sha256=file_sha256(manifest), evaluator_sha256=file_sha256(evaluator),
                  runner_sha256=file_sha256(Path('asi/experiments/bank_eval.py')),
                  tasks=TASKS, batch_size=2, batch_tokens=512, precision='bf16', residency='all', seed=1337)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report_path = a.output / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        status='partial', configuration=config, config_sha256=config_hash,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_sha256={r['pool']: r['sha256'] for r in checked['models']}, tasks={},
        limitations=['Descriptive benchmark results; no checkpoint selection.',
                      'Batch 2 differs from historical batch 4; BF16 can affect scores.',
                      'All weights resident for benchmark throughput, not a residency memory experiment.',
                      'Confidence intervals pending request-level analysis; no significance claim.'])
    if report['config_sha256'] != config_hash:
        p.error('Configuration changed; use a fresh output directory')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_DATASETS_CACHE=str(Path('results/hf_datasets_cache').resolve()),
               PYTHONIOENCODING='utf-8')
    # Generation and benchmarks share the GPU sequentially.
    durable_save(status_path, dict(status='running', task='generation', pid=os.getpid()))
    with (a.output / 'generation.log').open('a') as out, (a.output / 'generation.stderr.log').open('a') as err:
        code = subprocess.call([sys.executable, '-u', '-m', 'recovery_v2.bank_generation',
                                '--output', str(a.output), '--resume'], env=env, stdout=out, stderr=err)
    if code:
        durable_save(status_path, dict(status='failed', task='generation', exit_code=code))
        raise SystemExit(code)
    for task in TASKS:
        if task in report['tasks']:
            if file_sha256(Path(report['tasks'][task]['path'])) != report['tasks'][task]['sha256']:
                raise ValueError('Completed evidence changed: ' + task)
            continue
        directory = a.output / task
        directory.mkdir(exist_ok=True)
        index = 1
        while (directory / str(index)).exists():
            index += 1
        attempt = directory / str(index)
        attempt.mkdir()
        target = attempt / 'benchmark.json'
        command = [sys.executable, '-u', '-m', 'asi', 'bank-eval', '--bank', str(bank),
                   '--audit', str(audit), '--pool-manifest', str(manifest), '--output', str(target),
                   '--tasks', task, '--limit', '0', '--eval-batch-size', '2', '--batch-tokens', '512',
                   '--residency', 'all', '--trace-dir', str(attempt / 'traces')]
        durable_save(status_path, dict(status='running', task=task, completed_tasks=len(report['tasks']),
                                      total_tasks=len(TASKS), command=command, pid=os.getpid()))
        durable_save(report_path, report)
        with (attempt / 'stdout.log').open('w') as out, (attempt / 'stderr.log').open('w') as err:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err)
            code = child.wait()
        result = json.loads(target.read_text()) if target.exists() else {}
        if code or not result.get('completed') or task not in result.get('tasks', {}):
            report['status'] = 'failed'
            durable_save(report_path, report)
            durable_save(status_path, dict(status='failed', task=task, exit_code=code, attempt=str(attempt)))
            raise SystemExit(code or 1)
        report['tasks'][task] = dict(path=str(target), sha256=file_sha256(target), result=result['tasks'][task])
        report['status'] = 'partial'
        durable_save(report_path, report)
        print(task, 'complete', flush=True)
    report['status'] = 'complete'
    durable_save(report_path, report)
    durable_save(status_path, dict(status='complete', completed_tasks=len(TASKS), total_tasks=len(TASKS)))


if __name__ == '__main__':
    main()
