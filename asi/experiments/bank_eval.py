"""Evaluate the independent dense bank using the shared GPT-2 benchmark protocol."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import tiktoken
import torch

from asi.experiments.checkpoint_eval import load_benchmarks
from asi.experiments.benchmark_batch import evaluate
from asi.experiments.recovery_protocol import durable_save
from asi.models.original import file_sha256
from asi.runtime.ensemble import IndependentModelRouter, BankResidency
from asi.runtime.generation import inference_context
from asi.runtime.routing import DomainSessionRouter
from asi.runtime.cache import module_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--pool-manifest', type=Path, default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--gpt2-evaluator', type=Path, default=Path('../Creating-GPT-2-From-0/eval.py'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', default='hellaswag,lambada,piqa,arc_easy,boolq,arithmetic')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--batch-tokens', type=int, default=2048)
    p.add_argument('--residency', choices=['all', 'selected'], default='all')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--trace-dir', type=Path, help='New directory for per-request scores and generated token IDs')
    p.add_argument('--declared-routes', type=Path, help='Input-text SHA256 to externally annotated pool; all inputs must be covered')
    a = p.parse_args()
    if a.precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if a.output.exists(): p.error('Use a new output path')
    if a.trace_dir and a.trace_dir.exists(): p.error('Use a new trace directory')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.trace_dir: a.trace_dir.mkdir(parents=True)
    if a.limit < 0 or min(a.eval_batch_size, a.batch_tokens) < 1: p.error('Invalid sizes')
    audit = json.loads(a.audit.read_text(encoding='utf8'))
    manifest = json.loads(a.pool_manifest.read_text(encoding='utf8'))
    names = list(manifest['pools'])
    bank = json.loads(a.bank.read_text(encoding='utf8'))
    checked = {r['pool']: r for r in audit['models']}
    if not audit['complete'] or set(checked) != set(names): p.error('A complete bank audit is required')
    paths = {k: a.bank.parent / v for k, v in bank.items()}
    # The active-capacity-matched bank is larger than the historical 124M bank.
    # Keep the per-specialist guard, using the independently audited inventory.
    audited_budget = sum(checked[name]['parameters'] for name in names)
    if audited_budget != audit['parameters_expected']:
        p.error('Audited collection parameter count is inconsistent')
    manager = IndependentModelRouter(paths, names, 'cpu', total_budget=audited_budget)
    models = {}
    for name in names:
        if file_sha256(paths[name]) != checked[name]['sha256']: p.error('Checkpoint changed since audit: ' + name)
        model, _ = manager.select(name)
        if a.eval_batch_size > model.config.max_batch_size: p.error('Batch exceeds checkpoint cache capacity')
        models[name] = model
        # Retain each independently validated model in RAM before applying residency.
        manager.model = None
        manager.pool = None
    residency = BankResidency(models, a.device, a.residency)
    enc = tiktoken.get_encoding('gpt2')
    declared = json.loads(a.declared_routes.read_text(encoding='utf8')) if a.declared_routes else None
    router = None if declared is not None else DomainSessionRouter(names, max_pools=1, session_inertia=0, pool_manifest=manifest, device=a.device)
    bench = load_benchmarks(a.gpt2_evaluator)
    tasks = list(bench.TASKS) if a.tasks == 'all' else a.tasks.split(',')
    if not tasks or set(tasks) - set(bench.TASKS): p.error('Unknown tasks')
    routes = {}
    class Adapter(bench.Adapter):
        classifier_seconds = 0.
        def inference_context(self):
            return inference_context(a.device) if a.precision == 'bf16' else nullcontext()
        def prepare_routes(self, contexts):
            texts = list(dict.fromkeys(enc.decode(list(ids)) for ids in contexts))
            pending = [text for text in texts if text not in routes]
            if declared is not None:
                for text in pending:
                    key = hashlib.sha256(text.encode('utf8')).hexdigest()
                    if key not in declared or declared[key] not in names:
                        raise ValueError('Missing or invalid declared category for input ' + key)
                    routes[text] = declared[key]
            else:
                start = time.perf_counter()
                for offset in range(0, len(pending), 8):
                    chunk = pending[offset:offset+8]
                    for text, route in zip(chunk, router.route_many(chunk, 8)):
                        routes[text] = route.ranked_pools[0]
                if a.device.startswith('cuda'): torch.cuda.synchronize()
                self.classifier_seconds += time.perf_counter() - start
        def route_key(self, ids):
            self.prepare_routes([ids])
            return (routes[enc.decode(list(ids))],)
        def activate(self, route):
            self.model = residency.activate(route[0])
            self.config = self.model.config
        @torch.no_grad()
        def logits(self, ids):
            with self.inference_context(): return super().logits(ids).float()
    adapter = Adapter(models[names[0]], models[names[0]].config, a.device, 'independent_bank', enc)
    trace_files = {}
    if a.trace_dir:
        def record_requests(task, trace):
            path = a.trace_dir / (task + '.json')
            durable_save(path, trace)
            trace_files[task] = {'path': str(path.resolve()), 'sha256': file_sha256(path)}
        adapter.record_requests = record_requests
    datasets = {}
    original_ds = bench.ds
    def ds(path, name=None, split='validation'):
        key = str((path, name, split))
        if key not in datasets: datasets[key] = original_ds(path, name, split)
        return datasets[key]
    bench.ds = ds
    report = {'kind': 'independent_bank_benchmark', 'completed': False, 'tasks': {},
        'runner_sha256': file_sha256(Path(__file__)),
        'implementation_sha256': {str(path.relative_to(Path(__file__).resolve().parents[2])): file_sha256(path)
            for path in (Path(__file__).resolve().parent / 'benchmark_batch.py',
                         Path(__file__).resolve().parents[1] / 'runtime/ensemble.py',
                         Path(__file__).resolve().parents[1] / 'runtime/routing.py',
                         Path(__file__).resolve().parents[1] / 'models/domain.py')},
        'checkpoint_sha256': {k: checked[k]['sha256'] for k in names},
        'audit_sha256': file_sha256(a.audit), 'evaluator_sha256': file_sha256(a.gpt2_evaluator),
        'pool_manifest_sha256': file_sha256(a.pool_manifest),
        'declared_routes_sha256': file_sha256(a.declared_routes) if a.declared_routes else None,
        'routing': 'declared' if declared is not None else 'predicted_from_input_only',
        'limit': a.limit or None, 'parameters': sum(checked[k]['parameters'] for k in names),
        'runtime': {'device': a.device, 'torch': torch.__version__, 'batch_size': a.eval_batch_size,
                    'residency': a.residency,
                    'seed': 1337, 'gpu': torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
                    'batch_tokens': a.batch_tokens, 'precision': a.precision,
                    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
        'memory': {k: module_memory(v) for k, v in models.items()},
        'classifier_memory': module_memory(router.classifier) if router else None,
        'limitations': ['One model selected per input; residency policy is separate from activation.',
                       'Requests grouped by category; switch counts are not an online conversation workload.',
                       'Transfer counters include logical model weight bytes only, not buffers or measured bus traffic.',
                       'No oracle category is inferred from answers or task names. Declared categories require external annotations.',
                       'Subset results are exploratory; benchmark elapsed time includes data loading and classification.']}
    def save():
        report['request_traces'] = trace_files
        report['datasets'] = {k: {'fingerprint': v._fingerprint, 'rows': len(v)} for k, v in datasets.items()}
        report['input_routes'] = routes
        durable_save(a.output, report)
    torch.manual_seed(1337)
    for name in models:
        adapter.activate((name,))
        adapter.logits([1, 2, 3])
    if a.device.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    report['startup_residency'] = residency.snapshot()
    save()
    for task in tasks:
        start = time.perf_counter()
        before = adapter.classifier_seconds
        before_residency = residency.snapshot()
        try:
            result = evaluate(bench, task, adapter, a.limit or None, a.eval_batch_size, a.batch_tokens)
        except Exception as exc:
            report['failure'] = {'task': task, 'error': str(exc), 'type': type(exc).__name__}
            save()
            raise
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        result['seconds'] = time.perf_counter() - start
        result['classifier_seconds'] = adapter.classifier_seconds - before
        result['residency_before'] = before_residency
        result['residency_after'] = residency.snapshot()
        report['tasks'][task] = result
        save()
        print(task, result, flush=True)
    if a.device.startswith('cuda'): report['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
    report['memory'] = {k: module_memory(v) for k, v in models.items()}
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
