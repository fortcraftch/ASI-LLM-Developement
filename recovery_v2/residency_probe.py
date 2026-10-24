"""Small, resumable mechanism diagnostic; no recovery or sealed-test selection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import psutil
import torch
import tiktoken

from asi.models.original import load_original_model, native_moes
from asi.runtime.cache import NativeExpertSessionCache, move_backbone
from asi.experiments.decode_study import run_decode, check_incremental


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    status = a.output / 'STATUS.json'
    report_path = a.output / 'report.json'
    identity = sha(a.config)
    if report_path.exists():
        if not a.resume:
            p.error('Output exists: use --resume')
        report = json.loads(report_path.read_text())
        if report['config_sha256'] != identity:
            p.error('Configuration changed')
        if report['status'] == 'complete':
            return
    else:
        report = {'status': 'partial', 'config_sha256': identity, 'config': cfg,
                  'seed': cfg['seed'], 'records': [],
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'limits': ['Three constructed diagnostic prompts, not a quality benchmark.',
                             'Fixed pairs calibrated on separate short topic texts, not a trained classifier.',
                             'Serial prefill; timings not comparable to batched deployment.',
                             'OS cache uncontrolled. Bootstrap unit is prompt, not token.']}
    save(report_path, report)
    save(status, {'status': 'running', 'stage': 'loading'})
    try:
        torch.manual_seed(cfg['seed'])
        model, metadata = load_original_model(cfg['checkpoint'], max_batch_size=1)
        if 'metadata' in report and report['metadata'] != metadata:
            raise ValueError('Checkpoint or architecture changed')
        report['metadata'] = metadata
        report['torch'] = torch.__version__
        report['cuda'] = torch.version.cuda
        report['gpu'] = torch.cuda.get_device_name()
        import transformers
        report['transformers'] = transformers.__version__
        report['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
        enc = tiktoken.get_encoding('gpt2')
        model.to('cuda')
        report['incremental_parity'] = check_incremental(model, enc.encode('A short check of the model.'), 'cuda')
        if 'mapping' not in report:
            mapping = {'layers': {str(l): {} for l in native_moes(model)}}
            for case in cfg['cases']:
                r = run_decode(model, enc.encode(case['calibration']), enc, 'cuda', 1)
                for layer in mapping['layers']:
                    counts = Counter(e for event in r['events'] for e in np.asarray(event['routes'][layer]).reshape(-1).tolist())
                    mapping['layers'][layer][case['category']] = sorted(counts, key=lambda e: (-counts[e], e))[:2]
            report['mapping'] = mapping
            save(report_path, report)
        model.to('cpu')
        for case in cfg['cases']:
            for mode in ('resident', 'native_cache', 'fixed_pair'):
                if any(r['category'] == case['category'] and r['mode'] == mode for r in report['records']):
                    continue
                save(status, {'status': 'running', 'category': case['category'], 'mode': mode})
                cache = None
                stop = threading.Event()
                peaks = {'rss_bytes': 0, 'nvml_process_bytes': None}
                nv = None
                try:
                    import pynvml as nv
                    nv.nvmlInit()
                    handle = nv.nvmlDeviceGetHandleByIndex(0)
                except Exception:
                    nv = None
                def monitor():
                    while not stop.is_set():
                        peaks['rss_bytes'] = max(peaks['rss_bytes'], psutil.Process().memory_info().rss)
                        if nv:
                            try:
                                for proc in nv.nvmlDeviceGetComputeRunningProcesses(handle):
                                    if proc.pid == os.getpid() and proc.usedGpuMemory is not None and proc.usedGpuMemory < 2**60:
                                        peaks['nvml_process_bytes'] = max(peaks['nvml_process_bytes'] or 0, proc.usedGpuMemory)
                            except Exception:
                                pass
                        stop.wait(.05)
                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                try:
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    if mode == 'resident':
                        model.to('cuda')
                    else:
                        move_backbone(model, 'cuda')
                        cache = NativeExpertSessionCache(model, report['mapping'], 'cuda', 2 * len(native_moes(model)),
                            max_experts_per_layer=2, policy='fixed' if mode == 'fixed_pair' else 'prefetch')
                        if mode == 'fixed_pair':
                            cache.set_context_labels([case['category']])
                    torch.cuda.synchronize()
                    preparation = time.perf_counter() - start
                    before = cache.snapshot() if cache else {}
                    r = run_decode(model, enc.encode(case['prompt']), enc, 'cuda', cfg['new_tokens'], cache)
                    r.pop('probes')
                    r.update(category=case['category'], mode=mode, prompt=case['prompt'], preparation_seconds=preparation,
                             preparation_cache=before, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    r['h2d_generation_bytes'] = sum(e['cache_delta']['host_to_device_bytes'] for e in r['events'])
                    r['pair_changes'] = sum(sorted(np.asarray(x['routes'][l]).reshape(-1).tolist()) != sorted(np.asarray(y['routes'][l]).reshape(-1).tolist())
                        for x, y in zip(r['events'], r['events'][1:]) for l in x['routes'])
                    if mode == 'native_cache':
                        base = next(x for x in report['records'] if x['mode'] == 'resident' and x['category'] == case['category'])
                        if r['tokens'] != base['tokens'] or [e['routes'] for e in r['events']] != [e['routes'] for e in base['events']]:
                            raise ValueError('Exact-cache parity failed')
                    if mode == 'fixed_pair' and (r['h2d_generation_bytes'] or r['pair_changes']):
                        raise ValueError('Fixed residency gate failed')
                    stop.set(); thread.join()
                    r.update(peaks)
                    r['nvml_note'] = 'Unavailable on this driver' if peaks['nvml_process_bytes'] is None else 'Sampled every 50ms'
                    report['records'].append(r)
                    save(report_path, report)
                    print(case['category'], mode, r['pair_changes'], r['h2d_generation_bytes'], flush=True)
                finally:
                    stop.set(); thread.join()
                    if cache:
                        cache.close()
                    model.to('cpu')
                    torch.cuda.empty_cache()
        rng = np.random.default_rng(cfg['seed'])
        report['prompt_bootstrap'] = {}
        for mode in ('native_cache', 'fixed_pair'):
            values = np.array([r['h2d_generation_bytes'] for r in report['records'] if r['mode'] == mode])
            draws = rng.choice(values, (2000, len(values)), replace=True).mean(1)
            report['prompt_bootstrap'][mode] = {'mean_h2d_bytes': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(), 'n': len(values)}
        save(a.output / 'traces.json', report['records'])
        report['output_sha256'] = {'traces.json': sha(a.output / 'traces.json')}
        report['status'] = 'complete'
        save(report_path, report)
        save(status, {'status': 'complete', 'gate': 'mechanism_only_passed; F0 gate unchanged'})
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        save(report_path, report)
        save(status, {'status': 'failed', 'error': repr(exc)})
        (a.output / 'DECISION.md').write_text('Mechanism diagnostic stopped: ' + repr(exc) + '\nNo recovery phase advanced.')
        raise


if __name__ == '__main__':
    main()
