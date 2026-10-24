"""OLMoE diagnostic traces: native cache versus frozen category resident sets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil
import torch
import transformers

from recovery_v2.residency_probe import save, sha
from asi.experiments.public_moe import read_lock, input_ids, decode
from asi.experiments.recovery import enable_routers
from asi.models.olmoe_storage import load_disk_backed
from asi.models.hf_olmoe import OlmoeExpertCache
from asi.taxonomy import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding='utf8'))
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'report.json'
    status = a.output / 'STATUS.json'
    report = {'status': 'partial', 'config': cfg, 'config_sha256': sha(a.config), 'records': [],
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'limitations': ['Mechanism diagnostic, not quality or speed equivalence.',
                              'Known topic labels, no classifier latency.',
                              'OS cache not controlled; sequential variants.',
                              'Native cache capacity is global across layer/expert pairs, not reserved per layer.']}
    if path.exists():
        if not a.resume:
            p.error('Use --resume for existing output')
        old = json.loads(path.read_text(encoding='utf8'))
        if old['config_sha256'] != report['config_sha256']:
            p.error('Changed configuration')
        report = old
        if report['status'] == 'complete':
            return
    save(path, report)
    save(status, {'status': 'running', 'stage': 'source_verification'})
    try:
        source = read_lock(Path(cfg['model_dir']))
        study = json.loads(Path(cfg['study']).read_text(encoding='utf8'))
        corpus = json.loads(Path(cfg['corpus']).read_text(encoding='utf8'))
        if not study['completed'] or study['source_identity'] != digest(source) or study['corpus_identity'] != digest(corpus):
            raise ValueError('Calibration identity mismatch')
        if not set(study['train_ids']) <= {r['id'] for r in corpus['examples'] if r['split'] == 'train'}:
            raise ValueError('Calibration contains non-training rows')
        report.update(source=source, seed=cfg['seed'], torch=torch.__version__, transformers=transformers.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                      driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                      input_sha256={x: sha(x) for x in [cfg['study'], cfg['corpus'], str(a.config), __file__]})
        save(path, report)
        torch.manual_seed(cfg['seed'])
        for mode in cfg['modes']:
            pending = [c for c in cfg['cases'] if not any(r['mode'] == mode and r['category'] == c['category'] for r in report['records'])]
            if not pending:
                continue
            save(status, {'status': 'running', 'stage': 'loading', 'mode': mode})
            model, tokenizer, store = load_disk_backed(Path(cfg['model_dir']), int(cfg['expert_ram_gib']*2**30), int(cfg['process_ram_gib']*2**30))
            enable_routers(model)
            model.requires_grad_(False).eval()
            from recovery_v2.timed_olmoe_cache import TimedOlmoeCache
            cache_type = TimedOlmoeCache if cfg.get('profile_transfers') else OlmoeExpertCache
            cache = cache_type(model, capacity=cfg.get('native_cache_experts', 512) if mode == 'native' else int(mode)*model.config.num_hidden_layers, store=store)
            try:
                for case in pending:
                    save(status, {'status': 'running', 'mode': mode, 'category': case['category'], 'generated_tokens': 0})
                    cache.set_context(None)
                    cache.cache.clear()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    peaks = {'rss_bytes': 0, 'nvml_process_bytes': None}
                    stop = threading.Event()
                    try:
                        import pynvml as nv
                        nv.nvmlInit(); handle = nv.nvmlDeviceGetHandleByIndex(0)
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
                    thread = threading.Thread(target=monitor, daemon=True); thread.start()
                    try:
                        before = cache.snapshot(); start = time.perf_counter()
                        if mode != 'native':
                            cache.set_context(study['mappings'][mode][case['category']], mixture='restricted', active=8)
                        torch.cuda.synchronize(); prepared = cache.snapshot()
                        prepare_seconds = time.perf_counter()-start
                        ids = input_ids(tokenizer, case['prompt'], limit=512)
                        routes, events = {}, []
                        def capture(layer, indices, weights):
                            routes[str(layer)] = indices.detach().cpu().tolist()
                        cache.callbacks.append(capture)
                        previous = prepared
                        def step():
                            nonlocal previous
                            current = cache.snapshot()
                            events.append({'phase': 'prefill' if not events else 'decode', 'routes': dict(routes),
                                           'host_to_device_bytes': current.get('host_to_device_bytes', 0)-previous.get('host_to_device_bytes', 0),
                                           'loads': current.get('loads', 0)-previous.get('loads', 0),
                                           'resident_per_layer': current['resident_per_layer'],
                                           'timing_delta': {key: current.get(key, 0)-previous.get(key, 0) for key in ('weight_fetch_seconds', 'h2d_cuda_seconds', 'h2d_wall_seconds', 'profiled_load_seconds')}})
                            previous = current
                            save(status, {'status': 'running', 'mode': mode, 'category': case['category'], 'generated_tokens': len(events)})
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            result = decode(model, ids, cfg['new_tokens'], 'cuda', after_step=step, stop_token_ids=[tokenizer.eos_token_id])
                        cache.callbacks.clear()
                        stop.set(); thread.join()
                        row = dict(result, mode=mode, category=case['category'], prompt=case['prompt'], input_ids=ids.tolist(), events=events,
                                   output=tokenizer.decode(result['token_ids']), preparation_seconds=prepare_seconds,
                                   preparation_h2d_bytes=prepared.get('host_to_device_bytes', 0)-before.get('host_to_device_bytes', 0),
                                   generation_h2d_bytes=sum(e['host_to_device_bytes'] for e in events),
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), **peaks)
                        report['records'].append(row)
                        save(path, report)
                        if mode != 'native' and row['generation_h2d_bytes']:
                            raise ValueError('Unexpected transfer inside fixed residency')
                        print(mode, case['category'], len(result['token_ids']), row['generation_h2d_bytes'], flush=True)
                    finally:
                        stop.set(); thread.join(); cache.callbacks.clear()
            finally:
                cache.close()
                del cache, model, tokenizer, store
                import gc
                gc.collect(); torch.cuda.empty_cache()
        import numpy as np
        rng = np.random.default_rng(cfg['seed'])
        report['prompt_bootstrap'] = {}
        for mode in cfg['modes']:
            values = np.array([r['generation_h2d_bytes'] for r in report['records'] if r['mode'] == mode])
            means = rng.choice(values, (2000, len(values)), replace=True).mean(1)
            report['prompt_bootstrap'][mode] = {'mean_h2d_bytes': float(values.mean()), 'ci95': np.quantile(means, [.025,.975]).tolist(), 'n': len(values)}
        save(a.output/'traces.json', report['records'])
        report.update(status='complete', output_sha256={'traces.json': sha(a.output/'traces.json')})
        save(path, report)
        save(status, {'status': 'complete', 'gate': 'residency_mechanism_only; F0 remains pending'})
    except Exception as exc:
        report.update(status='failed', error=(type(exc).__name__ + ': ' + str(exc)[:1000])); save(path, report)
        save(status, {'status': 'failed', 'error': (type(exc).__name__ + ': ' + str(exc)[:1000])})
        (a.output/'DECISION.md').write_text('Diagnostic stopped: '+(type(exc).__name__ + ': ' + str(exc)[:1000])+'\nNo recovery phase advanced.')
        raise


if __name__ == '__main__':
    main()
