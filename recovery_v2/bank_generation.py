"""Matched full-prefix generation for the audited dense bank (one model resident)."""
import argparse
import gc
import json
import os
from pathlib import Path
import threading
import time

import psutil
import tiktoken
import torch

from asi.experiments.local_generation import PROBES, timed_generate
from asi.experiments.recovery_protocol import durable_save
from asi.models.original import file_sha256
from asi.runtime.cache import module_memory
from asi.runtime.ensemble import IndependentModelRouter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'generation.json'
    if path.exists() and not a.resume:
        p.error('Existing run: pass --resume')
    audit_path = a.output / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['complete']:
        p.error('Audit incomplete')
    paths = {r['pool']: Path(r['checkpoint']) for r in audit['models']}
    settings = dict(tokens=64, repeats=3, seed=1337, precision='FP32 weights/BF16 autocast',
                    audit_sha256=file_sha256(audit_path), runner_sha256=file_sha256(Path(__file__)),
                    generation_sha256=file_sha256(Path('asi/experiments/local_generation.py')))
    report = json.loads(path.read_text()) if path.exists() else dict(
        status='partial', settings=settings, pools={},
        hardware={'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda},
        limitations=['Historical diagnostic prompts, never used for selection.',
                      'Declared categories exclude external classifier cost.',
                      'One dense model resident; other models remain on disk in this measurement.',
                      'Full-prefix generation, no incremental KV cache; fixed 64-token timing ignores EOS.',
                      'Preparation excluded from decode rate and TTFT; logical H2D counts weights only.',
                      'Three timing repeats, no independent training seeds or significance claim.'])
    if report['settings'] != settings:
        p.error('Changed protocol: use fresh output')
    manager = IndependentModelRouter(paths, list(paths), 'cuda', total_budget=audit['parameters_expected'])
    enc = tiktoken.get_encoding('gpt2')
    torch.set_num_threads(4)
    for probe in PROBES:
        pool = probe['pool']
        if pool in report['pools']:
            continue
        row = next(r for r in audit['models'] if r['pool'] == pool)
        if file_sha256(paths[pool]) != row['sha256']:
            raise ValueError('Checkpoint changed')
        durable_save(a.output / 'generation_STATUS.json', dict(status='running', pool=pool, completed=len(report['pools'])))
        torch.manual_seed(1337)
        start = time.perf_counter()
        model, _ = manager.select(pool)
        torch.cuda.synchronize()
        preparation = time.perf_counter() - start
        timed_generate(model, enc, probe['prompt'], 'cuda', 64, stop_at_eos=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stop = threading.Event()
        peaks = dict(rss_bytes=0, nvml_process_bytes=None)
        def monitor():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                handle = None
            while not stop.is_set():
                peaks['rss_bytes'] = max(peaks['rss_bytes'], psutil.Process().memory_info().rss)
                if handle is not None:
                    try:
                        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                            if proc.pid == os.getpid() and proc.usedGpuMemory is not None and proc.usedGpuMemory < 2**60:
                                peaks['nvml_process_bytes'] = max(peaks['nvml_process_bytes'] or 0, proc.usedGpuMemory)
                    except Exception:
                        pass
                stop.wait(.05)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            runs = [timed_generate(model, enc, probe['prompt'], 'cuda', 64, stop_at_eos=False) for _ in range(3)]
            output = timed_generate(model, enc, probe['prompt'], 'cuda', 64)
        finally:
            stop.set()
            thread.join()
        report['pools'][pool] = dict(prompt=probe['prompt'], checkpoint_sha256=row['sha256'],
            parameters=row['parameters'], runs=runs, output=output, preparation_seconds=preparation,
            memory=module_memory(model), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            sampled_peaks=peaks, preparation_logical_h2d_bytes=row['weight_bytes'], decode_h2d_bytes=0)
        durable_save(path, report)
        manager.model = None
        manager.pool = None
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(pool, 'generation complete', flush=True)
    report['status'] = 'complete'
    durable_save(path, report)
    durable_save(a.output / 'generation_STATUS.json', dict(status='complete', completed=8))


if __name__ == '__main__':
    main()
