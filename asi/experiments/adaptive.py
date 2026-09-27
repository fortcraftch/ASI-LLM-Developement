"""Calibrate optional inference policies; does not train or modify an LLM."""
import argparse
import json
from pathlib import Path
import time

import torch
from asi.data.identity import manifest_matches

from asi.experiments.posthoc import sample_windows, read_window
from asi.runtime.generation import load_model, validate_pool_identity, inference_context
from asi.runtime.adaptive import ContextMixture, WarmExpertCache, fingerprint
from asi.runtime.cache import ExpertCacheManager


@torch.inference_mode()
def score(model, window, device):
    x, y = read_window(window, device)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with inference_context(device):
        logits, _ = model(x)
    losses = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    accuracy = (logits.argmax(-1) == y).float().mean()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return {'nll': float(losses), 'token_accuracy': float(accuracy), 'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool-manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--contexts', required=True, help='Semicolon-separated sets, e.g. health;health,biology')
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--windows-per-pool', type=int, default=16)
    parser.add_argument('--warm-mib', type=float, default=64)
    parser.add_argument('--seed', type=int, default=260927)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.seq_len, args.windows_per_pool) < 1 or args.warm_mib < 0:
        parser.error('Use a new output file and positive sizes')
    model, metadata = load_model(args.checkpoint)
    manifest = json.loads(args.pool_manifest.read_text(encoding='utf-8'))
    names = list(manifest['pools'])
    validate_pool_identity(model, metadata, names)
    if args.seq_len > model.config.block_size:
        parser.error('Context exceeds model capacity')
    contexts = [sorted(set(group.split(','))) for group in args.contexts.split(';')]
    if any(name not in names for group in contexts for name in group):
        parser.error('Unknown context pool')
    recipe = metadata.get('training_recipe') or {}
    if recipe.get('source_manifest_sha256') and not manifest_matches(args.data_root / 'manifest.json', recipe['source_manifest_sha256']):
        parser.error('Dataset identity differs from training recipe')
    windows, coverage = sample_windows(args.data_root, manifest, 'val', args.seq_len, args.windows_per_pool, args.seed)
    requested = set(name for group in contexts for name in group)
    if any(coverage[name]['windows'] != args.windows_per_pool for name in requested):
        parser.error('Not enough disjoint validation windows for every requested pool')
    reference = {}
    cache = ExpertCacheManager(args.device, len(names))
    cache.initialize(model)
    try:
        for i, window in enumerate(windows):
            if window['label'] not in requested:
                continue
            pool = names.index(window['label'])
            # Original baseline: unrestricted for base; oracle pool for categorized.
            pools = list(range(len(names))) if recipe.get('family') == 'base' else [pool]
            cache.prepare(model, pools)
            model.set_active_pools(pools)
            reference[i] = score(model, window, args.device)
    finally:
        cache.offload_all(model)
    profiles = []
    for mode in ['dense', 'int8']:
        for labels in contexts:
            ids = sorted(names.index(name) for name in labels)
            cache = (WarmExpertCache(args.device, len(ids), int(args.warm_mib * 1024**2)) if mode == 'int8'
                     else ExpertCacheManager(args.device, len(ids)))
            cache.initialize(model)
            records = []
            try:
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(args.device)
                cache.prepare(model, ids, list(range(len(names))))
                before = cache.snapshot()
                with ContextMixture(model, ids):
                    for i, window in enumerate(windows):
                        if window['label'] in labels:
                            result = score(model, window, args.device)
                            records.append({'window': window, 'reference': reference[i], 'candidate': result,
                                            'nll_delta': result['nll'] - reference[i]['nll']})
                after = cache.snapshot()
                if any(after.get(key, 0) != before.get(key, 0) for key in ['loads', 'host_to_device_bytes', 'evictions']):
                    raise RuntimeError('Residency changed inside a fixed context')
                profiles.append({'mode': mode, 'pools': ids, 'labels': labels,
                                 'nll_delta': sum(r['nll_delta'] for r in records) / len(records),
                                 'nll_delta_by_pool': {name: sum(r['nll_delta'] for r in records if r['window']['label'] == name) / args.windows_per_pool for name in labels},
                                 'records': records, 'memory': cache.inventory(), 'before': before, 'after': after})
            finally:
                cache.offload_all(model)
    report = {'schema': 1, 'identity': {'checkpoint_sha256': fingerprint(args.checkpoint),
              'pool_manifest_sha256': fingerprint(args.pool_manifest), 'policy': 'uniform_union_v1'},
              'profiles': profiles, 'settings': vars(args), 'coverage': coverage,
              'limitations': ['Validation windows are policy calibration data, not a final independent test.',
                              'Multiple pools are tested on separate single-domain windows, not mixed-domain questions.',
                              'Observed mean NLL is not a guarantee for individual prompts or longer contexts.',
                              'Uniform union executes all selected experts, unlike native top-k.',
                              'Timings are single-pass teacher-forced forwards, not generation throughput or warm-transition benchmarks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps([{'mode': p['mode'], 'labels': p['labels'], 'nll_delta': p['nll_delta']} for p in profiles], indent=2))


if __name__ == '__main__':
    main()
