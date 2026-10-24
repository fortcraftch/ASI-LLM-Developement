"""Frozen OLMoE mask interventions: train-only calibration and paired generation."""
import argparse
import json
from pathlib import Path
import random
import time

from asi.taxonomy import digest
from asi.experiments.public_moe import sha256, read_lock, input_ids, decode
from asi.experiments.recovery_protocol import durable_save, padded_batch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_masks(stats, old):
    """Equal-token-weight categories; shared core maximizes minimum usage."""
    categories = sorted(old)
    layers = sorted(old[categories[0]], key=int)
    if len(layers) != 16 or any(len(stats[c]['0']['counts']) != 64 for c in categories):
        raise ValueError('This preregistered allocation requires OLMoE 16 layers / 64 experts')
    names = ['recalibrated32', 'shared8_category24', 'first2_free', 'uniform36',
             'last2_free', 'first2_free_budget512']
    masks = {'old32': old, **{n: {} for n in names}}
    for c in categories:
        for n in names:
            masks[n][c] = {}
        for layer in layers:
            rank = sorted(range(64), key=lambda e: (-stats[c][layer]['counts'][e], e))
            shared = sorted(range(64), key=lambda e: (
                -min(stats[x][layer]['counts'][e] / stats[x][layer]['tokens'] for x in categories), e))[:8]
            for name in names:
                count = 36 if name == 'uniform36' else 32
                if name == 'first2_free_budget512':
                    count = 28 if 2 <= int(layer) < 8 else 27
                ids = rank[:count]
                if name.startswith('first2_free') and int(layer) < 2:
                    ids = list(range(64))
                if name == 'last2_free' and int(layer) >= 14:
                    ids = list(range(64))
                if name == 'shared8_category24':
                    ids = shared + [e for e in rank if e not in shared][:24]
                # Restricted routing uses native probabilities, not these placeholders.
                masks[name][c][layer] = {'experts': ids, 'weights': [1 / len(ids)] * len(ids)}
    return masks


def random_controls(old, seeds):
    """Uniform candidate sets, independent of calibration scores or answers.

    A seed selects one common set per layer for every category. This tests
    whether category-based selection improves on a category-blind allocation.
    """
    if len(set(seeds)) != len(seeds):
        raise ValueError('Random control seeds must be unique')
    result = {}
    categories = sorted(old)
    if not categories:
        raise ValueError('At least one category required')
    layers = sorted(old[categories[0]], key=int)
    for c in categories:
        if set(old[c]) != set(layers) or any(len(old[c][l]['experts']) != 32 for l in layers):
            raise ValueError('Random32 control requires matching 32-candidate layer budgets')
    for seed in seeds:
        rng = random.Random(seed)
        sets = {l: sorted(rng.sample(range(64), 32)) for l in layers}
        result[f'random32_seed{seed}'] = {
            c: {l: {'experts': list(sets[l]), 'weights': [1/32]*32} for l in layers}
            for c in categories}
    return result


def global_control(stats):
    """One shared mask ranked by mean per-token usage across calibration categories.

    Categories receive equal weight; prompt categories never select a mask.
    Expert-index tie breaking is deterministic and does not inspect evaluation.
    """
    import math
    import copy
    categories = sorted(stats)
    if not categories:
        raise ValueError('Empty calibration')
    for layers in stats.values():
        if set(layers) != {str(l) for l in range(16)}:
            raise ValueError('Expected 16 calibration layers')
        for row in layers.values():
            if not math.isfinite(row['tokens']) or row['tokens'] <= 0 or len(row['counts']) != 64:
                raise ValueError('Invalid calibration dimensions or token count')
            if any(not math.isfinite(n) or n < 0 for n in row['counts']):
                raise ValueError('Invalid expert counts')
    mask = {}
    for layer in map(str, range(16)):
        scores = [sum(stats[c][layer]['counts'][e] / stats[c][layer]['tokens']
                      for c in categories) / len(categories) for e in range(64)]
        ids = sorted(range(64), key=lambda e: (-scores[e], e))[:32]
        mask[layer] = {'experts': ids, 'weights': [1/32]*32}
    return {'global32': {c: copy.deepcopy(mask) for c in categories}}


def reuse_calibration(report, donor):
    """Copy completed calibration only, never donor outputs or completion state."""
    import copy
    for key in ('source', 'files'):
        if donor['identity'][key] != report['identity'][key]:
            raise ValueError('Calibration donor identity mismatch: ' + key)
    for key in ('per_category', 'batch_size', 'active', 'max_length', 'seed'):
        if donor['identity']['protocol'][key] != report['identity']['protocol'][key]:
            raise ValueError('Calibration donor protocol mismatch: ' + key)
    examples = donor['calibration_examples']
    ids = [r['id'] for r in examples]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(donor['calibration_done']):
        raise ValueError('Calibration donor is incomplete or duplicated')
    if set(ids) & {r['id'] for r in donor['evaluation_examples']}:
        raise ValueError('Calibration donor overlaps evaluation')
    for category in {r['category'] for r in examples}:
        layers = donor['stats'][category]
        if set(layers) != {str(l) for l in range(16)}:
            raise ValueError('Incomplete donor layers')
        if any(v['tokens'] <= 0 or len(v['counts']) != 64 for v in layers.values()):
            raise ValueError('Invalid donor statistics')
    for key in ('calibration_examples', 'evaluation_examples', 'calibration_done', 'stats'):
        report[key] = copy.deepcopy(donor[key])


def select_calibration(corpus, old_ids, rejected, categories, tokenizer, per_category, max_length):
    rng = random.Random(20261010)
    selected = []
    for category in categories:
        candidates = sorted((r for r in corpus['examples'] if r['split'] == 'train'
                             and r['category'] == category and r['id'] not in rejected), key=lambda r: r['id'])
        rng.shuffle(candidates)
        candidates.sort(key=lambda r: r['id'] not in old_ids)
        group = []
        for r in candidates:
            prompt = input_ids(tokenizer, r['prompt'], limit=max_length)[0].tolist()
            tokens = prompt + tokenizer.encode(r['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(tokens) <= max_length:
                group.append({'id': r['id'], 'category': category, 'tokens': tokens, 'prompt_length': len(prompt)})
            if len(group) == per_category:
                break
        if len(group) != per_category:
            raise ValueError('Insufficient eligible train examples: ' + category)
        selected.extend(group)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/mask_intervention_20261010'))
    p.add_argument('--model-dir', type=Path, default=Path('public_models/olmoe'))
    p.add_argument('--calibration-per-category', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--new-tokens', type=int, default=256)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--calibrate-only', action='store_true')
    p.add_argument('--random-seeds', type=int, nargs='+', default=[],
                   help='Add category-blind random32 controls; use a new output directory')
    p.add_argument('--reuse-calibration', type=Path)
    p.add_argument('--random-only', action='store_true')
    p.add_argument('--global-only', action='store_true', help='Only category-blind calibrated global32 control')
    args = p.parse_args()
    if args.random_only and not args.random_seeds:
        p.error('--random-only requires --random-seeds')
    if args.global_only and (args.random_only or args.random_seeds):
        p.error('--global-only cannot be combined with random controls')
    if min(args.calibration_per_category, args.batch_size, args.new_tokens) < 1:
        p.error('Counts must be positive')
    report_path = args.output / 'report.json'
    if report_path.exists() and not args.resume:
        p.error('Use --resume or a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import psutil
    from asi.models.olmoe_storage import load_disk_backed
    from asi.models.hf_olmoe import OlmoeExpertCache, CategoryCalibration
    from asi.experiments.recovery import objective, generation_metrics
    sources = [Path('results/recovery_corpus_v3/corpus.json'),
               Path('results/recovery_calibration_batched_v3/report.json'),
               Path('results/routing_paired_20261009/report.json'),
               Path('results/recovery_200_screened_20261008/plan.json')]
    corpus, old, paired, pilot = [read(x) for x in sources]
    identity = {'source': digest(read_lock(args.model_dir)),
                'files': {str(x): sha256(x) for x in sources},
                'protocol': {'per_category': args.calibration_per_category, 'batch_size': args.batch_size,
                             'new_tokens': args.new_tokens, 'active': 8, 'max_length': 384,
                             'seed': 20261010, 'version': 1}}
    if args.random_seeds:
        if len(set(args.random_seeds)) != len(args.random_seeds):
            p.error('Random seeds must be unique')
        identity['protocol'].update(version=2, random_seeds=args.random_seeds,
                                    random_control='category-blind uniform32 per layer')
    if args.reuse_calibration:
        identity['calibration_donor_sha256'] = sha256(args.reuse_calibration)
    if args.random_only:
        identity['protocol']['random_only'] = True
    if args.global_only:
        identity['protocol'].update(version=3, global_only=True,
                                    global_control='mean category-normalized per-token usage; index tie break')
    if old['source_identity'] != identity['source'] or paired['source_identity'] != identity['source']:
        raise ValueError('Model identity mismatch')
    report = read(report_path) if report_path.exists() else {
        'identity': identity, 'completed': False, 'calibration_done': [], 'stats': {}, 'records': [],
        'limitations': ['Exploratory six-item diagnostic, not an independent final benchmark.',
                        'Frozen base weights; no adapters or new training.',
                        'Greedy generation, EOS stopping, same token budget; no sampling.',
                        'First two free layers are our hypothesis, not an established result of the cited paper.',
                        'Timings use the existing Python expert cache implementation.']}
    if report['identity'] != identity:
        raise ValueError('Resume identity changed')
    if report['completed']:
        print('Already completed:', report_path, flush=True)
        return
    if args.reuse_calibration and 'calibration_examples' not in report:
        reuse_calibration(report, read(args.reuse_calibration))
        if report['evaluation_examples'] != paired['examples']:
            raise ValueError('Donor evaluation differs from frozen diagnostic')
        durable_save(report_path, report)
    model, tokenizer, store = load_disk_backed(args.model_dir, 2 * 2**30, 12 * 2**30)
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    cache = OlmoeExpertCache(model, 'cuda', capacity=576, store=store)
    device = torch.device('cuda')
    try:
        if 'calibration_examples' not in report:
            report['calibration_examples'] = select_calibration(
                corpus, set(old['train_ids']), set(pilot.get('rejected_candidates', {})),
                sorted(old['mappings']['32']), tokenizer, args.calibration_per_category, 384)
            report['evaluation_examples'] = paired['examples']
            train_ids = {r['id'] for r in report['calibration_examples']}
            if train_ids & {r['id'] for r in paired['examples']}:
                raise ValueError('Calibration/evaluation overlap')
            durable_save(report_path, report)
        for category in sorted(old['mappings']['32']):
            pending = [r for r in report['calibration_examples'] if r['category'] == category
                       and r['id'] not in report['calibration_done']]
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                calibration = CategoryCalibration(16, 64)
                calibration.labels = [category]
                tokens, attention = padded_batch(batch, tokenizer.eos_token_id, device)
                valid = attention.flatten().bool()
                def observe(layer, indices, weights):
                    calibration.observe(layer, indices[valid], weights[valid])
                cache.callbacks.append(observe)
                cache.set_context(None)
                began = time.perf_counter()
                try:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        # Avoid allocating vocabulary logits for calibration.
                        model.model(input_ids=tokens, attention_mask=attention, use_cache=False)
                    torch.cuda.synchronize()
                finally:
                    cache.callbacks.remove(observe)
                for (_, layer), row in calibration.rows.items():
                    target = report['stats'].setdefault(category, {}).setdefault(str(layer),
                        {'tokens': 0, 'counts': [0] * 64, 'mass': [0.] * 64})
                    target['tokens'] += row['tokens']
                    for key in ('counts', 'mass'):
                        target[key] = [a + b for a, b in zip(target[key], row[key].tolist())]
                report['calibration_done'].extend(r['id'] for r in batch)
                durable_save(report_path, report)
                print(f"Calibration {len(report['calibration_done'])}/{len(report['calibration_examples'])}: {time.perf_counter()-began:.1f}s", flush=True)
        masks = {} if (args.random_only or args.global_only) else build_masks(report['stats'], old['mappings']['32'])
        if args.global_only:
            masks.update(global_control(report['stats']))
        masks.update(random_controls(old['mappings']['32'], args.random_seeds))
        report['masks'] = masks
        durable_save(report_path, report)
        if args.calibrate_only:
            return
        done = {(r['variant'], r['id']) for r in report['records']}
        originals = {r['id']: r for r in corpus['examples']}
        for variant, mappings in masks.items():
            for row in report['evaluation_examples']:
                if (variant, row['id']) in done:
                    continue
                before = cache.snapshot()
                began = time.perf_counter()
                cache.set_context(mappings[row['category']], mixture='restricted', active=8)
                preparation = time.perf_counter() - began
                ready = cache.snapshot()
                with torch.inference_mode():
                    _, ce, _ = objective(model, row, device, alpha=1.)
                ce_value = ce.item()
                del ce
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = decode(model, torch.tensor([row['tokens'][:row['prompt_length']]]),
                                       args.new_tokens, device, stop_token_ids={tokenizer.eos_token_id})
                after = cache.snapshot()
                if after.get('loads', 0) != ready.get('loads', 0):
                    raise RuntimeError('Expert transfer during fixed-mask inference')
                text = tokenizer.decode(generated['token_ids'], skip_special_tokens=True)
                record = {'variant': variant, 'id': row['id'], 'category': row['category'],
                          'ce': ce_value, 'text': text, **generated,
                          'eos': bool(generated['token_ids'] and generated['token_ids'][-1] == tokenizer.eos_token_id),
                          'metrics': generation_metrics(generated['token_ids'], text, originals[row['id']]),
                          'prepare_seconds': preparation, 'before': before, 'ready': ready, 'after': after,
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'rss_bytes': psutil.Process().memory_info().rss}
                report['records'].append(record)
                durable_save(report_path, report)
                print(f"Evaluation {len(report['records'])}/{len(masks)*len(report['evaluation_examples'])} {variant} {row['id']}: CE={ce_value:.3f}", flush=True)
        report['completed'] = True
        durable_save(report_path, report)
    finally:
        cache.close()
    from asi.analysis.mask_intervention_report import build
    build(report_path)


if __name__ == '__main__':
    main()
