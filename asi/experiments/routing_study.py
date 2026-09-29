"""Controlled OLMoE residency/activation ablation; no LLM training."""
import argparse
import json
import time
import importlib.metadata
from pathlib import Path
import torch
from asi import ROOT
from asi.taxonomy import Taxonomy, digest
from asi.models.hf_olmoe import OlmoeExpertCache, CategoryCalibration
from asi.models.olmoe_storage import load_disk_backed
from asi.experiments.public_moe import (CATEGORIES, MODEL, read_lock, hardware,
    choose_capacity, sample_examples, input_ids, decode, check_rss)


CASES = [(8, 8, 'constant'), (8, 8, 'restricted'), (8, 8, 'restricted_native_mass'),
         (16, 8, 'restricted'), (32, 8, 'restricted'), (32, 16, 'restricted')]


def summarize_report(source, output):
    from collections import defaultdict
    import statistics
    report = json.loads(source.read_text(encoding='utf-8'))
    if not report.get('completed') or not report.get('records') or not report.get('baseline'):
        raise ValueError('Only summarize a completed study with a native baseline')
    groups = defaultdict(list)
    for row in report['records']:
        groups[(row['k'], row['active'], row['policy'])].append(row)
    rows = []
    for (k, active, policy), samples in groups.items():
        rows.append({'k': k, 'active': active, 'policy': policy, 'prompts': len(samples),
            'mean_native_continuation_nll': statistics.mean(x['native_continuation_nll'] for x in samples),
            'mean_teacher_kl': statistics.mean(x['teacher_kl'] for x in samples),
            'peak_vram_gib': max(x['peak_cuda_allocated_bytes'] for x in samples) / 2**30,
            'loads_during_generation': sum(x['cache_after'].get('loads', 0)-x['cache_before'].get('loads', 0) for x in samples),
            'logical_read_bytes_during_generation': sum(x['cache_after']['disk_ram'].get('logical_read_bytes', 0)-x['cache_before']['disk_ram'].get('logical_read_bytes', 0) for x in samples)})
    result = {'source': str(source), 'native_reference_nll': statistics.mean(x['native_continuation_nll'] for x in report['baseline']),
              'rows': rows, 'limitations': report['limitations']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f"{x['k']} residentes / {x['active']} activos\n" + {'constant': 'Constante', 'restricted': 'Router restringido',
              'restricted_native_mass': 'Router + reescala'}[x['policy']] for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].barh(labels, [x['mean_native_continuation_nll'] for x in rows], color='#337ca0')
    axes[0].axvline(result['native_reference_nll'], color='#bd4528', linestyle='--', label='Modelo original')
    axes[0].set_xlabel('NLL media de continuaciones nativas (menor es mejor)')
    axes[0].set_title('Acuerdo con la referencia, no exactitud de respuestas')
    axes[0].legend()
    axes[1].barh(labels, [x['peak_vram_gib'] for x in rows], color='#40806a')
    axes[1].set_xlabel('VRAM asignada máxima (GiB), generación + puntuación')
    axes[1].set_title('Memoria observada: ' + report['hardware']['gpu']['name'])
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('OLMoE 7B: fijar residencia y recuperar el routing | Piloto de 3 prompts')
    fig.tight_layout()
    fig.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


def reference_scores(model, ids, continuation, device, teacher=None):
    """Score only a frozen native continuation, not the prompt or an answer label."""
    sequence = torch.tensor([ids[0].tolist() + continuation], device=device)
    with torch.inference_mode():
        logits = model(sequence, use_cache=False).logits[0, ids.shape[1]-1:-1].float()
        logp = logits.log_softmax(-1)
        nll = -logp.gather(-1, torch.tensor(continuation, device=device)[:, None]).mean()
        kl = None if teacher is None else (teacher.exp() * (teacher - logp)).sum(-1).mean().item()
    return {'native_continuation_nll': nll.item(), 'teacher_kl': kl}, logp.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT / 'public_models/olmoe')
    p.add_argument('--native-report', type=Path)
    p.add_argument('--summarize', type=Path, help='Summarize a completed study without loading the model')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration-per-category', type=int, default=4)
    p.add_argument('--new-tokens', type=int, default=24)
    p.add_argument('--ram-gib', type=float, default=32)
    p.add_argument('--ram-cache-gib', type=float, default=.25)
    args = p.parse_args()
    if args.output.exists() or args.calibration_per_category < 1 or args.new_tokens < 1:
        p.error('Choose a new output and positive sample/token counts')
    if args.summarize:
        if args.output.with_suffix('.png').exists():
            p.error('Choose a new figure path too')
        summarize_report(args.summarize, args.output)
        return
    if args.native_report is None:
        p.error('--native-report is required to run the experiment')
    if args.ram_gib <= 0 or args.ram_cache_gib < 0:
        p.error('RAM budget must be positive and cache ceiling nonnegative')
    report = hardware(args.model_dir, args.ram_gib)
    if not report['gpu']:
        p.error('CUDA required')
    identity = digest(read_lock(args.model_dir))
    native = json.loads(args.native_report.read_text(encoding='utf-8'))
    if not native.get('executed') or native.get('backend') != 'native' or native['source_identity'] != identity:
        p.error('Need an executed native report for exactly this checkpoint')
    taxonomy = Taxonomy.read(ROOT / 'configs/knowledge_v1.json')
    train = sample_examples(ROOT / 'data/classifier_seed_v1/train_val.jsonl', taxonomy, 'train', args.calibration_per_category)
    validation = sample_examples(ROOT / 'data/classifier_seed_v1/train_val.jsonl', taxonomy, 'val', 1)
    native_rows = {r['prompt_id']: r for r in native['records'] if r['repeat'] == 0}
    model, tokenizer, store = load_disk_backed(args.model_dir, int(args.ram_cache_gib*2**30), int(args.ram_gib*2**30))
    feasible, blocked = [], []
    for k, active, policy in CASES:
        try:
            choose_capacity(model, k, report['gpu']['free_bytes'], 2*2**30)
            feasible.append((k, active, policy))
        except ValueError as e:
            blocked.append({'k': k, 'active': active, 'policy': policy, 'reason': str(e)})
    if not feasible:
        p.error('No configured resident set fits GPU')
    cache = OlmoeExpertCache(model, capacity=max(k for k, _, _ in feasible)*model.config.num_hidden_layers, store=store)
    result = {'model': MODEL, 'source_identity': identity, 'hardware': report, 'completed': False,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')},
        'new_tokens': args.new_tokens, 'calibration_per_category': args.calibration_per_category,
        'train_examples': train, 'validation_examples': validation, 'cases': feasible, 'blocked': blocked,
        'records': [], 'baseline': [], 'limitations': [
            'Synthetic small calibration; one validation prompt per category; no independent final test.',
            'NLL and KL use frozen original-model continuations, not human correctness labels.',
            'One pass, no stable latency conclusions; NLL forwards and generation are separately measured.',
            'Selection is by native routing frequency, not verified semantic specialization.',
            'No fine-tuning performed; activated experts differ only in explicitly marked cases.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    save()
    try:
        collector = CategoryCalibration(model.config.num_hidden_layers, model.config.num_experts)
        cache.callbacks.append(collector.observe)
        with torch.inference_mode():
            for row in train:
                print('Calibration: ' + row['id'], flush=True)
                collector.labels = [row['category']]
                model(input_ids(tokenizer, row['text']).cuda(), use_cache=False)
                check_rss(int(args.ram_gib*2**30))
        cache.callbacks.remove(collector.observe)
        result['calibration_statistics'] = [{'category': category, 'layer': layer, 'tokens': values['tokens'],
            'counts': values['counts'].tolist(), 'mass': values['mass'].tolist()}
            for (category, layer), values in collector.rows.items()]
        mappings = {k: collector.mapping(CATEGORIES, k) for k, _, _ in feasible}
        result['mappings'] = mappings
        teachers = {}
        for row in validation:
            print('Teacher: ' + row['id'], flush=True)
            ids = input_ids(tokenizer, row['text'])
            previous = native_rows[row['id']]
            if ids[0].tolist() != previous['input_ids']:
                raise ValueError('Native reference prompt tokens differ')
            scores, teacher = reference_scores(model, ids, previous['token_ids'], 'cuda')
            teachers[row['id']] = teacher.cpu()
            result['baseline'].append({'prompt_id': row['id'], **scores, 'reference_text': previous['text']})
            save()
        for k, active, policy in feasible:
            for row in validation:
                print(f"K={k}, active={active}, {policy}: {row['id']}", flush=True)
                started = time.perf_counter()
                cache.set_context(mappings[k][row['category']], mixture=policy, active=active)
                preparation = time.perf_counter() - started
                before = cache.snapshot()
                ids = input_ids(tokenizer, row['text'])
                torch.cuda.reset_peak_memory_stats()
                generated = decode(model, ids, args.new_tokens, 'cuda', lambda: check_rss(int(args.ram_gib*2**30)))
                after = cache.snapshot()
                scores, _ = reference_scores(model, ids, native_rows[row['id']]['token_ids'], 'cuda', teachers[row['id']].cuda())
                if after.get('loads', 0) != before.get('loads', 0):
                    raise RuntimeError('Unexpected loads during generation')
                result['records'].append({'prompt_id': row['id'], 'k': k, 'active': active, 'policy': policy,
                    'prepare_seconds': preparation, **generated, **scores,
                    'text': tokenizer.decode(generated['token_ids']), 'cache_before': before, 'cache_after': after,
                    'rss_bytes': check_rss(int(args.ram_gib*2**30)), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
                save()
        result['completed'] = True
        save()
    finally:
        cache.close()

