"""Frozen train/test session study of exact expert caching (TFG 1.4, 1.5, 2.2)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from asi import ROOT
from asi.models.original import DEFAULT_ARCHITECTURE, autocast, file_sha256, load_original_model, native_moes, sync
from asi.runtime.cache import NativeExpertSessionCache, module_memory, move_backbone
from asi.runtime.routing import DomainSessionRouter, ExpertUsagePredictor


POLICIES = ('resident', 'lru', 'popularity', 'semantic', 'learned')
COUNTERS = ('hits', 'misses', 'prefetch_loads', 'evictions', 'host_to_device_bytes')


def load_sessions(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ids = set()
    prompts = {'train': set(), 'test': set()}
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('Session IDs must be nonempty and unique')
        ids.add(row['id'])
        if row.get('split') not in prompts:
            raise ValueError('Every session requires an explicit train/test split')
        if not isinstance(row.get('prompts'), list) or len(row['prompts']) < 2:
            raise ValueError('A session needs at least two prompts')
        for prompt in row['prompts']:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('Prompts must be nonempty strings')
            prompts[row['split']].add(' '.join(prompt.lower().split()))
    if not all(prompts.values()) or prompts['train'] & prompts['test']:
        raise ValueError('Need separate train/test sessions without identical prompts across splits')
    return rows


@torch.inference_mode()
def forward_probe(model, tokens, device):
    """Time forward only; verify all route IDs, argmax and 64 logits/token."""
    x = torch.tensor([tokens[:-1]], device=device)
    y = torch.tensor(tokens[1:], device=device)
    captured = {}
    handles = [moe.gate.register_forward_hook(
        lambda gate, args, output, layer=layer: captured.__setitem__(layer, output[1].detach()))
        for layer, moe in native_moes(model).items()]
    sync(device)
    started = time.perf_counter()
    try:
        with autocast(device):
            logits, _ = model(x)
        sync(device)
        seconds = time.perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    routes = {layer: indices.cpu() for layer, indices in captured.items()}
    digest = hashlib.sha256()
    for layer, indices in sorted(routes.items()):
        digest.update(str(layer).encode())
        digest.update(indices.numpy().tobytes())
    demanded = [(layer, int(expert)) for layer, indices in routes.items() for expert in indices.unique().tolist()]
    columns = torch.linspace(0, logits.shape[-1] - 1, min(64, logits.shape[-1]), device=device).long()
    return {'forward_seconds': seconds,
            'nll': float(F.cross_entropy(logits[0].float(), y)),
            'route_sha256': digest.hexdigest(), 'demands': demanded,
            'probe': logits[0, :, columns].float().cpu(),
            'argmax': logits[0].argmax(-1).cpu()}


def summarize(records):
    results = {}
    for policy in POLICIES:
        rows = [row for row in records if row['policy'] == policy]
        if not rows:
            continue
        totals = Counter()
        for row in rows:
            totals.update(row['cache_delta'])
        requests = totals['hits'] + totals['misses']
        times = [row['pipeline_seconds'] for row in rows]
        tokens = sum(row['tokens'] for row in rows)
        results[policy] = {
            'turn_observations': len(rows), 'tokens': tokens,
            'nll_token_weighted': sum(row['nll'] * row['tokens'] for row in rows) / tokens,
            'pipeline_p50_seconds': float(np.percentile(times, 50)),
            'pipeline_p95_seconds': float(np.percentile(times, 95)),
            'hit_rate': totals['hits'] / requests if requests else None,
            'cache_totals': dict(totals),
            'max_probe_abs_difference': max(row['max_probe_abs_difference'] for row in rows),
            'route_mismatch_turns': sum(not row['routes_equal'] for row in rows),
            'argmax_mismatch_turns': sum(not row['argmax_equal'] for row in rows),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture', type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument('--expert-labels', type=Path, required=True)
    parser.add_argument('--sessions', type=Path, default=ROOT / 'examples/sessions.jsonl')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--classifier-device', default='cpu')
    parser.add_argument('--no-classifier', action='store_true')
    parser.add_argument('--max-hot-experts', type=int, default=52)
    parser.add_argument('--prefetch-experts', type=int, default=32, help='Same candidate budget for all prefetch policies')
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--pin-memory', action='store_true')
    args = parser.parse_args()
    if args.seq_len < 2 or args.repeats < 1 or not 0 <= args.prefetch_experts <= args.max_hot_experts:
        parser.error('Positive repeats, seq-len >= 2 and 0 <= prefetch <= capacity required')
    sessions = load_sessions(args.sessions)
    if args.output.exists():
        parser.error('Output already exists; select a new run directory')
    mapping = json.loads(args.expert_labels.read_text(encoding='utf-8-sig'))
    torch.manual_seed(args.seed)
    model, metadata = load_original_model(args.checkpoint, args.architecture)
    for key in ('checkpoint_sha256', 'architecture_sha256'):
        if mapping.get('provenance', {}).get(key) != metadata[key]:
            raise ValueError('Expert mapping provenance mismatch: ' + key)
    if args.seq_len > model.config.block_size:
        parser.error('seq-len exceeds checkpoint context length')
    moes = native_moes(model)
    if args.max_hot_experts < max(len(moe.experts) for moe in moes.values()):
        parser.error('Capacity must fit the potential union of experts in one layer')
    names = sorted({label for layer in mapping['layers'].values() for label in layer})
    router = DomainSessionRouter(names, device=args.classifier_device, max_pools=min(3, len(names)),
                                 load_classifier=not args.no_classifier)
    enc = tiktoken.get_encoding('gpt2')
    prepared = {}
    for session in sessions:
        history = ''
        prepared[session['id']] = []
        for prompt in session['prompts']:
            history = (history + '\n' + prompt).strip()
            tokens = enc.encode(history)[-(args.seq_len + 1):]
            if len(tokens) < 2:
                raise ValueError('Each retained context must contain at least two tokens')
            # Classification sees exactly the context fed into the model, not its target suffix.
            prepared[session['id']].append((tokens, enc.decode(tokens[:-1])))
    predictor = ExpertUsagePredictor((layer, eid) for layer, moe in moes.items() for eid in range(len(moe.experts)))
    predictor.provenance = {key: metadata[key] for key in ('checkpoint_sha256', 'architecture_sha256')}
    predictor.provenance.update({'classifier_enabled': not args.no_classifier,
                                'classifier_model': 'mdonigian/fineweb-edu-topic-classifier',
                                'max_labels': min(3, len(names)), 'sessions_sha256': file_sha256(args.sessions)})
    predictor.training_sessions = [s['id'] for s in sessions if s['split'] == 'train']
    args.output.mkdir(parents=True)
    (args.output / 'sessions.jsonl').write_text(args.sessions.read_text(encoding='utf-8-sig'), encoding='utf-8')
    model.to(args.device)
    # Warm up kernels and classifier outside the observations, then reset session state.
    warm_tokens, warm_context = prepared[sessions[0]['id']][0]
    router.route(warm_context)
    forward_probe(model, warm_tokens, args.device)
    train_records = []
    for session in sessions:
        if session['split'] != 'train':
            continue
        previous = []
        router.previous_pools = []
        for turn, (tokens, context) in enumerate(prepared[session['id']]):
            labels = router.route(context).ranked_pools
            observed = forward_probe(model, tokens, args.device)
            predictor.observe(previous, labels, observed['demands'])
            train_records.append({'session': session['id'], 'turn': turn, 'labels': labels,
                                  'previous_labels': previous, 'demands': observed['demands'], 'tokens': tokens})
            previous = labels
    frozen = json.dumps(predictor.to_dict(), sort_keys=True)
    (args.output / 'predictor.json').write_text(json.dumps(predictor.to_dict(), indent=2), encoding='utf-8')
    (args.output / 'training_observations.json').write_text(json.dumps(train_records, indent=2), encoding='utf-8')
    references = {}
    for session in sessions:
        if session['split'] == 'test':
            for turn, (tokens, _) in enumerate(prepared[session['id']]):
                references[session['id'], turn] = forward_probe(model, tokens, args.device)
    # Baseline fits in GPU for this experiment; this is not a large-checkpoint loader.
    model.to('cpu')
    records = []
    memories = {}
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        order = list(POLICIES)
        rng.shuffle(order)
        for policy in order:
            for session in sessions:
                if session['split'] != 'test':
                    continue
                cache = None
                router.previous_pools = []
                previous = []
                if policy == 'resident':
                    model.to(args.device)
                else:
                    move_backbone(model, args.device)
                    cache = NativeExpertSessionCache(model, mapping, args.device, args.max_hot_experts,
                                                     pin_memory=args.pin_memory)
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(args.device)
                try:
                    for turn, (tokens, context) in enumerate(prepared[session['id']]):
                        sync(args.classifier_device)
                        started = time.perf_counter()
                        labels = router.route(context).ranked_pools
                        sync(args.classifier_device)
                        classifier_seconds = time.perf_counter() - started
                        before = cache.snapshot() if cache else {}
                        sync(args.device)
                        started = time.perf_counter()
                        if cache:
                            if policy == 'lru':
                                candidates = []
                            elif policy == 'semantic':
                                by_layer = {layer: list(dict.fromkeys(eid for label in labels
                                    for eid in mapping['layers'].get(str(layer), {}).get(label, []))) for layer in moes}
                                candidates = [(layer, ids[i]) for i in range(max(map(len, by_layer.values()), default=0))
                                              for layer, ids in by_layer.items() if i < len(ids)]
                            else:
                                candidates = predictor.rank(labels, previous, mode=policy)
                            cache.prefetch_experts(candidates[:args.prefetch_experts])
                        sync(args.device)
                        prefetch_seconds = time.perf_counter() - started
                        result = forward_probe(model, tokens, args.device)
                        after = cache.snapshot() if cache else {}
                        reference = references[session['id'], turn]
                        record = {key: value for key, value in result.items() if key not in ('probe', 'argmax')}
                        record.update({'policy': policy, 'repeat': repeat, 'policy_order': order,
                            'session': session['id'], 'scenario': session.get('scenario', 'unspecified'), 'turn': turn,
                            'labels': labels, 'previous_labels': previous, 'tokens': len(tokens) - 1,
                            'classifier_seconds': classifier_seconds, 'prefetch_seconds': prefetch_seconds,
                            'pipeline_seconds': classifier_seconds + prefetch_seconds + result['forward_seconds'],
                            'cache_delta': {key: after.get(key, 0) - before.get(key, 0) for key in COUNTERS},
                            'resident_experts': after.get('resident_experts'),
                            'routes_equal': result['route_sha256'] == reference['route_sha256'],
                            'argmax_equal': torch.equal(result['argmax'], reference['argmax']),
                            'max_probe_abs_difference': float((result['probe'] - reference['probe']).abs().max()),
                            'nll_difference': result['nll'] - reference['nll']})
                        records.append(record)
                        # Incremental results survive an interrupted study.
                        with (args.output / 'turns.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(record) + '\n')
                        previous = labels
                    memories[f'{repeat}:{policy}:{session["id"]}'] = cache.inventory() if cache else module_memory(model)
                finally:
                    if cache:
                        cache.close()
                    model.to('cpu')
            print(f'Repeat {repeat + 1}/{args.repeats}: {policy} complete', flush=True)
    assert frozen == json.dumps(predictor.to_dict(), sort_keys=True), 'Evaluation mutated the predictor'
    summary = summarize(records)
    report = {'metadata': metadata, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'predictor_frozen_on_test': True, 'classifier_memory': module_memory(router.classifier),
              'summary': summary,
              'summary_by_scenario': {scenario: summarize([r for r in records if r['scenario'] == scenario])
                                      for scenario in sorted({r['scenario'] for r in records})},
              'memory': memories,
              'note': 'Sesiones redactadas de diagnóstico: contextos de prompts, un forward por turno, sin conversación generada. '
                      'El tiempo suma clasificación, selección/precarga y forward; excluye verificación e I/O. '
                      'El clasificador también se ejecuta en controles. La caché se reinicia entre sesiones. '
                      'Los logits muestreados no prueban equivalencia de todo el vocabulario. En CPU la residencia es solo lógica.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Estudio de caché por conversaciones', '', report['note'], '',
             '| Política | p50 ms | p95 ms | Hits | H2D MiB | NLL |', '|---|---:|---:|---:|---:|---:|']
    for policy, row in summary.items():
        hit = '-' if row['hit_rate'] is None else f'{100 * row["hit_rate"]:.1f}%'
        lines.append(f'| {policy} | {1000*row["pipeline_p50_seconds"]:.2f} | {1000*row["pipeline_p95_seconds"]:.2f} | '
                     f'{hit} | {row["cache_totals"].get("host_to_device_bytes", 0)/1024**2:.2f} | {row["nll_token_weighted"]:.5f} |')
    lines += ['', 'Las repeticiones no son conversaciones independientes. Comparar bytes y tiempo además de hits.',
              'Este piloto no demuestra generalización, calidad de tareas ni mejora en decode incremental.',
              'Ver report.json y turns.jsonl para igualdad de rutas, logits muestreados y resultados por escenario.']
    lines += ['', f'Turnos con diferencias de rutas: {sum(not r["routes_equal"] for r in records)}.',
              f'Turnos con diferencias de argmax: {sum(not r["argmax_equal"] for r in records)}.',
              f'Diferencia absoluta máxima en logits muestreados: {max(r["max_probe_abs_difference"] for r in records):.8g}.']
    (args.output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if any(not row['routes_equal'] or not row['argmax_equal'] or row['max_probe_abs_difference'] > 1e-5 for row in records):
        raise RuntimeError('Exact-cache equivalence check failed; inspect saved results')


if __name__ == '__main__':
    main()
