"""Strict per-layer residency study with serial prefill and real greedy decode."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import tiktoken
import torch

from asi import ROOT
from asi.experiments.cache_study import load_sessions, POLICIES, COUNTERS
from asi.models.original import DEFAULT_ARCHITECTURE, IncrementalDecoder, autocast, load_original_model, native_moes, sync
from asi.runtime.cache import NativeExpertSessionCache, move_backbone, module_memory
from asi.runtime.routing import DomainSessionRouter, ExpertUsagePredictor


@torch.inference_mode()
def run_decode(model, tokens, enc, device, new_tokens, cache=None):
    decoder = IncrementalDecoder(model)
    events, probes, outputs = [], [], []
    captures = {}
    handles = [moe.gate.register_forward_hook(
        lambda gate, args, out, lid=lid: captures.__setitem__(lid, out[1].detach()))
        for lid, moe in native_moes(model).items()]
    def step(token, phase):
        before = cache.snapshot() if cache else {}
        sync(device)
        started = time.perf_counter()
        with autocast(device):
            logits = decoder.step(token)
        sync(device)
        seconds = time.perf_counter() - started
        after = cache.snapshot() if cache else {}
        values = logits[0, -1].float()
        columns = torch.linspace(0, len(values)-1, 64, device=device).long()
        probes.append(values[columns].cpu())
        events.append({'phase': phase, 'position': decoder.position - 1, 'input_token': token,
                       'seconds': seconds, 'argmax': int(values.argmax()),
                       'routes': {str(lid): indices.cpu().tolist() for lid, indices in captures.items()},
                       'cache_delta': {key: after.get(key, 0)-before.get(key, 0) for key in COUNTERS},
                       'resident_by_layer': dict(Counter(k[0] for k in after.get('resident_experts', [])))})
        return int(values[:enc.n_vocab].argmax())
    try:
        for token in tokens:
            next_token = step(token, 'prefill')
        for index in range(new_tokens):
            outputs.append(next_token)
            if next_token == enc.eot_token or index == new_tokens-1:
                break
            next_token = step(next_token, 'decode')
    finally:
        for handle in handles:
            handle.remove()
    return {'tokens': outputs, 'text': enc.decode(outputs), 'events': events, 'probes': torch.stack(probes)}


@torch.inference_mode()
def check_incremental(model, tokens, device):
    """FP32 full-vocabulary check against the unmodified forward, before caching."""
    full = model(torch.tensor([tokens], device=device))[0].float()
    decoder = IncrementalDecoder(model)
    incremental = torch.cat([decoder.step(token) for token in tokens], dim=1).float()
    difference = float((full-incremental).abs().max())
    equivalent = torch.allclose(full, incremental, atol=2e-4, rtol=2e-4)
    if not equivalent:
        raise RuntimeError(f'Incremental adapter disagrees with original FP32 forward: max diff {difference}')
    return {'fp32_max_abs_logit_difference': difference, 'atol': 2e-4, 'rtol': 2e-4,
            'tokens_checked': len(tokens), 'all_vocabulary': True,
            'note': 'Serial and batched arithmetic need not be bitwise identical; cache policies share the same serial baseline.'}


def summarize_decode(rows):
    summary = {}
    for policy in POLICIES:
        turns = [r for r in rows if r['policy'] == policy]
        if not turns:
            continue
        item = {'turn_observations': len(turns), 'generated_tokens': sum(len(r['tokens']) for r in turns),
                'output_mismatch_turns': sum(not r['output_equal'] for r in turns),
                'route_mismatch_turns': sum(not r['routes_equal'] for r in turns),
                'max_probe_difference': max(r['max_probe_difference'] for r in turns),
                'prefetch_h2d_bytes': sum(r['prefetch_delta']['host_to_device_bytes'] for r in turns)}
        for phase in ('prefill', 'decode'):
            events = [e for r in turns for e in r['events'] if e['phase'] == phase]
            counts = Counter()
            for event in events:
                counts.update(event['cache_delta'])
            requests = counts['hits'] + counts['misses']
            item[phase] = {'forward_steps': len(events), 'totals': dict(counts),
                          'hit_rate': counts['hits']/requests if requests else None,
                          'steps_without_loads': sum(e['cache_delta']['misses'] == 0 for e in events),
                          'fraction_without_loads': sum(e['cache_delta']['misses'] == 0 for e in events)/len(events) if events else None,
                          'h2d_bytes_per_step': counts['host_to_device_bytes']/len(events) if events else None,
                          'p50_ms': float(np.percentile([e['seconds']*1000 for e in events], 50)) if events else None,
                          'p95_ms': float(np.percentile([e['seconds']*1000 for e in events], 95)) if events else None}
        summary[policy] = item
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--architecture', type=Path, default=DEFAULT_ARCHITECTURE)
    p.add_argument('--expert-labels', type=Path, required=True)
    p.add_argument('--usage-predictor', type=Path, required=True)
    p.add_argument('--sessions', type=Path, default=ROOT/'examples/sessions.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--experts-per-layer', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--classifier-device', default='cpu')
    p.add_argument('--no-classifier', action='store_true')
    p.add_argument('--pin-memory', action='store_true')
    a = p.parse_args()
    if min(a.experts_per_layer, a.max_new_tokens, a.seq_len, a.repeats) < 1:
        p.error('All budgets must be positive')
    if a.output.exists():
        p.error('Use a new output directory')
    sessions = load_sessions(a.sessions)
    tests = [s for s in sessions if s['split'] == 'test']
    mapping = json.loads(a.expert_labels.read_text(encoding='utf-8'))
    predictor = ExpertUsagePredictor.from_dict(json.loads(a.usage_predictor.read_text(encoding='utf-8')))
    if set(predictor.training_sessions) & {s['id'] for s in tests} or predictor.provenance.get('online_adapted'):
        p.error('Predictor must be frozen and trained on separate sessions')
    if predictor.provenance.get('classifier_enabled') != (not a.no_classifier):
        p.error('Classifier settings differ from predictor training')
    torch.manual_seed(a.seed)
    model, metadata = load_original_model(a.checkpoint, a.architecture)
    if a.seq_len + a.max_new_tokens - 1 > model.config.block_size:
        p.error('Prompt and generated tokens must fit the KV context')
    for source in (mapping, predictor.to_dict()):
        for key in ('checkpoint_sha256', 'architecture_sha256'):
            if source.get('provenance', {}).get(key) != metadata[key]:
                raise ValueError('Provenance mismatch: '+key)
    moes = native_moes(model)
    if any(moe.gate.topk > a.experts_per_layer for moe in moes.values()):
        p.error('The router top-k must fit the per-layer budget')
    if set(predictor.experts) != {(l, e) for l, moe in moes.items() for e in range(len(moe.experts))}:
        p.error('Predictor expert universe mismatch')
    capacity = len(moes) * a.experts_per_layer
    names = sorted({label for layer in mapping['layers'].values() for label in layer})
    router = DomainSessionRouter(names, device=a.classifier_device,
                                 max_pools=predictor.provenance['max_labels'], load_classifier=not a.no_classifier)
    enc = tiktoken.get_encoding('gpt2')
    model.to(a.device)
    check_tokens = enc.encode(tests[0]['prompts'][0])[:16]
    adapter_check = check_incremental(model, check_tokens, a.device)
    a.output.mkdir(parents=True)
    (a.output/'sessions.jsonl').write_text(a.sessions.read_text(encoding='utf-8'), encoding='utf-8')
    reference, contexts = {}, {}
    for session in tests:
        history = ''
        for turn, prompt in enumerate(session['prompts']):
            tokens = enc.encode((history+'\n'+prompt).strip())[-a.seq_len:]
            contexts[session['id'], turn] = tokens
            result = run_decode(model, tokens, enc, a.device, a.max_new_tokens)
            reference[session['id'], turn] = result
            history = enc.decode(tokens) + result['text']
    model.to('cpu')
    rows, memory = [], {}
    rng = random.Random(a.seed)
    frozen = predictor.to_dict()
    for repeat in range(a.repeats):
        order = list(POLICIES)
        rng.shuffle(order)
        for policy in order:
            for session in tests:
                cache = None
                router.previous_pools = []
                previous = []
                if policy == 'resident':
                    model.to(a.device)
                else:
                    move_backbone(model, a.device)
                    cache = NativeExpertSessionCache(model, mapping, a.device, capacity, pin_memory=a.pin_memory,
                                                     max_experts_per_layer=a.experts_per_layer)
                if torch.device(a.device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(a.device)
                try:
                    for turn, _ in enumerate(session['prompts']):
                        tokens = contexts[session['id'], turn]
                        sync(a.classifier_device)
                        start = time.perf_counter()
                        labels = router.route(enc.decode(tokens)).ranked_pools
                        sync(a.classifier_device)
                        classifier_seconds = time.perf_counter()-start
                        before = cache.snapshot() if cache else {}
                        sync(a.device); start = time.perf_counter()
                        if cache:
                            if policy == 'semantic':
                                cache.set_context_labels(labels)
                            else:
                                candidates = [] if policy == 'lru' else predictor.rank(labels, previous, mode=policy)
                                cache.prefetch_experts(candidates)
                        sync(a.device); prefetch_seconds = time.perf_counter()-start
                        after = cache.snapshot() if cache else {}
                        result = run_decode(model, tokens, enc, a.device, a.max_new_tokens, cache)
                        ref = reference[session['id'], turn]
                        probes_equal_shape = result['probes'].shape == ref['probes'].shape
                        row = {k: v for k, v in result.items() if k != 'probes'}
                        row.update({'session': session['id'], 'turn': turn, 'scenario': session.get('scenario'),
                            'policy': policy, 'repeat': repeat, 'order': order, 'input_tokens': tokens,
                            'labels': labels, 'classifier_seconds': classifier_seconds, 'prefetch_seconds': prefetch_seconds,
                            'prefetch_delta': {k: after.get(k, 0)-before.get(k, 0) for k in COUNTERS},
                            'output_equal': result['tokens'] == ref['tokens'],
                            'routes_equal': [e['routes'] for e in result['events']] == [e['routes'] for e in ref['events']],
                            'max_probe_difference': float((result['probes']-ref['probes']).abs().max()) if probes_equal_shape else None,
                            'cache': cache.snapshot() if cache else None})
                        rows.append(row)
                        with (a.output/'turns.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(row)+'\n')
                        if not probes_equal_shape or not row['output_equal'] or not row['routes_equal'] or row['max_probe_difference'] > 1e-5:
                            raise RuntimeError('Exact-cache decode mismatch; saved failing turn')
                        if cache and any(n > a.experts_per_layer for n in cache.peak_by_layer.values()):
                            raise RuntimeError('Per-layer GPU residency exceeded')
                        previous = labels
                    memory[f'{repeat}:{policy}:{session["id"]}'] = cache.inventory() if cache else module_memory(model)
                finally:
                    if cache:
                        cache.close()
                    model.to('cpu')
            print(f'Decode repeat {repeat+1}/{a.repeats}: {policy} complete', flush=True)
    assert predictor.to_dict() == frozen
    summary = summarize_decode(rows)
    report = {'metadata': metadata, 'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
              'adapter_check': adapter_check, 'capacity_total': capacity, 'summary': summary, 'memory': memory,
              'summary_by_scenario': {s: summarize_decode([r for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},
              'note': 'Objetivo: reducir residencia y swapping, no acelerar el modelo residente. Solo expertos enrutados están limitados; '
                      'backbone, shared y KV permanecen en GPU. Prefill serial; decode incremental real. El primer token generado '
                      'sale del último prefill; decode cuenta los forwards posteriores. Caché persiste entre turnos, KV se reconstruye '
                      'por turno sobre el contexto retenido. Repeticiones no son sesiones independientes. Predictor congelado del piloto anterior.'}
    (a.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Dos expertos por capa: memoria y swapping', '', report['note'], '',
             '| Política | Pasos decode | Sin cargas | H2D MiB/paso | Hit rate | p50 ms/paso |', '|---|---:|---:|---:|---:|---:|']
    for policy, result in summary.items():
        d = result['decode']
        if not d['forward_steps']:
            lines.append(f'| {policy} | 0 | N/A | N/A | N/A | N/A |')
            continue
        hit = '-' if d['hit_rate'] is None else f'{100*d["hit_rate"]:.1f}%'
        lines.append(f'| {policy} | {d["forward_steps"]} | {100*d["fraction_without_loads"]:.1f}% | '
                     f'{d["h2d_bytes_per_step"]/1024**2:.3f} | {hit} | {d["p50_ms"]:.2f} |')
    lines += ['', 'Prefill y precargas se contabilizan por separado en report.json; no quedan ocultos como decode gratuito.',
              'Equivalencia de salidas greedy, rutas y logits muestreados comprobada frente al baseline incremental residente.',
              'Solo tres sesiones diagnósticas distintas; no prueba generalización ni ejecución de un modelo oficial enorme.']
    (a.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
