"""Measure degradation when a class selects N-of-N experts without runtime E."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import tiktoken
import torch

from asi import DATA_ROOT, ROOT
from asi.experiments.posthoc import sample_windows, read_window, evaluate_window
from asi.experiments.cache_study import load_sessions, COUNTERS
from asi.experiments.decode_study import run_decode
from asi.models.original import DEFAULT_ARCHITECTURE, file_sha256, load_original_model, native_moes, sync
from asi.runtime.cache import NativeExpertSessionCache, move_backbone
from asi.runtime.routing import DomainSessionRouter


MODES = ('native', 'oracle_uniform', 'predicted_uniform', 'global_uniform', 'oracle_calibrated')


def fixed_mappings(mapping, evidence, n):
    """Freeze all choices before validation. No validation-driven fitting."""
    layers = {layer: {label: ids[:n] for label,ids in classes.items()} for layer,classes in mapping['layers'].items()}
    if any(len(ids) != n for classes in layers.values() for ids in classes.values()):
        raise ValueError('Every class needs at least N calibrated candidates')
    popularity = defaultdict(lambda: defaultdict(list))
    mass = defaultdict(lambda: defaultdict(float))
    for row in evidence:
        layer, label = str(row['layer']), row['label']
        popularity[layer][row['expert']].append(row['selection_rate'])
        mass[layer][label] += row['mean_routing_weight']
    global_layers = {}
    for layer in layers:
        if layer not in popularity:
            raise ValueError('Calibration missing layer')
        ids = sorted(popularity[layer], key=lambda i: (-float(np.mean(popularity[layer][i])), i))[:n]
        global_layers[layer] = {'global': ids}
    fixed = {'schema': 1, 'layers': layers, 'fixed_mass': dict(mass), 'provenance': mapping['provenance'],
             'note': 'Exactly N candidates per class/layer. A single class is active; classes may overlap.'}
    global_fixed = {'schema': 1, 'layers': global_layers, 'provenance': mapping['provenance']}
    return fixed, global_fixed


def metrics(rows, seed):
    results = {}
    for mode in MODES:
        chosen = [r for r in rows if r['mode'] == mode]
        if not chosen:
            continue
        tokens = sum(r['tokens'] for r in chosen)
        nll = sum(r['nll']*r['tokens'] for r in chosen)/tokens
        delta = sum(r['delta_nll']*r['tokens'] for r in chosen)/tokens
        # Resample paired windows, NOT tokens or repeated measurements.
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(1000):
            sample = rng.integers(0, len(chosen), size=len(chosen))
            boot.append(sum(chosen[i]['delta_nll']*chosen[i]['tokens'] for i in sample) /
                        sum(chosen[i]['tokens'] for i in sample))
        results[mode] = {'windows': len(chosen), 'tokens': tokens, 'nll': nll,
                         'perplexity': math.exp(nll), 'delta_nll': delta, 'perplexity_ratio_to_native': math.exp(delta),
                         'delta_nll_window_bootstrap_95': np.percentile(boot,[2.5,97.5]).tolist(),
                         'token_accuracy': sum(r['correct_tokens'] for r in chosen)/tokens,
                         'agreement_with_native': sum(r['native_agreement_tokens'] for r in chosen)/tokens,
                         'demand_misses': sum(r['demand_delta']['misses'] for r in chosen),
                         'prefetch_h2d_bytes': sum(r['prefetch_delta']['host_to_device_bytes'] for r in chosen),
                         'forward_h2d_bytes': sum(r['demand_delta']['host_to_device_bytes'] for r in chosen),
                         'forward_p50_ms': float(np.percentile([r['forward_seconds']*1000 for r in chosen],50))}
    return results


def paired_contrast(rows, left, right, seed):
    a = {r['window']:r for r in rows if r['mode']==left}
    b = {r['window']:r for r in rows if r['mode']==right}
    if set(a)!=set(b) or not a:
        raise ValueError('Contrasts require identical evaluation windows')
    keys=sorted(a)
    if any(a[k]['tokens']!=b[k]['tokens'] for k in keys):
        raise ValueError('Contrasts require identical token counts')
    differences=np.array([a[k]['nll']-b[k]['nll'] for k in keys])
    weights=np.array([a[k]['tokens'] for k in keys])
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(keys),size=(1000,len(keys)))
    boot=(differences[indices]*weights[indices]).sum(1)/weights[indices].sum(1)
    return {'left':left,'right':right,'delta_nll':float(np.average(differences,weights=weights)),
            'window_bootstrap_95':np.percentile(boot,[2.5,97.5]).tolist(),
            'note':'Negative favors left. Descriptive paired-window interval; document independence unverified.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--architecture',type=Path,default=DEFAULT_ARCHITECTURE)
    p.add_argument('--calibration-dir',type=Path,default=ROOT/'results/posthoc_04750')
    p.add_argument('--data-root',type=Path,default=DATA_ROOT)
    p.add_argument('--pool-manifest',type=Path,default=DATA_ROOT/'expert_pools.json')
    p.add_argument('--windows-per-pool',type=int,default=64)
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--sessions',type=Path,default=ROOT/'examples/sessions.jsonl')
    p.add_argument('--max-new-tokens',type=int,default=16)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--classifier-device',default='cpu')
    p.add_argument('--pin-memory',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if min(a.windows_per_pool,a.seq_len,a.max_new_tokens)<1 or a.output.exists():
        p.error('Positive budgets and a new output directory required')
    model, metadata = load_original_model(a.checkpoint,a.architecture)
    if a.seq_len+a.max_new_tokens > model.config.block_size:
        p.error('Requested context exceeds model limit')
    report_path = a.calibration_dir/'report.json'
    calibration_report = json.loads(report_path.read_text(encoding='utf-8'))
    mapping = json.loads((a.calibration_dir/'expert_labels.json').read_text(encoding='utf-8'))
    evidence = json.loads((a.calibration_dir/'calibration.json').read_text(encoding='utf-8'))['evidence']
    for source in (mapping['provenance'],calibration_report['metadata']):
        for key in ('checkpoint_sha256','architecture_sha256'):
            if source.get(key)!=metadata[key]:
                raise ValueError('Calibration provenance mismatch: '+key)
    manifest = json.loads(a.pool_manifest.read_text(encoding='utf-8'))
    if calibration_report['metadata'].get('pool_manifest_sha256') != file_sha256(a.pool_manifest):
        raise ValueError('Pool manifest differs from calibration')
    if any(w['split']!='train' for w in calibration_report['calibration_windows']):
        raise ValueError('Fixed mappings must come from train-only calibration')
    moes = native_moes(model)
    sizes = {moe.gate.topk for moe in moes.values()}
    if len(sizes)!=1:
        raise ValueError('This study requires the same native N in every layer')
    n = sizes.pop()
    fixed, global_fixed = fixed_mappings(mapping,evidence,n)
    windows, coverage = sample_windows(a.data_root,manifest,'val',a.seq_len,a.windows_per_pool,a.seed)
    if not windows:
        p.error('No validation windows available')
    random.Random(a.seed).shuffle(windows)
    names = sorted(manifest['pools'])
    router = DomainSessionRouter(names,device=a.classifier_device,max_pools=1,session_inertia=0)
    enc = tiktoken.get_encoding('gpt2')
    a.output.mkdir(parents=True)
    for name, value in [('fixed_labels.json',fixed),('global_labels.json',global_fixed)]:
        (a.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    classified = []
    targets, references, rows = [], [], []
    model.to(a.device)
    evaluate_window(model,windows[0],a.device)  # warm up outside recorded timings
    for index,w in enumerate(windows):
        x,y = read_window(w,'cpu')
        targets.append(y.flatten())
        router.previous_pools=[]
        start=time.perf_counter()
        predicted=router.route(enc.decode(x.flatten().tolist())).ranked_pools[0]
        sync(a.classifier_device)
        classified.append({'label':predicted,'seconds':time.perf_counter()-start})
        result=evaluate_window(model,w,a.device)
        references.append({'nll':result['nll'],'argmax':result['argmax']})
        rows.append({'mode':'native','window':index,'true_class':w['label'],'class':None,'tokens':a.seq_len,
                     'nll':result['nll'],'delta_nll':0.,'correct_tokens':int((result['argmax']==targets[-1]).sum()),
                     'native_agreement_tokens':a.seq_len,'forward_seconds':result['seconds'],
                     'prefetch_delta':dict.fromkeys(COUNTERS,0),'demand_delta':dict.fromkeys(COUNTERS,0)})
        if (index+1)%64==0: print(f'Native/classifier {index+1}/{len(windows)}',flush=True)
    model.to('cpu')
    memory = {}
    order = list(MODES[1:]); random.Random(a.seed).shuffle(order)
    for mode in order:
        selected_mapping = global_fixed if mode=='global_uniform' else fixed
        move_backbone(model,a.device)
        cache = NativeExpertSessionCache(model,selected_mapping,a.device,len(moes)*n,pin_memory=a.pin_memory,
                                         max_experts_per_layer=n,policy='fixed',
                                         fixed_weight_mode='calibrated' if mode=='oracle_calibrated' else 'uniform')
        if torch.device(a.device).type=='cuda': torch.cuda.reset_peak_memory_stats(a.device)
        try:
            for index,w in enumerate(windows):
                label = 'global' if mode=='global_uniform' else classified[index]['label'] if mode=='predicted_uniform' else w['label']
                before = cache.snapshot(); sync(a.device); start=time.perf_counter()
                cache.set_context_labels([label]); sync(a.device)
                prefetch_seconds=time.perf_counter()-start; prepared=cache.snapshot()
                result=evaluate_window(model,w,a.device); after=cache.snapshot()
                for layer,routes in result['routes'].items():
                    expected=selected_mapping['layers'][layer][label]
                    if any(ids != expected for ids in routes):
                        raise RuntimeError('Fixed N-of-N routing violation')
                demand={k:after.get(k,0)-prepared.get(k,0) for k in COUNTERS}
                if demand['misses'] or demand['host_to_device_bytes']:
                    raise RuntimeError('Fixed class should need no weight loads during its forward')
                rows.append({'mode':mode,'window':index,'true_class':w['label'],'class':label,'tokens':a.seq_len,
                             'nll':result['nll'],'delta_nll':result['nll']-references[index]['nll'],
                             'correct_tokens':int((result['argmax']==targets[index]).sum()),
                             'native_agreement_tokens':int((result['argmax']==references[index]['argmax']).sum()),
                             'forward_seconds':result['seconds'],'prefetch_seconds':prefetch_seconds,
                             'prefetch_delta':{k:prepared.get(k,0)-before.get(k,0) for k in COUNTERS},'demand_delta':demand})
                if (index+1)%64==0: print(f'{mode} {index+1}/{len(windows)}',flush=True)
            memory[mode]={'inventory':cache.inventory(),'cache':cache.snapshot()}
        finally:
            cache.close(); model.to('cpu')
        (a.output/'windows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # Generation: fixed class per turn, generated history maintained independently.
    sessions = [s for s in load_sessions(a.sessions) if s['split']=='test']
    generations=[]
    for mode in ('native','predicted_uniform','global_uniform'):
        cache=None
        if mode=='native': model.to(a.device)
        else:
            move_backbone(model,a.device)
            cache=NativeExpertSessionCache(model,global_fixed if mode=='global_uniform' else fixed,a.device,len(moes)*n,
                                           max_experts_per_layer=n,policy='fixed',pin_memory=a.pin_memory)
        try:
            for session in sessions:
                history=''; router.previous_pools=[]
                for turn,prompt in enumerate(session['prompts']):
                    tokens=enc.encode((history+'\n'+prompt).strip())[-min(a.seq_len,64):]
                    label=router.route(enc.decode(tokens)).ranked_pools[0] if mode!='global_uniform' else 'global'
                    before=cache.snapshot() if cache else {}
                    if cache: cache.set_context_labels([label])
                    prepared=cache.snapshot() if cache else {}
                    result=run_decode(model,tokens,enc,a.device,a.max_new_tokens,cache)
                    if cache and any(e['cache_delta']['misses'] for e in result['events']):
                        raise RuntimeError('Unexpected fixed-router load during prefill/decode')
                    record={k:v for k,v in result.items() if k!='probes'}
                    record.update({'mode':mode,'session':session['id'],'turn':turn,'prompt':prompt,'class':label,
                                   'input_tokens':tokens,'prefetch_delta':{k:prepared.get(k,0)-before.get(k,0) for k in COUNTERS}})
                    generations.append(record)
                    history=enc.decode(tokens)+result['text']
                    with (a.output/'generation.jsonl').open('a',encoding='utf-8') as out: out.write(json.dumps(record)+'\n')
            print(f'Generation {mode} complete',flush=True)
        finally:
            if cache: cache.close()
            model.to('cpu')
    summary=metrics(rows,a.seed)
    report={'metadata':metadata,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'torch_version':torch.__version__, 'gpu_name':torch.cuda.get_device_name(a.device) if torch.device(a.device).type=='cuda' else None,
            'generation_max_prompt_tokens':min(a.seq_len,64),
            'n_experts_per_class_per_layer':n,'score_func':model.config.score_func,'evaluation_coverage':coverage,
            'windows':windows,'classifier':classified,'classifier_top1_matches_corpus':sum(c['label']==w['label'] for c,w in zip(classified,windows))/len(windows),
            'summary':summary,'by_domain':{label:metrics([r for r in rows if r['true_class']==label],a.seed) for label in coverage if coverage[label]['windows']},
            'paired_contrasts':[paired_contrast(rows,left,right,a.seed) for left,right in
                                [('oracle_uniform','global_uniform'),('predicted_uniform','global_uniform'),
                                 ('oracle_calibrated','oracle_uniform'),('predicted_uniform','oracle_uniform')]],
            'memory':memory,'artifact_hashes':{name:file_sha256(a.calibration_dir/name) for name in ('expert_labels.json','calibration.json','report.json')},
            'generation_note':'Free-running sessions diverge in generated histories; use paired corpus windows for quantitative degradation.',
            'limitations':['No runtime E in any fixed variant. Uniform mass equals route_scale; calibrated mass is a train-only mean, not runtime E.',
                           'Oracle class uses dataset labels; predicted class uses one top-1 class, no multilabel union.',
                           'Windows are from the training corpus validation split; historical exposure and document independence are unverified.',
                           'Window bootstrap intervals are descriptive, not independent-document confidence bounds.',
                           'No quality claim from generated examples alone; NLL/perplexity measure token prediction, not task correctness.',
                           'Forward timings are diagnostic, not a repeated isolated performance benchmark.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Expertos fijos N sobre N, sin E', '', f'N={n} por clase y capa. E se omite por completo durante inferencia.', '',
           '| Modo | NLL | Delta NLL | PPL / nativa | Acierto token |', '|---|---:|---:|---:|---:|']
    for mode,r in summary.items():
        lines.append(f'| {mode} | {r["nll"]:.5f} | {r["delta_nll"]:+.5f} | {r["perplexity_ratio_to_native"]:.3f} | {100*r["token_accuracy"]:.2f}% |')
    lines += ['',f'Coincidencia top-1 del clasificador con la etiqueta del corpus: {100*report["classifier_top1_matches_corpus"]:.1f}%.',
              'Las variantes fijas no cargan pesos durante el forward, prefill ni decode tras preparar la clase; cambios de clase sí transfieren.',
              'NLL menor es mejor. El ratio de perplexity no es un porcentaje de pérdida de capacidad general.',
              'oracle_calibrated separa parcialmente el efecto de escala: el softmax original no renormaliza necesariamente sus N seleccionados.',
              'Validación por dominio y exclusiones en report.json. No hay validación para dominios sin ventanas.']
    (a.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
