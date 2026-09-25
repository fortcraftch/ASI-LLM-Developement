"""Post-hoc calibration and exact-cache comparison for an existing native MoE."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from asi.models.original import DEFAULT_ARCHITECTURE,load_original_model,autocast,sync,file_sha256
from asi.analysis.experts import ExpertCalibrator
from asi.runtime.cache import NativeExpertSessionCache
from asi.models.original import native_moes
from asi.runtime.cache import module_memory


def sample_windows(root,manifest,split,length,count,seed):
    """Disjoint windows; balanced domains and round-robin categories, no repetition."""
    rng=random.Random(seed); windows=[]; coverage={}
    for label,info in manifest['pools'].items():
        categories=[]
        for category in info['categories']:
            choices=[]
            for path in sorted((root/category).glob(f'{split}_*.npy')):
                if path.name.endswith('.part.npy'): continue
                arr=np.load(path,mmap_mode='r')
                if arr.ndim!=1 or arr.dtype!=np.uint16:
                    arr._mmap.close(); raise ValueError(f'Invalid token shard: {path}')
                available=len(arr)//(length+1); arr._mmap.close()
                # Bound enumeration for full-size corpora; random blocks do not overlap.
                for block in rng.sample(range(available),min(available,count)):
                    choices.append({'label':label,'split':split,'shard':str(path.resolve()),
                                    'start':block*(length+1),'length':length+1})
            rng.shuffle(choices)
            if choices: categories.append(choices)
        selected=[]
        while len(selected)<count and any(categories):
            for choices in categories:
                if choices and len(selected)<count: selected.append(choices.pop())
        windows.extend(selected)
        coverage[label]={'windows':len(selected),'tokens':len(selected)*length}
    return windows,coverage


def read_window(w,device):
    arr=np.load(w['shard'],mmap_mode='r')
    values=np.array(arr[w['start']:w['start']+w['length']],dtype=np.int64,copy=True)
    arr._mmap.close()
    if len(values)!=w['length']: raise ValueError('Dataset changed after window selection')
    tokens=torch.from_numpy(values).to(device)
    return tokens[:-1].unsqueeze(0),tokens[1:].unsqueeze(0)


@torch.inference_mode()
def evaluate_window(model,w,device):
    x,y=read_window(w,device)
    captures={}; handles=[]
    for layer,moe in native_moes(model).items():
        handles.append(moe.gate.register_forward_hook(lambda gate,args,out,lid=layer:captures.__setitem__(lid,out[1].detach())))
    sync(device); start=time.perf_counter()
    try:
        with autocast(device): logits,_=model(x)
        sync(device); seconds=time.perf_counter()-start
    finally:
        for handle in handles: handle.remove()
    losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.flatten(),reduction='none')
    probe_ids=torch.linspace(0,logits.shape[-1]-1,64,device=device).long()
    result={'nll':float(losses.mean()),'seconds':seconds,'routes':{str(k):v.cpu().tolist() for k,v in captures.items()},
            'probe':logits[0,:,probe_ids].float().cpu(),'argmax':logits[0].argmax(-1).cpu()}
    return result


def build_mapping(calibration,k):
    result=calibration.propose_mapping(k)
    result['selection']='token-normalized native routing frequency per layer and domain'
    return result


def coverage_curves(baselines,ranked_mapping,ks):
    counts=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for w,result in baselines:
        for layer,rows in result['routes'].items():
            ranking=ranked_mapping['layers'][layer][w['label']]
            for k in ks:
                allowed=set(ranking[:k]); key=f'{layer}:{k}'
                counts[w['label']][key][0]+=sum(e in allowed for row in rows for e in row)
                counts[w['label']][key][1]+=sum(len(row) for row in rows)
    records=[]
    for label,values in counts.items():
        for key,(hits,total) in values.items():
            layer,k=map(int,key.split(':'))
            records.append({'label':label,'layer':layer,'experts_per_label':k,'covered_selections':hits,
                            'total_selections':total,'recall':hits/total})
    return records


def public_result(result):
    return {k:v for k,v in result.items() if k not in ('probe','argmax','routes')}


def write_summary(output,payload,mapping,curves,global_curves):
    lines=['# Calibración posterior del checkpoint', '',
           f"Step: {payload['metadata']['step']}. Validación guardada: {payload['metadata']['val_loss']:.6f}.",
           '', 'Muestras disjuntas de calibración train y evaluación val del corpus proporcionado.',
           'La exposición durante el entrenamiento anterior no se ha verificado.', '',
           '| Modo | NLL | Forward medio (ms) | Precarga media (ms) |',
           '|---|---:|---:|---:|']
    reference=payload['summary']['prefetch']
    lines.append(f"| Residente, router original | {reference['baseline_nll']:.6f} | {reference['mean_baseline_forward_seconds']*1000:.2f} | 0 |")
    for name,row in payload['summary'].items():
        lines.append(f"| {name} | {row['nll']:.6f} | {row['mean_forward_seconds']*1000:.2f} | {row['mean_prefetch_seconds']*1000:.2f} |")
    lines+=['','Tiempos de un ensayo, con contextos de 128 tokens en esta ejecución; no son tokens/s de producción.',
            'Prefetch conserva el router y carga fallos bajo demanda; restrict cambia la selección.', '',
            '| Expertos candidatos/capa | Cobertura por dominio | Popularidad global |',
            '|---|---:|---:|']
    for k in sorted({r['experts_per_label'] for r in curves}):
        values=[]
        for data in (curves,global_curves):
            rows=[r for r in data if r['experts_per_label']==k]
            values.append(sum(r['covered_selections'] for r in rows)/sum(r['total_selections'] for r in rows))
        lines.append(f'| {k} | {values[0]:.2%} | {values[1]:.2%} |')
    lines+=['','La mejora frente a popularidad global debe medirse antes de afirmar especialización semántica.',
            'Las etiquetas son candidatas, se solapan y requieren revisión; no representan áreas de conocimiento exclusivas.', '',
            '## Memoria de expertos y transferencias','']
    for policy,data in payload['memory'].items():
        cache=data['cache']; memory=data['inventory']
        gpu=sum(v for k,v in memory['expert_weight_bytes_by_device'].items() if k.startswith('cuda'))
        lines.append(f"- {policy}: {gpu/2**20:.2f} MiB de pesos expertos en GPU, {cache['ram_backing_bytes']/2**20:.2f} MiB de respaldo RAM; {cache.get('host_to_device_bytes',0)/2**20:.2f} MiB transferidos; hit rate {cache['hit_rate']:.2%}.")
    lines+=['','El ahorro anterior solo incluye pesos enrutados. Backbone, shared experts, KV, temporales y allocator se contabilizan aparte en report.json.',
            '', '## Cobertura de validación','']
    for label,row in payload['evaluation_coverage'].items():
        lines.append(f"- {label}: {row['windows']} ventanas, {row['tokens']} tokens.")
    lines+=['','## Candidatos por capa y dominio','', '| Capa | Dominio | Expertos ordenados por frecuencia |','|---|---|---|']
    for layer,domains in mapping['layers'].items():
        for label,ids in domains.items(): lines.append(f"| {layer} | {label} | {', '.join(map(str,ids))} |")
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--architecture',type=Path,default=DEFAULT_ARCHITECTURE)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--pool-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/posthoc_04750'))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--calibration-windows',type=int,default=32)
    p.add_argument('--evaluation-windows',type=int,default=8)
    p.add_argument('--experts-per-label',type=int,default=4)
    p.add_argument('--max-hot-experts',type=int,default=52)
    p.add_argument('--pin-memory',action='store_true')
    p.add_argument('--seed',type=int,default=1337)
    args=p.parse_args()
    if min(args.seq_len,args.calibration_windows,args.evaluation_windows,args.experts_per_label,args.max_hot_experts)<1:
        p.error('Sizes must be positive')
    manifest=json.loads(args.pool_manifest.read_text(encoding='utf-8-sig'))
    calibration_windows,train_coverage=sample_windows(args.data_root,manifest,'train',args.seq_len,args.calibration_windows,args.seed)
    eval_windows,val_coverage=sample_windows(args.data_root,manifest,'val',args.seq_len,args.evaluation_windows,args.seed+1)
    if not calibration_windows or not eval_windows: p.error('Need both calibration train windows and evaluation val windows')
    missing=[label for label,v in train_coverage.items() if not v['windows']]
    if missing: p.error(f'No calibration windows for domains: {missing}')
    model,metadata=load_original_model(args.checkpoint,args.architecture)
    if args.seq_len>model.config.block_size: p.error('seq-len exceeds model context')
    if not model.config.n_activated_experts<=args.experts_per_label<=model.config.n_routed_experts:
        p.error('experts-per-label must lie between native top-k and expert count')
    if args.max_hot_experts<model.config.n_routed_experts:
        p.error('Budget must fit at least one full layer for unrestricted prefill')
    metadata['pool_manifest_sha256']=file_sha256(args.pool_manifest)
    print(f'Loaded original step={metadata["step"]}, val_loss={metadata["val_loss"]}, experts/layer={model.config.n_routed_experts}',flush=True)
    model.to(args.device)
    # Warm kernels, independent of calibration counts.
    evaluate_window(model,calibration_windows[0],args.device)
    calibration=ExpertCalibrator(model)
    with calibration,torch.inference_mode():
        for i,w in enumerate(calibration_windows):
            calibration.set_labels([w['label']]); x,_=read_window(w,args.device)
            with autocast(args.device): model(x)
            if (i+1)%32==0: print(f'Calibration {i+1}/{len(calibration_windows)}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    calibration.write(args.output/'calibration.json')
    calibration.export_router_vectors(args.output/'router_vectors')
    mapping=build_mapping(calibration,args.experts_per_label)
    mapping['provenance']={**metadata, 'calibration_split': 'train'}
    # Freeze a domain-independent control using train evidence only.
    global_rates=defaultdict(lambda:defaultdict(float))
    for row in calibration.report()['evidence']:
        global_rates[str(row['layer'])][row['expert']]+=row['selection_rate']
    mapping['global_layers']={layer:{'global':sorted(scores,key=lambda e:(-scores[e],e))}
                              for layer,scores in global_rates.items()}
    (args.output/'expert_labels.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    evidence=calibration.report()['evidence']
    by_expert=defaultdict(list)
    for row in evidence: by_expert[(row['layer'],row['expert'])].append(row['selection_rate'])
    for row in evidence:
        rates=by_expert[(row['layer'],row['expert'])]
        avg=sum(rates)/len(rates)
        row['relative_to_mean_domain_rate']=row['selection_rate']/avg if avg else None
    with (args.output/'expert_domain_scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(evidence[0])); writer.writeheader(); writer.writerows(evidence)
    baselines=[]
    for i,w in enumerate(eval_windows):
        baselines.append((w,evaluate_window(model,w,args.device)))
        if (i+1)%16==0: print(f'Baseline {i+1}/{len(eval_windows)}',flush=True)
    curves=coverage_curves(baselines,build_mapping(calibration,model.config.n_routed_experts),
                           sorted({model.config.n_activated_experts,args.experts_per_label,6,model.config.n_routed_experts}&set(range(1,model.config.n_routed_experts+1))))
    (args.output/'coverage_curves.json').write_text(json.dumps(curves,indent=2),encoding='utf-8')
    # A domain-independent popularity baseline tests whether labels add predictive value.
    rates=defaultdict(lambda:defaultdict(float))
    for row in evidence: rates[str(row['layer'])][row['expert']]+=row['selection_rate']
    global_mapping={'layers':{layer:{label:sorted(scores,key=scores.get,reverse=True)
                                     for label in manifest['pools']} for layer,scores in rates.items()}}
    global_curves=coverage_curves(baselines,global_mapping,sorted({r['experts_per_label'] for r in curves}))
    (args.output/'global_popularity_coverage.json').write_text(json.dumps(global_curves,indent=2),encoding='utf-8')
    (args.output/'native_routes.json').write_text(json.dumps([
        {'window':w,'routes':result['routes']} for w,result in baselines],indent=2),encoding='utf-8')
    records=[]; final_memory={}; initial_memory=module_memory(model)
    for policy in ['prefetch','restrict']:
        cache=NativeExpertSessionCache(model,mapping,device=args.device,max_hot_experts=args.max_hot_experts,pin_memory=args.pin_memory,policy=policy)
        try:
            if torch.device(args.device).type=='cuda': torch.cuda.reset_peak_memory_stats(args.device)
            for index,(w,reference) in enumerate(baselines):
                before=cache.snapshot(); sync(args.device); start=time.perf_counter()
                # Dataset labels are an oracle here; input classification is evaluated separately in chat mode.
                cache.set_context_labels([w['label']])
                sync(args.device); prefetch_seconds=time.perf_counter()-start
                result=evaluate_window(model,w,args.device)
                after=cache.snapshot()
                delta={k:after.get(k,0)-before.get(k,0) for k in ['hits','misses','evictions','host_to_device_bytes','prefetch_loads','transfer_and_lookup_seconds']}
                route_mismatch=sum(a!=b for layer,rows in reference['routes'].items()
                                   for rr,ss in zip(rows,result['routes'][layer]) for a,b in zip(rr,ss))
                error=float((result['probe']-reference['probe']).abs().max())
                match=float((result['argmax']==reference['argmax']).float().mean())
                if policy=='prefetch' and (route_mismatch or error>1e-4 or abs(result['nll']-reference['nll'])>1e-5):
                    raise RuntimeError('Exact-cache equivalence failed')
                records.append({'window':w,'policy':policy,'baseline_nll':reference['nll'],
                                'baseline_forward_seconds':reference['seconds'],**public_result(result),
                                'prefetch_seconds':prefetch_seconds,'nll_delta':result['nll']-reference['nll'],
                                'routing_slot_mismatches':route_mismatch,'probe_max_abs_logit_error':error,
                                'argmax_agreement':match,'cache_delta':delta})
                if (index+1)%16==0: print(f'{policy} {index+1}/{len(baselines)}',flush=True)
            final_memory[policy]={'cache':cache.snapshot(),'inventory':cache.inventory()}
        finally: cache.close()
    summary={}
    for policy in ['prefetch','restrict']:
        rows=[r for r in records if r['policy']==policy]
        summary[policy]={'windows':len(rows),'baseline_nll':sum(r['baseline_nll'] for r in rows)/len(rows),
                         'nll':sum(r['nll'] for r in rows)/len(rows),
                         'mean_nll_delta':sum(r['nll_delta'] for r in rows)/len(rows),
                         'mean_baseline_forward_seconds':sum(r['baseline_forward_seconds'] for r in rows)/len(rows),
                         'mean_forward_seconds':sum(r['seconds'] for r in rows)/len(rows),
                         'mean_prefetch_seconds':sum(r['prefetch_seconds'] for r in rows)/len(rows),
                         'route_mismatches':sum(r['routing_slot_mismatches'] for r in rows),
                         'probe_max_abs_logit_error':max(r['probe_max_abs_logit_error'] for r in rows)}
    payload={'metadata':metadata,'seed':args.seed,'device':args.device,'torch':torch.__version__,
             'experts_per_label':args.experts_per_label,'max_hot_experts':args.max_hot_experts,
             'calibration_coverage':train_coverage,'evaluation_coverage':val_coverage,
             'calibration_windows':calibration_windows,'baseline_model_tensors':initial_memory,
             'summary':summary,'memory':final_memory,'results':records,
             'limitations':['Labels come from dataset pools, not the input classifier. This is oracle-label cache evaluation.',
                            'Calibration train fragments and val evaluation fragments are disjoint; previous training exposure is unknown.',
                            'No repeated trials or confidence intervals; latency is a pilot with cold and warm cache transitions.',
                            'Expert labels overlap, are per layer, and are routing associations, not proof of semantic knowledge.',
                            'Exact comparison checks routes, all token argmax, NLL and 64 vocabulary probes per token, not every logit.']}
    (args.output/'report.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    write_summary(args.output,payload,mapping,curves,global_curves)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
