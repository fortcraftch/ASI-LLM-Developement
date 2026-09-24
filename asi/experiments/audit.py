"""Reproducible prompt audit: routing, domain ablation, RAM/GPU residency and NLL."""
import argparse
import json
import time
from pathlib import Path
from asi import ROOT
from dataclasses import asdict
import torch
import tiktoken
from asi.models.domain import GPT, GPTConfig
from asi.runtime.routing import DomainSessionRouter
from asi.runtime.cache import ExpertCacheManager, module_memory
from asi.runtime.generation import load_model, load_manifest, validate_pool_identity, inference_context, generate
from asi.analysis.experts import RoutingTrace
from asi.analysis.experts import ExpertCalibrator
from contextlib import nullcontext


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path)
    source.add_argument('--smoke',action='store_true',help='Random tiny model; NOT evidence of specialization')
    p.add_argument('--pool-manifest',type=Path)
    p.add_argument('--prompts',type=Path,default=(ROOT / 'examples/prompts.jsonl'))
    p.add_argument('--output',type=Path,default=Path('results/expert_audit'))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--modes',nargs='+',choices=['restricted','unrestricted','ablated'],default=['restricted','unrestricted','ablated'])
    p.add_argument('--max-pools',type=int,default=3)
    p.add_argument('--max-hot-pools',type=int,default=3)
    p.add_argument('--pin-memory',action='store_true')
    p.add_argument('--max-pinned-mib',type=int,default=512)
    p.add_argument('--no-classifier',action='store_true')
    p.add_argument('--classifier-device',default='cpu')
    p.add_argument('--classifier-model',default='mdonigian/fineweb-edu-topic-classifier')
    p.add_argument('--max-new-tokens',type=int,default=8)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--export-router-vectors',action='store_true')
    a=p.parse_args()
    torch.manual_seed(a.seed)
    if a.smoke:
        names=json.loads((ROOT / 'configs/domain_experts.json').read_text())['pool_order']
        model=GPT(GPTConfig(block_size=128,max_seq_len=128,original_seq_len=128,max_batch_size=1,
                            n_layer=3,n_head=2,n_embd=32,inter_dim=64,moe_inter_dim=16,
                            kv_lora_rank=16,qk_nope_head_dim=8,qk_rope_head_dim=4,v_head_dim=8)).eval()
        metadata={'experiment':'RANDOM_SMOKE_ONLY','pool_names':names}
    else:
        if not a.pool_manifest:
            p.error('--pool-manifest is required with --checkpoint')
        model,metadata=load_model(a.checkpoint)
        names,_=load_manifest(a.pool_manifest)
        validate_pool_identity(model,metadata,names)
    name_to_id={name:i for i,name in enumerate(names)}
    samples=[json.loads(line) for line in a.prompts.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    enc=tiktoken.get_encoding('gpt2')
    a.output.mkdir(parents=True,exist_ok=True)
    router=DomainSessionRouter(names,device=a.classifier_device,max_pools=a.max_pools,
                               classifier_model=a.classifier_model,load_classifier=not (a.no_classifier or a.smoke))
    results=[]
    calibration=ExpertCalibrator(model)
    for mode in a.modes:
        capacity=a.max_hot_pools if mode=='restricted' else len(names)
        cache=ExpertCacheManager(a.device,capacity,a.pin_memory,a.max_pinned_mib*1024**2)
        cache.initialize(model)
        router.previous_pools=[]
        # Warm CUDA kernels and allocator, then start cold expert residency counters.
        cache.prepare(model,[0]); model.set_active_pools([0])
        with torch.no_grad(),inference_context(a.device):
            model(torch.tensor([[enc.eot_token,enc.eot_token]],device=a.device))
        cache.offload_all(model)
        pinned=cache.stats['pinned_bytes']; cache.stats.clear(); cache.stats['pinned_bytes']=pinned
        for index,sample in enumerate(samples):
            prompt=sample['prompt']
            # Classification sees the input only; completion is held out for scoring.
            start=time.perf_counter(); route=router.route(prompt); routing_seconds=time.perf_counter()-start
            selected=[name_to_id[name] for name in sample.get('pools',route.ranked_pools)]
            expected=[name_to_id[name] for name in sample.get('expected_pools',[])]
            if mode=='restricted': active=selected
            elif mode=='unrestricted': active=list(range(len(names)))
            else: active=[i for i in range(len(names)) if i not in selected]
            if len(active)*model.config.experts_per_pool<model.config.n_activated_experts:
                results.append({'sample':index,'mode':mode,'skipped':'Ablation leaves fewer than top-k experts'})
                continue
            before=cache.snapshot()
            cache.prepare(model,active); model.set_active_pools(active)
            after=cache.snapshot()
            prompt_ids=enc.encode(prompt)
            completion_ids=enc.encode(sample.get('completion',''))
            ids=(prompt_ids+completion_ids)[:model.config.block_size+1]
            if len(ids)<2:
                raise ValueError('Every sample must have at least two tokens')
            x=torch.tensor([ids[:-1]],device=a.device)
            y=torch.tensor([ids[1:]],device=a.device)
            if torch.device(a.device).type=='cuda': torch.cuda.reset_peak_memory_stats(a.device)
            sync(a.device); start=time.perf_counter()
            with torch.no_grad(),inference_context(a.device):
                logits,_=model(x)
            sync(a.device); forward_seconds=time.perf_counter()-start
            losses=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.flatten(),reduction='none')
            offset=max(0,len(prompt_ids)-1) if completion_ids else 0
            if offset>=len(losses):
                raise ValueError('Completion falls outside context window; shorten the prompt')
            nll=float(losses[offset:].mean())
            del logits,losses
            # Trace in a separate pass: synchronized callbacks would distort latency.
            with RoutingTrace(model,names) as trace,torch.no_grad(),inference_context(a.device):
                calibration.set_labels(sample.get('expected_pools') or [sample.get('label','unlabeled')])
                with calibration if mode=='unrestricted' else nullcontext():
                    model(x)
                text=generate(model,enc,prompt,a.device,a.max_new_tokens,1.0,1) if a.max_new_tokens else ''
            trace_path=a.output/f'{mode}_{index:03d}_routes.json'
            trace.write(trace_path)
            report=trace.report()
            violations=sum(layer['mask_violations'] for layer in report['layers'].values())
            selections=sum(sum(layer['selections']) for layer in report['layers'].values())
            outside_expected=sum(count for layer in report['layers'].values()
                                 for eid,count in enumerate(layer['selections'])
                                 if eid//model.config.experts_per_pool not in expected)
            memory=cache.inventory()
            result={'sample':index,'label':sample.get('label','unlabeled'),'mode':mode,
                    'prompt':prompt,'selected_pools':[names[i] for i in selected],
                    'classifier_pools':route.ranked_pools,'expected_pools':[names[i] for i in expected],
                    'active_pools':[names[i] for i in active], 'nll':nll,
                    'scored_tokens':len(ids)-1-offset,'score_region':'completion' if completion_ids else 'prompt',
                    'routing_seconds':routing_seconds,'forward_seconds_without_trace':forward_seconds,
                    'cache_delta':{k:after.get(k,0)-before.get(k,0) for k in ['hits','misses','evictions','host_to_device_bytes','prepare_seconds']},
                    'cache':after,'memory':memory,'mask_violations':violations,
                    'outside_expected_pool_fraction':outside_expected/selections if expected and selections else None,
                    'generated':text,'trace_file':trace_path.name}
            results.append(result)
            print(f'{mode} sample={index} NLL={nll:.4f} violations={violations} cache={result["cache_delta"]}')
            if violations: raise RuntimeError('Forbidden expert selected; see routing trace')
        cache.offload_all(model)
    aggregates={}
    for mode in a.modes:
        rows=[r for r in results if r['mode']==mode and 'nll' in r]
        total=sum(r['scored_tokens'] for r in rows)
        aggregates[mode]={'scored_tokens':total,'token_weighted_nll':sum(r['nll']*r['scored_tokens'] for r in rows)/total if total else None}
    (a.output/'report.json').write_text(json.dumps({'metadata':metadata,'checkpoint':str(a.checkpoint),
        'torch':torch.__version__,'device':a.device,'gpu':torch.cuda.get_device_name(a.device) if torch.device(a.device).type=='cuda' else None,
        'config':asdict(model.config),'seed':a.seed,'classifier_enabled':router.classifier is not None,
        'classifier_memory':module_memory(router.classifier),
        'notes':'NLL on illustrative samples is not a benchmark. Shadow routes and depth edges are not causal proof. Unrestricted/ablated use larger cache capacity. Forward timing excludes classification, transfers and tracing.',
        'aggregate':aggregates,'results':results},indent=2),encoding='utf-8')
    if 'unrestricted' in a.modes:
        calibration.write(a.output/'calibration.json')
        (a.output/'proposed_expert_labels.json').write_text(
            json.dumps(calibration.propose_mapping(model.config.experts_per_pool),indent=2),encoding='utf-8')
    if a.export_router_vectors:
        calibration.export_router_vectors(a.output/'router_vectors')

if __name__=='__main__': main()
