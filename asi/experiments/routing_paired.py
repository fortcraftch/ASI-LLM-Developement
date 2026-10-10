"""Paired teacher-forced routing diagnostic; fresh process, no optimizer."""
import argparse
import json
from pathlib import Path
import time
import torch
from asi.taxonomy import digest
from asi.experiments.public_moe import read_lock, sha256, input_ids
from asi.experiments.recovery import restore_routers, router_state
from asi.experiments.recovery_protocol import durable_save
from asi.models.olmoe_storage import load_disk_backed
from asi.models.hf_olmoe import OlmoeExpertCache
from asi.models.recovery_lora import ExpertLoRA


def summarize(indices, weights, probabilities, allowed, active):
    """Token-normalized summaries; top-k exclusion is measured before masking."""
    ids=torch.as_tensor(allowed,device=probabilities.device)
    top=probabilities.topk(active+1,dim=-1)
    included=torch.isin(top.indices[:,:active],ids)
    return dict(tokens=len(indices),counts=torch.bincount(indices.flatten(),minlength=probabilities.shape[-1]).cpu().tolist(),
        mean_probabilities=probabilities.mean(0).cpu().tolist(),
        mean_executed_mass=weights.float().sum(-1).mean().item(),
        outside_topk_fraction=(~included).float().mean().item(),
        outside_topk_mass=(top.values[:,:active]*(~included)).sum(-1).mean().item(),
        outside_total_mass=(1-probabilities[:,ids].sum(-1)).mean().item(),
        mean_topk_margin=(top.values[:,active-1]-top.values[:,active]).mean().item())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=Path('results/recovery_200_screened_20261008'))
    p.add_argument('--output',type=Path,default=Path('results/routing_paired_20261009'))
    p.add_argument('--model-dir',type=Path,default=Path('public_models/olmoe'))
    p.add_argument('--per-category',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.resume:p.error('Use a fresh output directory or --resume')
    if args.per_category<1:p.error('Positive per-category required')
    plan=json.loads((args.pilot/'plan.json').read_text(encoding='utf-8'))
    for path,expected in plan['source_hashes'].items():
        if sha256(Path(path))!=expected:raise ValueError('Pilot source changed: '+path)
    corpus_path=Path('results/recovery_corpus_v3/corpus.json')
    corpus=json.loads(corpus_path.read_text(encoding='utf-8'))
    study_path=Path('results/recovery_calibration_batched_v3/report.json')
    study=json.loads(study_path.read_text(encoding='utf-8'));mapping=study['mappings']['32']
    lock=read_lock(args.model_dir)
    # Recovery source identity uses the complete verified source lock.
    source=digest(lock)
    if study['source_identity']!=source:raise ValueError('Source/calibration mismatch')
    excluded=set(study['train_ids'])|set(plan.get('excluded_ids',[]))
    pilot_ids=json.loads((args.pilot/'ids.json').read_text(encoding='utf-8'))
    if isinstance(pilot_ids,dict):
        excluded.update(i for v in pilot_ids.values() if isinstance(v,list) for i in v)
    else:excluded.update(pilot_ids)
    model,tokenizer,store=load_disk_backed(args.model_dir,2*2**30,12*2**30)
    selected=[]
    for category in sorted(mapping):
        candidates=sorted((r for r in corpus['examples'] if r['split']=='test' and r['category']==category and r['id'] not in excluded),key=lambda r:r['id'])
        group=[]
        for row in candidates:
            prompt=input_ids(tokenizer,row['prompt'],limit=384)[0].tolist()
            tokens=prompt+tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(tokens)>384:continue
            group.append(dict(id=row['id'],category=category,tokens=tokens,prompt_length=len(prompt)))
            if len(group)==args.per_category:break
        if len(group)!=args.per_category:raise ValueError('Insufficient diagnostic examples')
        selected.extend(group)
    cache=OlmoeExpertCache(model,capacity=32*model.config.num_hidden_layers,store=store)
    # Preserve learned FP32 router updates exactly, matching recovery evaluation.
    for layer in model.model.layers:
        layer.mlp.gate.weight.data=layer.mlp.gate.weight.data.float()
    model.requires_grad_(False)
    initial=router_state(model)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    report=dict(completed=False,source_identity=source,corpus_sha256=sha256(corpus_path),mapping_sha256=sha256(study_path),
        protocol='Identical full prompt + reference answer + EOS tokens; batch 1; BF16; teacher forcing; 32 resident / 8 active. Hooks include CPU synchronization; timing is instrumentation cost, not inference benchmark.',
        examples=selected,records=[],checkpoint_hashes={},limitations=['Diagnostic reference text is not a quality target verified here.','Same tokens do not imply identical hidden states after restrictions/adaptation.','Unrestricted preferences in adapted arms are counterfactual only at their own hidden states.'])
    def save():durable_save(args.output/'report.json',report)
    if args.resume:
        previous=json.loads((args.output/'report.json').read_text(encoding='utf-8'))
        for key in ('source_identity','corpus_sha256','mapping_sha256','examples'):
            if previous[key]!=report[key]:raise ValueError('Resume mismatch: '+key)
        # Only original traces from the interrupted pre-adaptation run are reusable.
        report['records']=[r for r in previous['records'] if r['arm']=='original']
    save()
    hooks=[];current={};details={}
    for layer,block in enumerate(cache.blocks):
        def hook(module,inputs,output,lid=layer):
            logits,weights,indices=output
            probs=logits.detach().float().softmax(-1)
            entry=mapping[current['category']][str(lid)]
            details[str(lid)]=summarize(indices.detach(),weights.detach(),probs,entry['experts'],8)
            details[str(lid)]['indices']=indices.detach().cpu().tolist()
        hooks.append(block.gate.register_forward_hook(hook))
    try:
        for arm in ('original','restricted','ce','ce_kl'):
            restore_routers(model,initial);cache.expert_adapter=None
            if arm in ('ce','ce_kl'):
                path=args.pilot/arm/'adaptation_best.pt'
                state=torch.load(path,map_location='cpu',weights_only=True)
                if state['identity']['source']!=source or state['identity']['mapping']!=digest(mapping):raise ValueError('Adapter identity mismatch')
                adapter=ExpertLoRA(model,{'32':mapping},rank=8)
                adapter.load_state_dict(state['adapters'],strict=True)
                restore_routers(model,state['routers']);cache.expert_adapter=adapter
                report['checkpoint_hashes'][arm]=sha256(path)
                del state
            for row in selected:
                if any(r['arm']==arm and r['id']==row['id'] for r in report['records']):continue
                current.clear();current.update(row);details.clear()
                print(arm,row['id'],flush=True)
                cache.set_context(None if arm=='original' else mapping[row['category']],mixture='restricted',active=8)
                before=cache.snapshot();start=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(torch.tensor([row['tokens']],device='cuda'),use_cache=False)
                del output
                torch.cuda.synchronize()
                after=cache.snapshot()
                if arm!='original' and after.get('loads',0)!=before.get('loads',0):raise RuntimeError('Restricted forward loaded an expert')
                if len(details)!=model.config.num_hidden_layers:raise RuntimeError('Missing layer trace')
                report['records'].append(dict(arm=arm,id=row['id'],category=row['category'],layers=dict(details),instrumented_seconds=time.perf_counter()-start,cache_before=before,cache_after=after))
                save()
        report['completed']=True;save()
    finally:
        for handle in hooks:handle.remove()
        cache.close()


if __name__=='__main__':main()
