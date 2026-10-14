"""Assistant-only OASST fine-tuning of the domain model, preserving fixed pools."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from asi.experiments.checkpoint_eval import load_benchmarks
from asi.experiments.public_moe import sha256
from asi.experiments.recovery_protocol import durable_save
from asi.models.domain import GPT, GPTConfig, Gate
from asi.runtime.generation import inference_context, validate_pool_identity
from asi.runtime.routing import DomainSessionRouter


def assistant_loss(logits, targets, mask):
    if not mask.sum(): raise ValueError('No supervised assistant tokens')
    losses=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),targets.reshape(-1),reduction='none')
    return (losses*mask.reshape(-1)).sum()/mask.sum()


def coverage_selection(pools, pool_count, limit, minimum):
    """Reserve unique examples for every pool, then fill in original shuffled order."""
    reserved=[]
    for pool in range(pool_count):
        ids=[i for i,value in enumerate(pools) if value==pool]
        if len(ids)<minimum: raise ValueError(f'Pool {pool} has only {len(ids)} examples; needs {minimum}')
        reserved.extend(ids[:minimum])
    chosen=set(reserved)
    target=limit or len(pools)
    if len(chosen)>target: raise ValueError('Example limit is smaller than coverage minimum')
    for i in range(len(pools)):
        if len(chosen)>=target:break
        chosen.add(i)
    return sorted(chosen)


def prepare(a):
    if a.data.exists(): raise ValueError('Use a fresh data directory')
    external=load_benchmarks(a.gpt2_sft)
    manifest=json.loads(a.pool_manifest.read_text(encoding='utf-8'))
    names=list(manifest['pools'])
    router=DomainSessionRouter(names,max_pools=1,session_inertia=0,pool_manifest=manifest)
    enc=tiktoken.get_encoding('gpt2')
    splits={s:external.load_oasst_split(s) for s in ['train','validation']}
    examples={s:external.build_examples(d) for s,d in splits.items()}
    roots={s:{r[0]['message_id'] for r in rows} for s,rows in examples.items()}
    if roots['train'] & roots['validation']: raise ValueError('Conversation roots overlap across splits')
    metadata={'pool_names':names,'block_size':a.seq_len,'seed':a.seed,
        'source':'OpenAssistant/oasst1','fingerprints':{s:d._fingerprint for s,d in splits.items()},
        'encoder_script_sha256':sha256(a.gpt2_sft),'pool_manifest_sha256':sha256(a.pool_manifest),
        'routing':'one frozen predicted pool per conversation, user messages only',
        'min_examples_per_pool':a.min_examples_per_pool,
        'truncation':'Same left truncation as GPT-2 SFT; may truncate user context for long answers',
        'splits':{}}
    packed={}
    for split,rows in examples.items():
        random.Random(a.seed).shuffle(rows)
        limit=a.train_examples if split=='train' else a.val_examples
        if limit and not a.min_examples_per_pool: rows=rows[:limit]
        arrays=[[],[],[],[]]; records=[]
        counts=Counter()
        for index,messages in enumerate(rows):
            encoded=external.encode_conversation(messages,enc,a.seq_len)
            if encoded is None or not encoded[2].sum(): continue
            # Never classify the target answer. No classifier updates on validation.
            prompt='\n'.join(m['text'] for m in messages if m['role']=='prompter')
            pool=names.index(router.route(prompt).ranked_pools[0])
            for dest,value in zip(arrays[:3],encoded): dest.append(value)
            arrays[3].append(pool)
            counts[pool]+=1
            records.append({'id':messages[-1]['message_id'],'root':messages[0]['message_id'],'pool':names[pool]})
            if index%50==0: print(split,index,len(rows),flush=True)
            if limit and len(records)>=limit and all(counts[i]>=a.min_examples_per_pool for i in range(len(names))):break
        if not records: raise ValueError('Empty supervised split')
        selected=coverage_selection(arrays[3],len(names),limit,a.min_examples_per_pool)
        scanned=len(records)
        arrays=[[values[i] for i in selected] for values in arrays]
        records=[records[i] for i in selected]
        packed[split]=dict(tokens=np.stack(arrays[0]),targets=np.stack(arrays[1]),masks=np.stack(arrays[2]),pools=np.array(arrays[3]))
        metadata['splits'][split]={'examples':len(records),'classified_candidates':scanned,'supervised_tokens':int(packed[split]['masks'].sum()),
            'pool_counts':dict(Counter(r['pool'] for r in records)),'records':records}
    a.data.mkdir(parents=True)
    for split,values in packed.items():
        path=a.data/(split+'.npz');np.savez(path,**values)
        metadata['splits'][split]['sha256']=sha256(path)
    durable_save(a.data/'manifest.json',metadata)
    print(json.dumps({s:{k:v for k,v in x.items() if k!='records'} for s,x in metadata['splits'].items()},indent=2))


def configure_trainable(model, router_only=False):
    if router_only:
        model.requires_grad_(False)
        model.config.routing_mode = 'learned'
    for gate in (m for m in model.modules() if isinstance(m, Gate)):
        if router_only:
            gate.routing_mode = 'learned'
        for param in gate.parameters():
            param.requires_grad_(router_only)


def train(a):
    if a.output.exists(): raise ValueError('Use a fresh output directory; base checkpoint is never overwritten')
    meta=json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    splits={}
    for s in ['train','validation']:
        path=a.data/(s+'.npz')
        if sha256(path)!=meta['splits'][s]['sha256']: raise ValueError('SFT data changed')
        with np.load(path) as d: splits[s]={k:d[k] for k in d.files}
    torch.manual_seed(a.seed)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=GPTConfig(**{**ckpt['config'],'max_batch_size':1})
    if config.routing_mode!='uniform_pool': raise ValueError('This experiment requires the fixed-pool model')
    if meta['block_size']>config.block_size: raise ValueError('Data exceeds checkpoint context')
    model=GPT(config)
    model.load_state_dict(ckpt['model'])
    validate_pool_identity(model,ckpt,meta['pool_names'])
    base_step=ckpt['step'];del ckpt
    configure_trainable(model, a.router_only)
    for layer in model.layers:
        if hasattr(layer.ffn,'moe_backend'): layer.ffn.moe_backend='bmm'
    model.to(a.device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,betas=(.9,.95),weight_decay=.01)
    rng=random.Random(a.seed); order=list(range(len(splits['train']['pools'])));rng.shuffle(order);cursor=0
    pool_orders={pool:[i for i in order if int(splits['train']['pools'][i])==pool] for pool in range(len(meta['pool_names']))}
    if a.sampling=='uniform-pool' and any(not ids for ids in pool_orders.values()):
        raise ValueError('Uniform-pool sampling requires train examples in every pool')
    pool_cursors=dict.fromkeys(pool_orders,0)
    exposures=Counter()
    a.output.mkdir(parents=True)
    report={'kind':'domain_router_oasst' if a.router_only else 'domain_oasst_sft','completed':False,'source_checkpoint':str(a.checkpoint),
        'source_sha256':sha256(a.checkpoint),'base_step':base_step,'data_manifest_sha256':sha256(a.data/'manifest.json'),
        'settings':vars(a),'pool_names':meta['pool_names'],'updates':[], 'validation':[],
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'selection':'minimum assistant-token weighted validation NLL, including step 0',
        'routing':('learned top-2 across all pools; only gate parameters trained' if a.router_only else 'unchanged uniform_pool; backbone/shared and selected expert weights trained, gate frozen')}
    # vars contains paths; normalize once for portable checkpoint metadata.
    report['settings']=json.loads(json.dumps(report['settings'],default=str))
    def row(split,index):
        data=splits[split]; model.set_active_pools(list(range(config.n_pools)) if a.router_only else [int(data['pools'][index])])
        # Remove right padding only; preserve exactly the GPT-2 target mask.
        end=int(np.flatnonzero(data['masks'][index])[-1])+1
        return [torch.tensor(data[k][index:index+1,:end].astype(np.int64),device=a.device) for k in ['tokens','targets','masks']]
    def evaluate():
        model.eval(); total=0.; count=0;pool_total=Counter();pool_tokens=Counter()
        with torch.no_grad():
            for i in range(len(splits['validation']['pools'])):
                x,y,m=row('validation',i)
                with inference_context(a.device): logits,_=model(x)
                n=int(m.sum());weighted=float(assistant_loss(logits,y,m))*n;total+=weighted;count+=n
                pool=int(splits['validation']['pools'][i]);pool_total[pool]+=weighted;pool_tokens[pool]+=n
        model.train();return total/count,{meta['pool_names'][p]:pool_total[p]/pool_tokens[p] for p in pool_tokens}
    def save_weights(name,step,loss):
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        durable_save(a.output/name,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'config':asdict(config),'step':step,'val_loss':loss,'pool_names':meta['pool_names'],
            'experiment':report['kind'],'sft_provenance':{k:report[k] for k in ['source_sha256','base_step','data_manifest_sha256','settings','routing']}},tensor=True)
    best,by_pool=evaluate();best_step=0;bad=0
    report['validation'].append({'step':0,'nll':best,'by_pool':by_pool});save_weights('best.pt',0,best)
    started=time.perf_counter()
    for step in range(1,a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        batch=[]
        for _ in range(a.accumulation):
            if a.sampling=='uniform-pool':
                pool=rng.randrange(len(pool_orders));ids=pool_orders[pool]
                if pool_cursors[pool]==len(ids):rng.shuffle(ids);pool_cursors[pool]=0
                index=ids[pool_cursors[pool]];pool_cursors[pool]+=1
            else:
                if cursor==len(order):rng.shuffle(order);cursor=0
                index=order[cursor];cursor+=1
            batch.append(index);exposures[meta['pool_names'][int(splits['train']['pools'][index])]]+=1
        # Token weighting across microbatches, not equal weighting of answers.
        denominator=sum(int(splits['train']['masks'][i].sum()) for i in batch)
        loss_sum=0.
        for i in batch:
            x,y,m=row('train',i)
            with inference_context(a.device): logits,_=model(x)
            loss=assistant_loss(logits,y,m)*m.sum()/denominator
            loss.backward();loss_sum+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not math.isfinite(loss_sum) or not torch.isfinite(norm): raise ValueError('Nonfinite SFT gradient/loss')
        optimizer.step()
        report['updates'].append({'step':step,'loss':loss_sum,'example_indices':batch})
        if step%a.eval_interval==0 or step==a.steps:
            val,by_pool=evaluate();report['validation'].append({'step':step,'nll':val,'by_pool':by_pool})
            if val<best:
                best=val;best_step=step;bad=0;save_weights('best.pt',step,val)
            else: bad+=1
            save_weights('last.pt',step,val)
            report.update(best_step=best_step,best_val_nll=best,steps_completed=step,
                effective_epochs=step*a.accumulation/len(order),seconds=time.perf_counter()-started)
            report['pool_example_exposures']=dict(exposures)
            durable_save(a.output/'report.json',report)
            print(f'step={step} train={loss_sum:.4f} val={val:.4f} best={best_step}',flush=True)
            if a.patience and bad>=a.patience: break
    report['completed']=True;durable_save(a.output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train'])
    p.add_argument('--gpt2-sft',type=Path,default=Path('../Creating-GPT-2-From-0/finetune_oasst.py'))
    p.add_argument('--data',type=Path,default=Path('data/domain_sft_oasst_pilot'))
    p.add_argument('--pool-manifest',type=Path,default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--checkpoint',type=Path,default=Path('model_37595.pt'))
    p.add_argument('--output',type=Path,default=Path('results/domain_sft_oasst_pilot'))
    p.add_argument('--seq-len',type=int,default=256)
    p.add_argument('--train-examples',type=int,default=1024,help='0 = full split')
    p.add_argument('--val-examples',type=int,default=128,help='0 = full split')
    p.add_argument('--min-examples-per-pool',type=int,default=0,help='Reserve coverage within each original split; may classify extra candidates')
    p.add_argument('--router-only',action='store_true',help='Train only E across all pools; freeze every other weight')
    p.add_argument('--sampling',choices=['sequential','uniform-pool'],default='sequential')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--eval-interval',type=int,default=50)
    p.add_argument('--patience',type=int,default=3,help='Validation checks without improvement; 0 disables')
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.steps,a.accumulation,a.eval_interval,a.seq_len,a.lr)<=0 or min(a.train_examples,a.val_examples,a.patience,a.min_examples_per_pool)<0:
        p.error('Invalid nonpositive settings')
    (prepare if a.action=='prepare' else train)(a)


if __name__=='__main__':main()
