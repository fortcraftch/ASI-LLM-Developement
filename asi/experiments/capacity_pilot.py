"""Bounded, paired capacity pilot on frozen local windows, not a language threshold study."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
import torch

from asi import DATA_ROOT
from asi.experiments.capacity_study import variants
from asi.experiments.train import PoolData, PoolManifest
from asi.experiments.recovery_protocol import durable_save
from asi.models.domain import GPT, GPTConfig, MoE
from asi.runtime.generation import inference_context


def normalize_learned_gates(model):
    """Experimental amplitude control; selected IDs and native logits stay unchanged."""
    for layer in model.layers:
        if not isinstance(layer.ffn,MoE) or layer.ffn.gate.routing_mode!='learned': continue
        gate=layer.ffn.gate
        original=gate.route_logits
        def normalized(self,logits,allowed_expert_mask=None,_original=original):
            weights,indices=_original(logits,allowed_expert_mask)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            return weights*self.route_scale,indices
        gate.route_logits=MethodType(normalized,gate)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool_schedule(count, steps, seed):
    if steps % count: raise ValueError('Steps must be a multiple of pool count for equal exposure')
    rng=random.Random(seed); result=[]
    for _ in range(steps//count):
        block=list(range(count));rng.shuffle(block);result.extend(block)
    return result


@torch.no_grad()
def initialize_paired(model, seed):
    """Same-shaped named parameters are identical across variants at initialization."""
    common=hashlib.sha256()
    for name,p in model.named_parameters():
        value=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator(device=p.device).manual_seed(value)
        if p.ndim<2:
            p.fill_(0 if name.endswith('bias') else 1)
        else:
            module=model.get_submodule(name.rsplit('.',1)[0])
            std=.02*((2*model.config.n_layer)**-.5 if hasattr(module,'NANOGPT_SCALE_INIT') else 1)
            p.normal_(0,std,generator=generator)
        if '.ffn.experts.' not in name and '.ffn.gate.' not in name:
            common.update(name.encode());common.update(p.cpu().numpy().tobytes())
    return common.hexdigest()


class Usage:
    def __init__(self, model):
        self.layers={str(i):layer.ffn for i,layer in enumerate(model.layers) if isinstance(layer.ffn,MoE)}
        self.tokens={i:torch.zeros(m.n_routed_experts,dtype=torch.long,device=m.gate.weight.device) for i,m in self.layers.items()}
        self.grad_steps={i:torch.zeros_like(v) for i,v in self.tokens.items()}
        self.current_pool=0
        self.by_pool={i:torch.zeros((model.config.n_pools,len(v)),device=v.device,dtype=torch.long) for i,v in self.tokens.items()}
        self.mass={i:torch.zeros((),device=v.device) for i,v in self.tokens.items()}
        self.observed={i:0 for i in self.tokens}
        for i,m in self.layers.items():
            def observe(gate,x,logits,weights,indices,mask,key=i):
                if gate.training:
                    counts=torch.bincount(indices.detach().flatten(),minlength=len(self.tokens[key]))
                    self.tokens[key].add_(counts)
                    self.by_pool[key][self.current_pool].add_(counts)
                    self.mass[key].add_(weights.detach().float().sum())
                    self.observed[key]+=indices.shape[0]
            m.gate.routing_observer=observe

    @torch.no_grad()
    def gradients(self):
        for i,m in self.layers.items():
            values=[]
            for expert in m.experts:
                grads=[p.grad for p in expert.parameters() if p.grad is not None]
                values.append(torch.stack([g.detach().abs().sum() for g in grads]).sum()>0 if grads else self.tokens[i].new_zeros((),dtype=torch.bool))
            self.grad_steps[i].add_(torch.stack(values).long())

    def report(self):
        return {i:{'token_assignments':self.tokens[i].cpu().tolist(),
                   'assignments_by_input_pool':self.by_pool[i].cpu().tolist(),
                   'mixture_weight_sum':float(self.mass[i]),'observed_tokens':self.observed[i],
                   'mean_selected_weight_mass':float(self.mass[i])/max(1,self.observed[i]),
                   'steps_with_nonzero_gradient':self.grad_steps[i].cpu().tolist(),
                   'unused_experts':int((self.tokens[i]==0).sum())} for i in self.layers}


def freeze_data(a, names):
    manifest=PoolManifest(a.pool_manifest)
    data_seed=getattr(a,'data_seed',None)
    if data_seed is None:data_seed=a.seed
    schedule=pool_schedule(len(names),a.steps,data_seed)
    arrays={}; provenance={}
    for split in ('train','val'):
        loader=PoolData(a.data_root,manifest,split,a.batch_size,a.seq_len,'random',data_seed+(100000 if split=='val' else 0))
        if loader.names()!=names:raise ValueError(f'Every pool must have {split} windows')
        pairs=[]
        pools=([p for p in schedule for _ in range(a.accumulation)] if split=='train'
               else [p for p in range(len(names)) for _ in range(a.val_batches)])
        for pool in pools:pairs.append(loader.next_batch(names[pool]))
        arrays[split+'_x']=np.stack([x.numpy().astype(np.uint16) for x,_ in pairs])
        arrays[split+'_y']=np.stack([y.numpy().astype(np.uint16) for _,y in pairs])
        arrays[split+'_pools']=np.array(pools)
        provenance[split]={name:stream.identity for name,stream in loader.streams.items()}
    a.output.mkdir(parents=True)
    np.savez(a.output/'windows.npz',**arrays)
    return {'windows_sha256':digest(a.output/'windows.npz'),'shards':provenance,'schedule':schedule}


def run_variant(a,row,arrays,names,output):
    config=GPTConfig(**{**row['model'],'block_size':a.seq_len,'max_seq_len':a.seq_len,
        'original_seq_len':a.seq_len,'max_batch_size':a.batch_size})
    torch.manual_seed(a.seed)
    model=GPT(config)
    if getattr(a,'normalize_selected',False): normalize_learned_gates(model)
    common_hash=initialize_paired(model,a.seed)
    for layer in model.layers:
        if isinstance(layer.ffn,MoE):layer.ffn.moe_backend='bmm'
    model.to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1)
    usage=Usage(model)
    def activate(pool):
        usage.current_pool=pool
        model.set_active_pools([pool] if config.routing_mode=='uniform_pool' else list(range(len(names))))
    def batch(split,i):
        activate(int(arrays[split+'_pools'][i]))
        return [torch.tensor(arrays[split+'_'+k][i].astype(np.int64),device=a.device) for k in ('x','y')]
    @torch.no_grad()
    def validate():
        model.eval();losses={n:[] for n in names}
        for i,p in enumerate(arrays['val_pools']):
            x,y=batch('val',i)
            with inference_context(a.device):_,loss=model(x,y)
            losses[names[int(p)]].append(float(loss))
        return {n:sum(v)/len(v) for n,v in losses.items()}
    def sync():
        if a.device.startswith('cuda'):torch.cuda.synchronize()
    report={'variant':row,'status':'running','kind':'short_optimization_pilot_not_language_quality',
            'normalize_selected_learned_weights':bool(getattr(a,'normalize_selected',False)),
            'common_initial_weights_sha256':common_hash,'updates':[],
            'parameters':sum(p.numel() for p in model.parameters()),'validation':[],
            'usage_history':[]}
    before=validate();report['validation'].append({'updates':0,'by_pool':before,'mean':sum(before.values())/len(names)})
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        model.train();optimizer.zero_grad(set_to_none=True);sync();started=time.perf_counter();total=0.
        lr=a.lr*min(1.,(step+1)/max(1,a.warmup_steps))
        for group in optimizer.param_groups:group['lr']=lr
        for micro in range(a.accumulation):
            x,y=batch('train',step*a.accumulation+micro)
            with inference_context(a.device):_,loss=model(x,y)
            (loss/a.accumulation).backward();total+=float(loss.detach())/a.accumulation
        usage.gradients()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not np.isfinite(total) or not torch.isfinite(norm):raise ValueError('Nonfinite pilot loss/gradient')
        optimizer.step();sync();elapsed=time.perf_counter()-started
        report['updates'].append({'update':step+1,'train_nll':total,'seconds':elapsed,'lr':lr,
            'pool':names[int(arrays['train_pools'][step*a.accumulation])]})
        interval=getattr(a,'eval_interval',None) or max(1,a.steps//2)
        if (step+1)%interval==0 or step+1==a.steps:
            values=validate();mean=sum(values.values())/len(names)
            report['validation'].append({'updates':step+1,'by_pool':values,'mean':mean})
            report['usage_history'].append({'updates':step+1,'layers':usage.report()})
            durable_save(output,report)
            print(f"{row['name']} step={step+1} val={mean:.4f}",flush=True)
    tokens=a.steps*a.accumulation*a.batch_size*a.seq_len
    seconds=sum(s['seconds'] for s in report['updates'])
    report.update(status='completed',tokens=tokens,training_seconds=seconds,tokens_per_second=tokens/seconds,
                  expert_usage=usage.report(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None)
    durable_save(output,report)
    del optimizer,usage,model;gc.collect()
    if a.device.startswith('cuda'):torch.cuda.empty_cache()
    return report


def summarize(output,plan):
    reports=[]
    lines=['# Piloto de capacidad: viabilidad, no umbral de lenguaje','',
        'Ventanas fijas y exposición equilibrada. Modelos inicializados desde cero; no son checkpoints recuperados.',
        'El corpus local es distinto del manifiesto de homelab. No mezclar estas pérdidas con el entrenamiento principal.','',
        '| Variante | Parámetros | Tokens | NLL inicial | NLL final | Tokens/s entrenamiento | Pico CUDA MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in plan['variants']:
        p=output/(name+'.json')
        if not p.exists():continue
        r=json.loads(p.read_text(encoding='utf8'))
        if r['status']!='completed':continue
        reports.append(r)
        peak=f"{r['peak_cuda_bytes']/2**20:.1f}" if r['peak_cuda_bytes'] is not None else 'N/A'
        lines.append(f"| {name} | {r['parameters']} | {r['tokens']} | {r['validation'][0]['mean']:.4f} | {r['validation'][-1]['mean']:.4f} | {r['tokens_per_second']:.0f} | {peak} |")
    lines+=['','Tiempo incluye telemetría de routing/gradientes y transferencia de ventanas; excluye inicialización y validación. No es rendimiento de producción.',
             'No elegir una arquitectura por este piloto corto. La comparación científica requiere preentrenamiento suficiente, varias semillas y benchmarks reservados.']
    if reports:
        same=len({r['common_initial_weights_sha256'] for r in reports})==1
        lines+=['',f'Inicialización de parámetros comunes idéntica entre variantes completadas: {same}.',
                '','## Uso de expertos durante entrenamiento','',
                '| Variante | Expertos sin tokens (suma de capas) | Expertos efectivos por capa (media) |',
                '|---|---:|---:|']
        for r in reports:
            effective=[]
            for layer in r['expert_usage'].values():
                counts=np.array(layer['token_assignments'],dtype=float);prob=counts/counts.sum()
                effective.append(float(np.exp(-(prob[prob>0]*np.log(prob[prob>0])).sum())))
            lines.append(f"| {r['variant']['name']} | {sum(x['unused_experts'] for x in r['expert_usage'].values())} | {np.mean(effective):.2f} |")
        lines+=['','El número efectivo resume el reparto de tokens: coincide con el total si todos reciben lo mismo y se aproxima a uno si domina un experto. No mide conocimiento ni calidad.',
                'Los JSON conservan los tokens y los pasos con gradiente no nulo de cada experto. Cada token cuenta tantas asignaciones como expertos activos.']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,axis in zip(axes,('capacity','granularity')):
            for routing,label in [('uniform_pool','Pool restringido'),('learned','Routing libre')]:
                points=[]
                for r in reports:
                    c=r['variant']['model']
                    if c['routing_mode']!=routing:continue
                    if axis=='capacity' and c['n_activated_experts']!=2:continue
                    if axis=='granularity' and c['moe_inter_dim']*c['n_activated_experts']!=240:continue
                    x=(c['moe_inter_dim'] if axis=='capacity' else c['n_activated_experts'])
                    points.append((x,r['validation'][-1]['mean']))
                points.sort()
                if points:ax.plot([v[0] for v in points],[v[1] for v in points],'-o',label=label)
            ax.set_xlabel('Ancho de experto' if axis=='capacity' else 'Expertos activos por capa')
            ax.set_xticks([60,120,240] if axis=='capacity' else [1,2,4])
            ax.set_ylabel('NLL de validación (menor es mejor)');ax.set_title('Capacidad' if axis=='capacity' else 'Granularidad');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f"Piloto corto: {plan['settings']['steps']} actualizaciones; no mide calidad del lenguaje")
        fig.tight_layout();fig.savefig(output/'capacity_pilot.png',dpi=160);plt.close(fig)
        lines+=['','![Capacidad y granularidad](capacity_pilot.png)']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=Path('configs/experiment_124m_v1/domain_16_uniform_pool.json'))
    p.add_argument('--pool-manifest',type=Path,default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--data-root',type=Path,default=DATA_ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--val-batches',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--warmup-steps',type=int,default=4)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--data-seed',type=int,default=None,help='Fix windows independently of initialization seed')
    p.add_argument('--eval-interval',type=int,default=None)
    p.add_argument('--variants',nargs='+',default=None,help='Run only these exact variant names')
    p.add_argument('--normalize-selected',action='store_true',help='Experimental control: renormalize selected learned gate weights without changing top-k')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true',help='Skip completed variants; restart interrupted variant from identical initialization')
    p.add_argument('--report-only',action='store_true',help='Rebuild report and plot without loading or training model weights')
    a=p.parse_args()
    if a.report_only:
        summarize(a.output,json.loads((a.output/'plan.json').read_text(encoding='utf8')));return
    if min(a.steps,a.batch_size,a.seq_len,a.accumulation,a.val_batches,a.lr)<1e-12 or a.warmup_steps<0:p.error('Invalid pilot settings')
    if a.eval_interval is not None and a.eval_interval<1:p.error('Positive evaluation interval required')
    source=json.loads(a.recipe.read_text(encoding='utf8')); names=source['pool_order']
    if PoolManifest(a.pool_manifest).names!=names:p.error('Pool order mismatch')
    rows=[];seen=set()
    for row in variants(source):
        key=json.dumps(row['model'],sort_keys=True)
        if key not in seen:rows.append(row);seen.add(key)
    if a.variants:
        if set(a.variants)-{r['name'] for r in rows}:p.error('Unknown variant name')
        rows=[r for r in rows if r['name'] in a.variants]
    settings={k:v for k,v in vars(a).items() if k not in ('resume','output','report_only')}
    settings=json.loads(json.dumps(settings,default=str))
    if not settings.get('normalize_selected'): settings.pop('normalize_selected',None)
    for key in ('data_seed','eval_interval','variants'):
        if settings.get(key) is None:settings.pop(key,None)
    identity={'settings':settings,'source_recipe_sha256':digest(a.recipe),'pool_manifest_sha256':digest(a.pool_manifest),
        'local_manifest_sha256':digest(a.data_root/'manifest.json'),'variants':[r['name'] for r in rows]}
    if a.resume:
        plan=json.loads((a.output/'plan.json').read_text(encoding='utf8'))
        if any(plan[k]!=v for k,v in identity.items()):p.error('Pilot identity/settings changed')
        if digest(a.output/'windows.npz')!=plan['data']['windows_sha256']:p.error('Frozen windows changed')
    else:
        if a.output.exists():p.error('Use fresh output or --resume')
        plan={**identity,'status':'running','data':freeze_data(a,names),
              'reference_manifest_matches':identity['local_manifest_sha256']==source['source_manifest_sha256'],
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
              'limitation':'Small local optimization pilot; not a language-capacity threshold. No weights retained.'}
        durable_save(a.output/'plan.json',plan)
    with np.load(a.output/'windows.npz') as data:arrays={k:data[k] for k in data.files}
    for row in rows:
        output=a.output/(row['name']+'.json')
        if output.exists() and json.loads(output.read_text(encoding='utf8'))['status']=='completed':continue
        run_variant(a,row,arrays,names,output);summarize(a.output,plan)
    plan['status']='completed';durable_save(a.output/'plan.json',plan)
    summarize(a.output,plan)
    print(a.output/'summary.md')


if __name__=='__main__':main()
