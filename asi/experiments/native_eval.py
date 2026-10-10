"""Evaluate the original free-routing DeepSeek checkpoint with the GPT-2 protocol."""
import argparse
import json
from pathlib import Path
import time

import tiktoken
import torch

from asi.experiments.checkpoint_eval import load_benchmarks
from asi.experiments.benchmark_batch import evaluate
from asi.experiments.recovery_protocol import durable_save
from asi.models.original import DEFAULT_ARCHITECTURE,load_original_model,file_sha256,native_moes
from asi.runtime.generation import inference_context
from asi.runtime.cache import module_memory


class NativeView(torch.nn.Module):
    """Compatibility view: original forward always re-prefills, no invented KV API."""
    def __init__(self,model):
        super().__init__();self.native=model;self.config=model.config
    def forward(self,x,start_pos=0,logits_to_keep=0):
        if start_pos:raise ValueError('Native architecture does not expose incremental decoding')
        logits,loss=self.native(x)
        return (logits[:,-logits_to_keep:] if logits_to_keep else logits),loss


@torch.no_grad()
def audit_routing(model,enc,prompts,device,new_tokens):
    """Unpadded probes, separated from benchmarks; hooks never replace the gate."""
    result=[]
    moes=native_moes(model)
    for prompt in prompts:
        ids=enc.encode(prompt)[-model.config.block_size:]
        if not ids:raise ValueError('Empty audit prompt')
        original=len(ids);steps=[];prefill={};current={};handles=[]
        for layer,moe in moes.items():
            def observe(gate,inputs,output,key=str(layer),total=moe.n_routed_experts):
                weights,indices=output
                if not steps:prefill[key]=torch.bincount(indices.flatten(),minlength=total).cpu().tolist()
                current[key]={'experts':indices[-1].detach().cpu().tolist(),
                              'weights':weights[-1].detach().float().cpu().tolist()}
            handles.append(moe.gate.register_forward_hook(observe))
        try:
            for _ in range(new_tokens):
                current.clear()
                with inference_context(device):logits,_=model(torch.tensor([ids[-model.config.block_size:]],device=device))
                token=int(logits[0,-1].argmax());ids.append(token)
                steps.append({'next_token':token,'last_position_routes':dict(current)})
                if token==enc.eot_token:break
        finally:
            for handle in handles:handle.remove()
        layers={}
        for layer,moe in moes.items():
            key=str(layer);counts=[0]*moe.n_routed_experts;changes=0;previous=None
            for step in steps:
                chosen=step['last_position_routes'][key]['experts']
                for i in chosen:counts[i]+=1
                selected=set(chosen)
                if previous is not None:changes+=selected!=previous
                previous=selected
            layers[key]={'resident_experts':moe.n_routed_experts,'activated_per_position':moe.gate.topk,
                'prefill_assignments':prefill.get(key,[0]*moe.n_routed_experts),
                'last_position_assignments':counts,'unique_experts_at_last_position':sum(c>0 for c in counts),
                'unique_experts_including_prefill':sum(c>0 or p>0 for c,p in zip(counts,prefill.get(key,[0]*moe.n_routed_experts))),
                'selection_changes_between_steps':changes}
        result.append({'prompt':prompt,'text':enc.decode(ids[original:]),'generated_tokens':len(steps),
                       'layers':layers,'steps':steps})
    return {'probes':result,'note':'All experts resident. Selection changes are NOT swaps. Prefill counts each input position once; generation counts only the final position of each forward, excluding recomputed prefixes. Shared experts are additional.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--architecture',type=Path,default=DEFAULT_ARCHITECTURE)
    p.add_argument('--gpt2-evaluator',type=Path,default=Path('../Creating-GPT-2-From-0/eval.py'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',default='hellaswag,lambada,piqa,arc_easy,boolq,arithmetic')
    p.add_argument('--limit',type=int,default=100,help='0 = full datasets; arithmetic defaults to 500')
    p.add_argument('--device',default='cuda')
    p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--batch-tokens',type=int,default=2048)
    p.add_argument('--reference-eval',action='store_true')
    p.add_argument('--audit-routing',action='store_true')
    p.add_argument('--audit-tokens',type=int,default=32)
    p.add_argument('--audit-prompts',type=Path,help='JSON array of nonempty strings')
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output file')
    if a.limit<0 or min(a.eval_batch_size,a.batch_tokens,a.audit_tokens)<1:p.error('Invalid sizes')
    bench=load_benchmarks(a.gpt2_evaluator)
    tasks=list(bench.TASKS) if a.tasks=='all' else a.tasks.split(',')
    if set(tasks)-set(bench.TASKS):p.error('Unknown benchmark')
    native,metadata=load_original_model(a.checkpoint,a.architecture,max_batch_size=1 if a.reference_eval else a.eval_batch_size)
    native.to(a.device).eval();view=NativeView(native).eval()
    enc=tiktoken.get_encoding('gpt2')
    class Adapter(bench.Adapter):
        supports_kv_cache=False
        def route_key(self,ids):return ()
        def activate(self,route):pass
        @torch.no_grad()
        def logits(self,ids):
            with inference_context(a.device):return super().logits(ids).float()
    adapter=Adapter(view,native.config,a.device,'deepseek_native',enc)
    datasets={};original_ds=bench.ds
    def ds(path,name=None,split='validation'):
        key=str((path,name,split))
        if key not in datasets:datasets[key]=original_ds(path,name,split)
        return datasets[key]
    bench.ds=ds
    report={'kind':'native_deepseek_benchmark','metadata':metadata,
        'checkpoint_sha256':metadata['checkpoint_sha256'],'evaluator_sha256':file_sha256(a.gpt2_evaluator),
        'parameters':sum(v.numel() for v in native.parameters()),'limit':a.limit or None,
        'precision':'FP32 weights; BF16 autocast on CUDA','routing':'native learned; all experts resident; no classifier or restrictions',
        'runtime':{'device':a.device,'torch':torch.__version__,'batch_size':1 if a.reference_eval else a.eval_batch_size,
                   'batch_tokens':a.batch_tokens,'incremental_kv':False,'reference_eval':a.reference_eval},
        'tasks':{},'completed':False}
    def save():
        report['datasets']={k:{'fingerprint':v._fingerprint,'rows':len(v)} for k,v in datasets.items()}
        durable_save(a.output,report)
    adapter.logits([1,2,3])
    if a.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        start=time.perf_counter()
        try:
            r=bench.TASKS[task](adapter,a.limit or None) if a.reference_eval else evaluate(bench,task,adapter,a.limit or None,a.eval_batch_size,a.batch_tokens)
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            r['seconds']=time.perf_counter()-start
        except (OSError,ConnectionError,RuntimeError) as error:
            if isinstance(error,RuntimeError) and 'Could not load dataset' not in str(error):raise
            r={'error':str(error),'seconds':time.perf_counter()-start}
        report['tasks'][task]=r;save();print(task,r,flush=True)
    report['memory']=module_memory(native)
    if a.device.startswith('cuda'):report['benchmark_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    if a.audit_routing:
        prompts=json.loads(a.audit_prompts.read_text(encoding='utf8')) if a.audit_prompts else [
            'Python is a programming language. To sort a list in Python,',
            'The derivative of x squared is','DNA carries genetic information. A gene is']
        if not isinstance(prompts,list) or not prompts or any(not isinstance(v,str) or not v.strip() for v in prompts):p.error('Audit prompts must be nonempty strings')
        report['routing_audit']=audit_routing(native,enc,prompts,a.device,a.audit_tokens)
    report['completed']=all('error' not in v for v in report['tasks'].values());save()


if __name__=='__main__':main()
