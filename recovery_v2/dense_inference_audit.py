"""Isolated full-prefix inference ablations; no weights or production paths change."""
import argparse
from contextlib import contextmanager
import gc
import json
import random
from pathlib import Path
import platform
import subprocess
import sys
import threading
import os
import types

import psutil
import tiktoken
import torch
import torch.nn.functional as F

from asi.experiments.local_generation import PROBES, load_variant, timed_generate
from asi.experiments.recovery_protocol import durable_save
from asi.models.original import DEFAULT_ARCHITECTURE, file_sha256
from asi.models.domain import apply_rotary_emb
from asi.runtime.generation import load_model, inference_context
from asi.runtime.cache import module_memory


def sdpa_forward(self, x, start_pos=0, freqs_cis=None, mask=None):
    """Direct MLA factorization, causal SDPA, preserving the inference scale."""
    if start_pos != 0 or freqs_cis is None:
        raise ValueError('This diagnostic supports full-prefix inference only')
    b,t,_=x.shape
    q=self.wq(x) if self.q_lora_rank==0 else self.wq_b(self.q_norm(self.wq_a(x)))
    q=q.view(b,t,self.n_local_heads,self.qk_head_dim)
    qn,qr=q.split([self.qk_nope_head_dim,self.qk_rope_head_dim],dim=-1)
    qr=apply_rotary_emb(qr,freqs_cis)
    kv,kr=self.wkv_a(x).split([self.kv_lora_rank,self.qk_rope_head_dim],dim=-1)
    kr=apply_rotary_emb(kr.unsqueeze(2),freqs_cis)
    kv=self.wkv_b(self.kv_norm(kv)).view(b,t,self.n_local_heads,self.qk_nope_head_dim+self.v_head_dim)
    kn,v=kv.split([self.qk_nope_head_dim,self.v_head_dim],dim=-1)
    q=torch.cat([qn,qr],dim=-1).transpose(1,2)
    k=torch.cat([kn,kr.expand(-1,-1,self.n_local_heads,-1)],dim=-1).transpose(1,2)
    v=v.transpose(1,2)
    # Padding V changes neither attention probabilities nor retained outputs;
    # matching head sizes lets CUDA select fused SDPA where supported.
    v=F.pad(v,(0,self.qk_head_dim-self.v_head_dim))
    y=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.softmax_scale)
    y=y[...,:self.v_head_dim].transpose(1,2).contiguous().view(b,t,-1)
    return self.wo(y)


@contextmanager
def mode(model, name):
    hook=None; originals=[]
    try:
        if name!='baseline':
            head=model.lm_head if hasattr(model,'lm_head') else model.head
            hook=head.register_forward_pre_hook(lambda m,args:(args[0][:,-1:,:],))
        if name=='sdpa_last':
            for layer in model.layers:
                attn=layer.attn; originals.append((attn,attn.forward))
                attn.forward=types.MethodType(sdpa_forward,attn)
        yield
    finally:
        if hook: hook.remove()
        for attn,forward in originals: attn.forward=forward


@torch.inference_mode()
def logits(model, ids, bf16):
    with inference_context('cuda') if bf16 else torch.autocast('cuda',enabled=False):
        out=model(torch.tensor([ids],device='cuda'))
    return (out[0] if isinstance(out,tuple) else out)[0,-1].float().cpu()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'report.json'
    if path.exists() and not a.resume: p.error('Use --resume')
    historical=json.loads(Path('results/deepseek_19072_reference_20261010/retry_cache/generation/report.json').read_text(encoding='utf8'))
    gpt_path=Path(next(m['checkpoint'] for m in historical['models'] if m['variant']=='gpt2'))
    audit=json.loads(Path('results/v2/F0/bank_active_matched_20261022/audit.json').read_text(encoding='utf8'))
    checkpoints={r['pool']:Path(r['checkpoint']) for r in audit['models']}
    settings=dict(script_sha256=file_sha256(Path(__file__)),seed=1337,tokens=64,repeats=3,
                  protocol='FP32 weights, BF16 autocast, batch1, full prefix, greedy; no compile; 4 CPU threads; randomized modes/repeats within each prompt',
                  evaluator_sha256=file_sha256(Path('../Creating-GPT-2-From-0/eval.py')))
    report=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(status='partial',settings=settings,results={},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version()),
        gpu=torch.cuda.get_device_name(),limitations=['No claim of universal equivalence from eight diagnostic prompts.',
        'SDPA changes MLA factorization and BF16 rounding; must pass both logit and output checks.',
        'Fixed-token timing ignores EOS; separate EOS output is checked.',
        'No incremental KV cache in either family. No transfer or classifier in timed region.',
        'Order randomized within prompts; timing uncertainty is descriptive, not independent model seeds.'])
    if report['settings']!=settings: p.error('Protocol changed: use a new directory')
    torch.set_num_threads(4); torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32=False
    enc=tiktoken.get_encoding('gpt2')
    for family in ['bank','gpt2']:
      for probe in PROBES:
        key=family+'/'+probe['pool']
        if key in report['results']: continue
        durable_save(a.output/'STATUS.json',dict(status='running',current=key,complete=len(report['results']),total=16,pid=os.getpid()))
        checkpoint=checkpoints[probe['pool']] if family=='bank' else gpt_path
        model,_=load_model(checkpoint) if family=='bank' else load_variant('gpt2',checkpoint,DEFAULT_ARCHITECTURE,Path('../Creating-GPT-2-From-0/eval.py'))
        model.to('cuda').eval()
        modes=['baseline','last']+(['sdpa_last'] if family=='bank' else [])
        reference=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
        eos=timed_generate(model,enc,probe['prompt'],'cuda',64)
        prefixes=[reference['input_ids']+reference['output_ids'][:n] for n in [0,16,48]]
        refs={bf:[logits(model,ids,bf) for ids in prefixes] for bf in [False,True]}
        record=dict(checkpoint_sha256=file_sha256(checkpoint),prompt=probe['prompt'],reference=reference,eos_reference=eos,modes={})
        for name in modes:
          with mode(model,name):
            checks={}
            for bf in [False,True]:
                diffs=[(logits(model,ids,bf)-target).abs() for ids,target in zip(prefixes,refs[bf])]
                checks['bf16' if bf else 'fp32']=dict(max_abs=max(float(d.max()) for d in diffs),
                    allclose=all(torch.allclose(logits(model,ids,bf),target,rtol=1e-4,atol=1e-4) for ids,target in zip(prefixes,refs[bf])))
            fixed=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
            readable=timed_generate(model,enc,probe['prompt'],'cuda',64)
            record['modes'][name]=dict(logit_checks=checks,
                fixed_tokens_identical=fixed['output_ids']==reference['output_ids'],
                eos_tokens_identical=readable['output_ids']==eos['output_ids'],
                fixed_output=fixed, eos_output=readable,runs=[])
        order=[name for name in modes for _ in range(3)]; random.Random(1337+PROBES.index(probe)).shuffle(order)
        for name in order:
          with mode(model,name):
            torch.cuda.reset_peak_memory_stats()
            run=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
            run.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),memory=module_memory(model),rss_bytes=psutil.Process().memory_info().rss,expert_h2d_bytes=0)
            record['modes'][name]['runs'].append(run)
        report['results'][key]=record; durable_save(path,report)
        print(key,{k:(v['fixed_tokens_identical'],v['eos_tokens_identical']) for k,v in record['modes'].items()},flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    report['status']='complete';durable_save(path,report)
    durable_save(a.output/'STATUS.json',dict(status='complete',complete=16,total=16))


if __name__=='__main__': main()
