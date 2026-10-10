"""Matched chat prompts and sampling for local GPT-2 and categorized checkpoints."""
import argparse
import gc
import json
from pathlib import Path
from collections import Counter
import tiktoken
import torch
from asi.experiments.checkpoint_eval import load_benchmarks
from asi.experiments.public_moe import sha256
from asi.experiments.recovery_protocol import durable_save
from asi.runtime.generation import load_model,validate_pool_identity,domain_chat_context,generate
from asi.runtime.routing import DomainSessionRouter
from asi.runtime.cache import ExpertCacheManager


PROMPTS=['What is math?','Show Python code that sorts the list [3, 1, 2].',
    'What is the derivative of x squared? Explain briefly.',
    'Explain what a gene is in two sentences.','Explain the role of carbon dioxide in photosynthesis.',
    'Explain inflation in two sentences.','Name two characteristics of the Roman Empire.',
    'Suggest two activities to help children learn to read.']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain-checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--gpt2-checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--gpt2-evaluator',type=Path,default=Path('../Creating-GPT-2-From-0/eval.py'))
    p.add_argument('--pool-manifest',type=Path,default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--prompts',type=Path,help='JSON array of prompt strings; default eight diagnostic prompts')
    p.add_argument('--seeds',type=int,nargs='+',default=[1337,2027])
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output directory')
    if a.max_new_tokens<1 or a.temperature<0 or a.top_k<0:p.error('Invalid generation settings')
    prompts=json.loads(a.prompts.read_text(encoding='utf8')) if a.prompts else PROMPTS
    if not isinstance(prompts,list) or not prompts or not all(isinstance(x,str) and x.strip() for x in prompts):p.error('Expected nonempty prompt strings')
    enc=tiktoken.get_encoding('gpt2');bench=load_benchmarks(a.gpt2_evaluator)
    manifest=json.loads(a.pool_manifest.read_text(encoding='utf8'));names=list(manifest['pools'])
    router=DomainSessionRouter(names,max_pools=1,session_inertia=0,pool_manifest=manifest)
    routes={prompt:router.route(prompt).ranked_pools[0] for prompt in prompts}
    report={'completed':False,'settings':json.loads(json.dumps(vars(a),default=str)),
            'prompts':prompts,'input_routes':routes,'evaluator_sha256':sha256(a.gpt2_evaluator),
            'protocol':'Same OASST format, no history, seed per prompt, FP32 weights/BF16 CUDA autocast, full-context forward for both. Quality comparison, not a latency benchmark.',
            'limitation':'Historical training exposures differ; GPT-2 checkpoint lacks full SFT settings. No equal-training or causal architecture claim.',
            'models':[]}
    a.output.mkdir(parents=True)
    for kind,paths in [('gpt2',a.gpt2_checkpoints),('domain',a.domain_checkpoints)]:
        for path in paths:
            cache=None
            if kind=='gpt2':
                state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                model=bench.GPT(bench.GPTConfig(**state['config']));model.load_state_dict(state['model'])
                metadata={k:state[k] for k in ['step','val_loss'] if k in state};del state
                model.to(a.device).eval()
            else:
                model,metadata=load_model(path);validate_pool_identity(model,metadata,names)
                for layer in model.layers:
                    if hasattr(layer.ffn,'moe_backend'):layer.ffn.moe_backend='bmm'
                cache=ExpertCacheManager(a.device,1);cache.initialize(model)
            if a.max_new_tokens>=model.config.block_size:p.error('Response exceeds context')
            record={'kind':kind,'checkpoint':str(path),'sha256':sha256(path),'metadata':metadata,'generations':[]}
            report['models'].append(record)
            for prompt in prompts:
                context,_=domain_chat_context(enc,[],prompt,'oasst',model.config.block_size-a.max_new_tokens)
                if cache:
                    pool=names.index(routes[prompt]);cache.prepare(model,[pool]);model.set_active_pools([pool])
                for seed in a.seeds:
                    torch.manual_seed(seed)
                    result=generate(model,enc,context,a.device,a.max_new_tokens,a.temperature,a.top_k,
                                    return_info=True,incremental=False)
                    result['text']=result['text'][len(context):].split('<|endoftext|>',1)[0]
                    record['generations'].append({'prompt':prompt,'seed':seed,**result})
                durable_save(a.output/'report.json',report)
            record['stop_counts']=dict(Counter(g['stop_reason'] for g in record['generations']))
            print(path,record['stop_counts'],flush=True)
            if cache:cache.offload_all(model)
            del cache,model;gc.collect()
            if a.device.startswith('cuda'):torch.cuda.empty_cache()
    report['completed']=True;durable_save(a.output/'report.json',report)
    lines=['# Conversación bajo el mismo protocolo','',report['protocol'],'',report['limitation']]
    for r in report['models']:
        lines+=['','## '+r['checkpoint'],'',str(r['stop_counts'])]
        for g in r['generations']:
            lines+=['',f"### {g['prompt']} (semilla {g['seed']})",'',g['text'],'',
                    f"Parada: {g['stop_reason']}; tokens: {g['generated_tokens']}. P(EOS) al último paso: {g['eos_probability_last_step']:.5f}."]
    (a.output/'review.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':main()
