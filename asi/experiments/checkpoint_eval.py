"""Evaluate a domain checkpoint with the benchmark protocol of the GPT-2 project.

The external evaluator is explicit and hashed; requests are batched across tasks.
Uniform-all is an inference ablation, not a trained unrestricted router.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from asi.models.domain import GPT, GPTConfig, Gate
from asi.runtime.cache import ExpertCacheManager, module_memory
from asi.runtime.generation import inference_context, generate, validate_pool_identity
from asi.runtime.routing import DomainSessionRouter
from asi.experiments.public_moe import sha256
from asi.experiments.benchmark_batch import evaluate as evaluate_batched


def load_benchmarks(path):
    spec = importlib.util.spec_from_file_location('asi_external_gpt2_eval', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_mode(model, mode, trained_mode):
    for gate in (m for m in model.modules() if isinstance(m, Gate)):
        gate.routing_mode = 'learned' if mode == 'all_router' else trained_mode
        gate.topk = model.config.n_activated_experts
        if mode == 'all_uniform':
            gate.routing_mode = 'uniform_pool'
            gate.topk = model.config.n_routed_experts


@torch.no_grad()
def choice_losses(adapter, context, endings, batch_size):
    """Same tokenization/truncation/loss as the sibling evaluator, batched options."""
    c = adapter.enc.encode(context)
    adapter.select(c)
    encoded=[]
    for ending in endings:
        ids=c+adapter.enc.encode(' '+ending)
        start=len(c)
        if len(ids)>adapter.block_size:
            start=max(1,start-(len(ids)-adapter.block_size));ids=ids[-adapter.block_size:]
        encoded.append((ids,max(0,start-1)))
    sums=[];means=[]
    for offset in range(0,len(encoded),batch_size):
        rows=encoded[offset:offset+batch_size]
        length=max(len(ids)-1 for ids,_ in rows)
        x=torch.full((len(rows),length),adapter.enc.eot_token,dtype=torch.long,device=adapter.device)
        y=x.clone();mask=torch.zeros_like(x,dtype=torch.bool)
        for i,(ids,start) in enumerate(rows):
            n=len(ids)-1
            x[i,:n]=torch.tensor(ids[:-1],device=adapter.device)
            y[i,:n]=torch.tensor(ids[1:],device=adapter.device)
            mask[i,start:n]=True
        logits=adapter.logits(x)
        losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),reduction='none').view_as(y)
        total=(losses*mask).sum(-1);count=mask.sum(-1)
        sums.extend(total.masked_fill(count==0,float('inf')).tolist())
        means.extend((total/count.clamp_min(1)).masked_fill(count==0,float('inf')).tolist())
    return sums,means


def hellaswag_batched(bench,adapter,limit,batch_size):
    data=bench.limited(bench.ds('allenai/hellaswag',split='validation'),limit)
    correct=normalized=0;predictions=[]
    for ex in bench.tqdm(data,desc='HellaSwag batched'):
        if batch_size:
            sums,means=choice_losses(adapter,ex['ctx'],ex['endings'],batch_size)
        else:
            c=adapter.enc.encode(ex['ctx'])
            pairs=[adapter.completion_losses(c+adapter.enc.encode(' '+ending),len(c)) for ending in ex['endings']]
            sums,means=map(list,zip(*pairs))
        p=min(range(4),key=sums.__getitem__);n=min(range(4),key=means.__getitem__)
        label=int(ex['label']);correct+=p==label;normalized+=n==label
        predictions.append({'sum':p,'normalized':n,'label':label,'losses_sum':sums,'losses_mean':means})
    return {'accuracy':correct/len(predictions),'accuracy_norm':normalized/len(predictions),
            'total':len(predictions),'predictions':predictions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpt2-evaluator', type=Path, required=True)
    p.add_argument('--pool-manifest', type=Path, default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', default='hellaswag,lambada,piqa,arc_easy,boolq,arithmetic', help='Comma-separated benchmark names, or all')
    p.add_argument('--limit', type=int, default=100, help='Examples per task; 0 uses the full dataset (500 synthetic arithmetic questions)')
    p.add_argument('--modes', nargs='+', choices=['classified','all_uniform','all_router'], default=['classified'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--probe-repeats', type=int, default=3)
    p.add_argument('--skip-probes',action='store_true',help='Benchmark tasks only, for runtime comparisons')
    p.add_argument('--oracle-probes', action='store_true', help='Use declared categories for the eight qualitative probes only')
    p.add_argument('--choice-batch-size',type=int,choices=[1,2,4],default=None,help='Legacy alias for --eval-batch-size')
    p.add_argument('--eval-batch-size',type=int,default=8,help='Maximum requests per batch, grouped by active pool')
    p.add_argument('--batch-tokens',type=int,default=2048,help='Maximum padded input tokens per batch (one long request may exceed this)')
    p.add_argument('--expert-residency',choices=['all','selected'],default='all',help='all keeps weights resident but does not activate extra pools; selected measures transfers')
    p.add_argument('--classifier-device',default=None,help='Defaults to the model device; use cpu for lower VRAM usage')
    p.add_argument('--classifier-batch-size',type=int,default=8)
    p.add_argument('--reference-eval',action='store_true',help='Disable fixed-pool fast path, choice batching and redundant-cache avoidance')
    p.add_argument('--chat-probes', action='store_true', help='Use OASST chat headers for qualitative probes only; benchmark scoring is unchanged')
    a = p.parse_args()
    if a.choice_batch_size is not None:
        a.eval_batch_size = a.choice_batch_size
    if a.limit < 0 or a.new_tokens < 0 or a.probe_repeats < 1 or min(a.eval_batch_size,a.batch_tokens,a.classifier_batch_size)<1:
        p.error('Nonnegative limit/new-tokens and positive probe-repeats required')
    if a.output.exists():
        p.error('Use a fresh output file')
    bench = load_benchmarks(a.gpt2_evaluator)
    tasks = list(bench.TASKS) if a.tasks=='all' else [t for t in a.tasks.split(',') if t]
    limit = a.limit or None
    if set(tasks) - set(bench.TASKS):
        p.error('Unknown benchmark task')
    torch.manual_seed(1337)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = GPTConfig(**{**ckpt['config'], 'max_batch_size': 1 if a.reference_eval else a.eval_batch_size})
    model = GPT(config).eval()
    model.load_state_dict(ckpt['model'])
    names = ckpt['pool_names']
    manifest = json.loads(a.pool_manifest.read_text(encoding='utf-8'))
    validate_pool_identity(model, ckpt, list(manifest['pools']))
    metadata = {k: ckpt[k] for k in ['step','val_loss','val_loss_timing','training_settings','experiment','sft_provenance'] if k in ckpt}
    del ckpt
    # Stable bmm backend avoids relying on the grouped kernel implicated in training timeouts.
    for layer in model.layers:
        if hasattr(layer.ffn, 'moe_backend'):
            layer.ffn.moe_backend = 'bmm'
            layer.ffn.uniform_fastpath = not a.reference_eval
    classifier_device = a.classifier_device or ('cpu' if a.reference_eval else a.device)
    router = DomainSessionRouter(names, max_pools=1, session_inertia=0, pool_manifest=manifest,device=classifier_device)
    enc = tiktoken.get_encoding('gpt2')
    routes = {}
    datasets = {}
    original_ds = bench.ds
    def recorded_ds(path, name=None, split='validation'):
        key = str((path,name,split))
        if key not in datasets:
            datasets[key] = original_ds(path,name,split)
        return datasets[key]
    bench.ds = recorded_ds
    report = {'metadata': metadata, 'checkpoint': str(a.checkpoint.resolve()),
              'checkpoint_sha256': sha256(a.checkpoint),
              'evaluator_sha256': hashlib.sha256(a.gpt2_evaluator.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'device': a.device,
              'gpu': torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
              'parameters': sum(p.numel() for p in model.parameters()),
              'precision': 'FP32 weights, BF16 autocast on CUDA', 'moe_backend': 'bmm',
              'evaluation_runtime': {'fixed_pool_fastpath':not a.reference_eval,
                   'choice_batch_size':1 if a.reference_eval else a.eval_batch_size,
                   'engine':'reference' if a.reference_eval else 'grouped_requests_v1',
                   'eval_batch_size':1 if a.reference_eval else a.eval_batch_size,
                   'batch_tokens':a.batch_tokens,'expert_residency':a.expert_residency,
                   'generation_kv_cache':not a.reference_eval,
                   'classifier_device':classifier_device,'classifier_batch_size':a.classifier_batch_size,
                   'skip_unchanged_pool_prepare':not a.reference_eval},
              'classifier_memory': module_memory(router.classifier),
              'limit': limit, 'new_tokens': a.new_tokens, 'generation_repeats': a.probe_repeats,
              'probe_categories': 'declared' if a.oracle_probes else 'predicted',
              'probe_format': 'oasst' if a.chat_probes else 'completion',
              'trained_routing': config.routing_mode, 'results': {},
              'notes': ['Full selected datasets, matching sibling evaluator; arithmetic uses its default 500 cases.' if limit is None else 'First N examples, matching sibling evaluator; exploratory subset, not full benchmark.',
                        'all_router uses UNTRAINED gates when trained_routing=uniform_pool.',
                        'all_uniform activates all experts with total mixture weight unchanged; out-of-training distribution.',
                        'classified chooses from input only, fixed across answer choices; classifier cost is reported separately.',
                        'Benchmark seconds include scoring, classification, transfers and Python overhead; generation timings separate.',
                        'Optimized requests are grouped by pool; generation uses KV cache and rolling-window fallback.',
                        'All-resident evaluation measures quality/throughput, not limited-residency swapping performance.',
                        'No semantic coherence score is inferred from repetition or NLL.']}
    def save():
        report['datasets'] = {k: {'fingerprint': v._fingerprint, 'rows': len(v)} for k,v in datasets.items()}
        report['input_routes'] = routes
        a.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(a.output)
    def sync():
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
    for mode in a.modes:
        configure_mode(model, mode, config.routing_mode)
        all_resident = a.expert_residency == 'all' or mode != 'classified'
        cache = ExpertCacheManager(a.device, len(names) if all_resident else 1)
        cache.initialize(model)
        if all_resident:
            cache.prepare(model,list(range(len(names))))
        class Adapter(bench.Adapter):
            selected_active = None
            classifier_seconds = 0.
            def prepare_routes(self, contexts):
                if mode != 'classified': return
                pending = list(dict.fromkeys(enc.decode(list(ids)) for ids in contexts))
                pending = [text for text in pending if text not in routes]
                for offset in range(0,len(pending),a.classifier_batch_size):
                    texts = pending[offset:offset+a.classifier_batch_size]
                    started = time.perf_counter()
                    classified = router.route_many(texts,a.classifier_batch_size)
                    self.classifier_seconds += time.perf_counter()-started
                    for text,result in zip(texts,classified):
                        routes[text] = result.ranked_pools[0]

            def route_key(self, ids):
                text = enc.decode(list(ids))
                if mode == 'classified':
                    if text not in routes:
                        started = time.perf_counter()
                        routes[text] = router.route(text).ranked_pools[0]
                        self.classifier_seconds += time.perf_counter()-started
                    active = [names.index(routes[text])]
                else:
                    active = list(range(len(names)))
                return tuple(active)

            def activate(self, active):
                if a.reference_eval or self.selected_active != tuple(active):
                    if not all_resident:
                        cache.prepare(model, active)
                    model.set_active_pools(active)
                    self.selected_active=tuple(active)

            def select(self, ids):
                self.activate(self.route_key(ids))

            def completion_losses(self, ids, completion_start):
                self.select(ids[:completion_start])
                return super().completion_losses(ids, completion_start)

            def greedy_generate(self, prompt, max_new_tokens=20):
                self.select(enc.encode(prompt))
                return super().greedy_generate(prompt, max_new_tokens)

            @torch.no_grad()
            def logits(self, ids):
                with inference_context(a.device):
                    return super().logits(ids).float()
        adapter = Adapter(model, config, a.device, mode, enc)
        adapter.select(enc.encode('This is a warmup.'))
        adapter.logits([1,2,3])
        sync()
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        results = report['results'][mode] = {'tasks': {}, 'generations': [],
            'startup_cache':cache.snapshot(),'active_pools_per_question':1 if mode=='classified' else len(names)}
        for task in tasks:
            before = cache.snapshot()
            classifier_before = adapter.classifier_seconds
            sync(); start = time.perf_counter()
            try:
                if a.reference_eval:
                    result = (hellaswag_batched(bench,adapter,limit,0) if task=='hellaswag'
                              else bench.TASKS[task](adapter, limit))
                else:
                    result = evaluate_batched(bench,task,adapter,limit,a.eval_batch_size,a.batch_tokens)
                sync()
                result['seconds'] = time.perf_counter()-start
                result['classifier_seconds'] = adapter.classifier_seconds-classifier_before
                after = cache.snapshot()
                result['cache_delta'] = {k:after.get(k,0)-before.get(k,0) for k in ['hits','misses','evictions','host_to_device_bytes']}
            except (ConnectionError, OSError, RuntimeError) as error:
                if isinstance(error, RuntimeError) and 'Could not load dataset' not in str(error):
                    raise
                result = {'error': str(error)}
            results['tasks'][task] = result
            save()
            print(mode, task, {k:v for k,v in result.items() if k!='predictions'}, flush=True)
        probes = ['Python is a programming language. To sort a list in Python,',
                  'The derivative of x squared is', 'DNA carries genetic information. A gene is',
                  'Vaccines help the immune system by', 'Plants absorb carbon dioxide and',
                  'Inflation is an increase in', 'The Roman Empire was',
                  'Children learn to read by']
        if a.chat_probes:
            probes=['Show Python code that sorts the list [3, 1, 2].',
                    'What is the derivative of x squared? Explain briefly.',
                    'Explain what a gene is in two sentences.',
                    'Explain how vaccines help the immune system in two sentences.',
                    'Explain the role of carbon dioxide in photosynthesis.',
                    'Explain inflation in two sentences.',
                    'Name two characteristics of the Roman Empire.',
                    'Suggest two activities to help children learn to read.']
        for pool_id,prompt in enumerate(probes):
            if a.skip_probes: break
            if a.oracle_probes and mode=='classified':
                expected=['technology','math_physics','biology','health','environment','society_economy','history_arts','education_language'][pool_id]
                selected=names.index(expected)
                adapter.activate((selected,))
            else:
                adapter.select(enc.encode(prompt))
            model_prompt='<|user|>\n'+prompt+'\n<|assistant|>\n' if a.chat_probes else prompt
            for repeat in range(a.probe_repeats):
                before = cache.snapshot()
                sync(); start=time.perf_counter()
                with torch.no_grad():
                    output=generate(model,enc,model_prompt,a.device,a.new_tokens,1.,1)
                sync(); elapsed=time.perf_counter()-start
                text=output[len(model_prompt):]
                count=len(enc.encode(text,allowed_special={'<|endoftext|>'}))
                results['generations'].append({'prompt':prompt,'text':text,'repeat':repeat,
                    'seconds':elapsed,'tokens':count,'tokens_per_second':count/elapsed,
                    'loads_during_generation':cache.snapshot().get('misses',0)-before.get('misses',0)})
            save()
        results['inventory'] = cache.inventory()
        save()
        cache.offload_all(model)
    report['completed'] = all('error' not in r for mode in report['results'].values() for r in mode['tasks'].values())
    save()


if __name__ == '__main__':
    main()
