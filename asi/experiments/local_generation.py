"""Native chat and matched full-context generation for the small local checkpoints.

This deliberately measures the existing implementations, not architectural FLOPs.
Fixed-length timing runs ignore EOS; separate readable runs stop at EOS.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time

import tiktoken
import torch

from asi.models.original import DEFAULT_ARCHITECTURE, load_original_model, file_sha256, sync
from asi.models.domain import GPT, GPTConfig
from asi.runtime.generation import inference_context, domain_chat_context, validate_pool_identity
from asi.runtime.cache import ExpertCacheManager, module_memory
from asi.experiments.checkpoint_eval import load_benchmarks, configure_mode
from asi.experiments.recovery_protocol import durable_save


PROBES = [
    {'prompt': 'Python is a programming language. To sort a list in Python,', 'pool': 'technology'},
    {'prompt': 'The derivative of x squared is', 'pool': 'math_physics'},
    {'prompt': 'DNA carries genetic information. A gene is', 'pool': 'biology'},
    {'prompt': 'Vaccines help the immune system by', 'pool': 'health'},
    {'prompt': 'Plants absorb carbon dioxide and', 'pool': 'environment'},
    {'prompt': 'Inflation is an increase in', 'pool': 'society_economy'},
    {'prompt': 'The Roman Empire was', 'pool': 'history_arts'},
    {'prompt': 'Children learn to read by', 'pool': 'education_language'},
]


@torch.inference_mode()
def timed_generate(model, enc, prompt, device, new_tokens, *, stop_at_eos=True,
                   temperature=0., top_k=0):
    """Same full-prefix forward and sampling path for dense, native and domain models."""
    ids = enc.encode(prompt, allowed_special={'<|endoftext|>'})
    if not ids or new_tokens < 1 or len(ids) + new_tokens > model.config.block_size:
        raise ValueError('Nonempty prompt and response must fit the context without truncation')
    if temperature < 0 or top_k < 0:
        raise ValueError('Invalid sampling settings')
    input_ids = list(ids)
    output_ids = []
    first_seconds = None
    sync(device)
    started = time.perf_counter()
    for _ in range(new_tokens):
        x = torch.tensor([ids], device=device)
        with inference_context(device):
            output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        scores = logits[0, -1, :enc.n_vocab].float()
        if temperature == 0:
            token = int(scores.argmax())
        else:
            scores = scores / temperature
            if top_k:
                scores = scores.masked_fill(scores < scores.topk(min(top_k, len(scores))).values[-1], -float('inf'))
            token = int(torch.multinomial(scores.softmax(-1), 1))
        ids.append(token)
        output_ids.append(token)
        if first_seconds is None:
            sync(device)
            first_seconds = time.perf_counter() - started
        if stop_at_eos and token == enc.eot_token:
            break
    sync(device)
    seconds = time.perf_counter() - started
    return {'text': enc.decode(output_ids), 'input_ids': input_ids, 'output_ids': output_ids,
            'generated_tokens': len(output_ids), 'seconds': seconds,
            'tokens_per_second': len(output_ids) / seconds, 'first_token_seconds': first_seconds,
            'decode_tokens_per_second': ((len(output_ids)-1)/(seconds-first_seconds)
                                         if len(output_ids)>1 else None),
            'stop_reason': 'eos' if stop_at_eos and output_ids[-1] == enc.eot_token else 'token_limit',
            'stop_at_eos': stop_at_eos}


def chat_main():
    p = argparse.ArgumentParser(description='Generate with the original free-routing DeepSeek checkpoint')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--architecture', type=Path, default=DEFAULT_ARCHITECTURE)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--prompt')
    p.add_argument('--chat-format', choices=['completion', 'oasst'], default='completion')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.output and a.output.exists(): p.error('Use a fresh output file')
    if a.max_new_tokens < 1 or a.temperature < 0 or a.top_k < 0: p.error('Invalid generation settings')
    model, metadata = load_original_model(a.checkpoint, a.architecture, max_batch_size=1)
    model.to(a.device).eval()
    if a.max_new_tokens >= model.config.block_size: p.error('Response must fit context')
    enc = tiktoken.get_encoding('gpt2')
    torch.manual_seed(a.seed)
    report = {'metadata': metadata, 'settings': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'note': 'Independent turns; original learned router, all experts resident; full-context forwards.', 'turns': []}
    while True:
        try: prompt = a.prompt if a.prompt is not None else input('\nYou> ').strip()
        except (EOFError, KeyboardInterrupt): break
        if prompt.lower() in ('quit', 'exit') and a.prompt is None: break
        if not prompt.strip():
            if a.prompt is not None: p.error('Empty prompt')
            continue
        context, _ = domain_chat_context(enc, [], prompt, a.chat_format, model.config.block_size-a.max_new_tokens)
        result = timed_generate(model, enc, context, a.device, a.max_new_tokens,
                                temperature=a.temperature, top_k=a.top_k)
        print('\nModel>', result['text'], flush=True)
        print(f"[{result['stop_reason']}; {result['tokens_per_second']:.2f} tokens/s; {result['seconds']:.2f} s]")
        report['turns'].append({'prompt': prompt, **result})
        if a.output: durable_save(a.output, report)
        if a.prompt is not None: break


def load_variant(kind, path, architecture, evaluator):
    if kind == 'deepseek':
        return load_original_model(path, architecture, max_batch_size=1)
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if kind == 'gpt2':
        bench = load_benchmarks(evaluator)
        model = bench.GPT(bench.GPTConfig(**state['config']))
    else:
        model = GPT(GPTConfig(**{**state['config'], 'max_batch_size': 1}))
    model.load_state_dict(state['model'], strict=True)
    metadata = {k:state[k] for k in ('step','val_loss','pool_names','config') if k in state}
    metadata['checkpoint_sha256'] = file_sha256(path)
    if kind == 'gpt2': metadata['architecture_sha256'] = file_sha256(evaluator)
    return model.eval(), metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deepseek-checkpoint', type=Path, required=True)
    p.add_argument('--domain-checkpoint', type=Path, required=True)
    p.add_argument('--gpt2-checkpoint', type=Path)
    p.add_argument('--architecture', type=Path, default=DEFAULT_ARCHITECTURE)
    p.add_argument('--gpt2-evaluator', type=Path, default=Path('../Creating-GPT-2-From-0/eval.py'))
    p.add_argument('--pool-manifest', type=Path, default=Path('configs/experiment_124m_v1/expert_pools.json'))
    p.add_argument('--prompts', type=Path, help='JSON array of {prompt, pool}; declared pools isolate LLM cost')
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Use a fresh output directory')
    if min(a.new_tokens, a.repeats) < 1: p.error('Positive token/repeat counts required')
    probes = json.loads(a.prompts.read_text(encoding='utf8')) if a.prompts else PROBES
    manifest = json.loads(a.pool_manifest.read_text(encoding='utf8'))
    names = list(manifest['pools'])
    if not isinstance(probes,list) or not probes or any(not isinstance(v,dict) or not isinstance(v.get('prompt'),str)
            or not v['prompt'].strip() or v.get('pool') not in names for v in probes):
        p.error('Each probe needs a nonempty prompt and a pool from the manifest')
    enc = tiktoken.get_encoding('gpt2')
    report = {'completed':False, 'kind':'local_124m_generation',
              'protocol': 'FP32 weights, CUDA BF16 autocast, batch 1, greedy, full-prefix forwards for every variant; one warmup per prompt; equal fixed-token timing ignores EOS; separate EOS-stopping outputs. Declared pools exclude classifier cost.',
              'limitations': ['Unequal training exposures and expert widths; implementation-specific speed, not causal architecture superiority.',
                             'No KV incremental fast path in this comparison. Timings exclude model loading and pool preparation.',
                             'Three repeats describe local variability, not independent model seeds. Desktop GPU may have other workloads.',
                             'Checkpoint files can contain optimizer state; disk size is not model weight size.'],
              'settings': {'new_tokens':a.new_tokens,'repeats':a.repeats,'device':a.device,'torch':torch.__version__,
                           'platform':platform.platform(),'gpu':torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None},
              'source_sha256': {'asi/experiments/local_generation.py':file_sha256(__file__),
                                'asi/models/domain.py':file_sha256(Path(__file__).parents[1]/'models/domain.py')},
              'pool_manifest_sha256':file_sha256(a.pool_manifest), 'probes':probes,'models':[]}
    a.output.mkdir(parents=True)
    variants = [('deepseek', a.deepseek_checkpoint), ('domain_selected', a.domain_checkpoint),
                ('domain_all_resident', a.domain_checkpoint), ('domain_all_uniform', a.domain_checkpoint)]
    if a.gpt2_checkpoint: variants.append(('gpt2', a.gpt2_checkpoint))
    for kind,path in variants:
        torch.manual_seed(1337)
        model,metadata = load_variant(kind, path, a.architecture, a.gpt2_evaluator)
        cache = None
        if kind.startswith('domain'):
            validate_pool_identity(model, metadata, names)
            for layer in model.layers:
                if hasattr(layer.ffn,'moe_backend'):
                    layer.ffn.moe_backend = 'bmm'
                    layer.ffn.uniform_fastpath = True
            configure_mode(model, 'all_uniform' if kind=='domain_all_uniform' else 'classified', model.config.routing_mode)
            cache = ExpertCacheManager(a.device, 1 if kind=='domain_selected' else len(names))
            cache.initialize(model)
            if kind!='domain_selected': cache.prepare(model, list(range(len(names))))
        else: model.to(a.device)
        config = vars(model.config)
        record = {'variant':kind,'checkpoint':str(path.resolve()),'checkpoint_file_bytes':path.stat().st_size,
                  'metadata':metadata,'config':config,'parameters':sum(p.numel() for p in model.parameters()),
                  'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'runs':[], 'outputs':[]}
        report['models'].append(record)
        for probe in probes:
            sync(a.device); started=time.perf_counter()
            before_prepare=cache.snapshot() if cache else {}
            if cache:
                active=list(range(len(names))) if kind=='domain_all_uniform' else [names.index(probe['pool'])]
                if kind=='domain_selected': cache.prepare(model,active)
                model.set_active_pools(active)
            sync(a.device); prepare=time.perf_counter()-started
            after_prepare=cache.snapshot() if cache else {}
            timed_generate(model,enc,probe['prompt'],a.device,a.new_tokens,stop_at_eos=False)
            if a.device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(a.device)
            for repeat in range(a.repeats):
                before=cache.snapshot() if cache else {}
                result=timed_generate(model,enc,probe['prompt'],a.device,a.new_tokens,stop_at_eos=False)
                after=cache.snapshot() if cache else {}
                result.update({'prompt':probe['prompt'],'pool':probe['pool'],'repeat':repeat,
                               'prepare_seconds':prepare if repeat==0 else 0.,
                               'prepare_h2d_bytes':after_prepare.get('host_to_device_bytes',0)-before_prepare.get('host_to_device_bytes',0) if repeat==0 else 0,
                               'loads_during_generation':after.get('loads',0)-before.get('loads',0),
                               'h2d_bytes_during_generation':after.get('host_to_device_bytes',0)-before.get('host_to_device_bytes',0),
                               'memory':module_memory(model),
                               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None})
                record['runs'].append(result)
                durable_save(a.output/'report.json',report)
            record['outputs'].append({'prompt':probe['prompt'], 'pool':probe['pool'],
                                     **timed_generate(model,enc,probe['prompt'],a.device,a.new_tokens)})
            print(kind,probe['pool'],f"{result['tokens_per_second']:.2f} tokens/s",flush=True)
        record['median_tokens_per_second']=statistics.median(r['tokens_per_second'] for r in record['runs'])
        if cache: cache.offload_all(model)
        del model, cache
        gc.collect()
        if a.device.startswith('cuda'): torch.cuda.empty_cache()
    report['completed']=True
    durable_save(a.output/'report.json',report)
    from asi.analysis.local_generation_report import build
    build(a.output/'report.json',a.output)


if __name__ == '__main__': main()
