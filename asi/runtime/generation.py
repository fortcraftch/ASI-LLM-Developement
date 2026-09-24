#!/usr/bin/env python3
"""Conversation entry points for domain-trained and existing V3 checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from asi import DATA_ROOT

import tiktoken
import torch
import torch.nn.functional as F

from asi.models.domain import GPT, GPTConfig
from asi.runtime.routing import DomainSessionRouter, ExpertUsagePredictor
from asi.runtime.cache import ExpertCacheManager, NativeExpertSessionCache, module_memory, move_backbone
from asi.analysis.experts import RoutingTrace
from asi.models.original import DEFAULT_ARCHITECTURE,load_original_model,generate_original,sync,native_moes,generate_incremental,load_streamed_model


def load_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    pools = payload["pools"]
    names = list(pools.keys())
    return names, {name: i for i, name in enumerate(names)}


def load_model(checkpoint: Path, device: str = "cpu"):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    # Release optimizer and duplicated checkpoint tensors before building RAM backing.
    metadata = {k: ckpt[k] for k in ("step", "val_loss", "pool_names", "experiment") if k in ckpt}
    return model, metadata


def validate_pool_identity(model, metadata, names):
    if len(names) != model.config.n_pools or len(set(names)) != len(names):
        raise ValueError("Pool manifest does not match model pool count")
    if metadata.get("pool_names") != names:
        raise ValueError("Checkpoint pool_names and manifest order differ or identity metadata is missing")


def inference_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()


@torch.no_grad()
def generate(model, enc, prompt: str, device: str, max_new_tokens: int, temperature: float, top_k: int):
    prompt_ids = enc.encode(prompt)
    if not prompt_ids:
        return ""
    if len(prompt_ids) >= model.config.block_size:
        prompt_ids = prompt_ids[-(model.config.block_size - 1):]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with inference_context(device):
        logits, _ = model(idx, start_pos=0)
    next_logits = logits[:, -1, :]

    generated = list(prompt_ids)
    for pos in range(len(prompt_ids), min(model.config.block_size, len(prompt_ids) + max_new_tokens)):
        logits = next_logits / max(temperature, 1e-5)
        logits[:, enc.n_vocab:] = float("-inf")
        if top_k:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == enc.eot_token:
            break
        if pos + 1 == min(model.config.block_size, len(prompt_ids) + max_new_tokens):
            break
        with inference_context(device):
            step_logits, _ = model(token, start_pos=pos)
        next_logits = step_logits[:, -1, :]

    return enc.decode(generated)


def domain_main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool-manifest", type=Path, default=(DATA_ROOT / "expert_pools.json"))
    p.add_argument("--classifier-model", default="mdonigian/fineweb-edu-topic-classifier")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-pools", type=int, default=3)
    p.add_argument("--max-hot-pools", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--classifier-device", default="cpu")
    p.add_argument("--no-classifier", action="store_true", help="Keyword-only diagnostic mode")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--max-pinned-mib", type=int, default=512)
    p.add_argument("--trace-dir", type=Path)
    p.add_argument("--session-context", action="store_true", help="Include prior turns and re-prefill after routing changes")
    args = p.parse_args()

    pool_names, pool_to_id = load_manifest(args.pool_manifest)
    model, ckpt = load_model(args.checkpoint)
    validate_pool_identity(model, ckpt, pool_names)
    router = DomainSessionRouter(
        pool_names=pool_names,
        classifier_model=args.classifier_model,
        device=args.classifier_device,
        max_pools=args.max_pools,
        load_classifier=not args.no_classifier,
    )
    cache = ExpertCacheManager(args.device, max_hot_pools=args.max_hot_pools,
                               pin_memory=args.pin_memory, max_pinned_bytes=args.max_pinned_mib * 1024**2)
    cache.initialize(model)
    enc = tiktoken.get_encoding("gpt2")

    print(f"Loaded step={ckpt.get('step')} | val_loss={ckpt.get('val_loss')}")
    print("Classifier memory:", module_memory(router.classifier))
    print("Pool order:")
    for i, name in enumerate(pool_names):
        print(f"  {i:02d} -> {name}")
    print("Type 'exit' to quit.")
    session_text = ""
    turn = 0

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        context = (session_text + "\n" + prompt).strip() if args.session_context else prompt
        context = enc.decode(enc.encode(context)[-(model.config.block_size - 1):])
        route = router.route(context)
        requested_ids = [pool_to_id[name] for name in route.ranked_pools]
        hot_ids = cache.prepare(model, requested_ids, requested_ids)
        model.set_active_pools(requested_ids)

        print("Pools:", ", ".join(route.ranked_pools))
        print("Scores:", ", ".join(f"{k}={v:.2f}" for k, v in route.scores.items() if v > 0.25))
        print("Cache:", cache.snapshot())

        trace = RoutingTrace(model, pool_names) if args.trace_dir else None
        with trace if trace else nullcontext():
            text = generate(model, enc, context, args.device, args.max_new_tokens,
                            args.temperature, args.top_k)
        if trace:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            trace.write(args.trace_dir / f"turn_{turn:04d}_routes.json")
            (args.trace_dir / f"turn_{turn:04d}_memory.json").write_text(
                json.dumps({"cache": cache.snapshot(), "memory": cache.inventory()}, indent=2), encoding="utf-8")
        print("Memory:", {k: v for k, v in cache.inventory().items() if k != "experts"})
        session_text = text
        turn += 1
        print("Model>", text)




def existing_main():
    p=argparse.ArgumentParser(description=__doc__)
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint',type=Path)
    weights.add_argument('--expert-store',type=Path, help='Stream exported expert shards instead of loading a full checkpoint')
    p.add_argument('--ram-expert-mib',type=float,default=32,help='Expert tensor RAM cache when using --expert-store')
    p.add_argument('--experts-per-layer',type=int,help='Hard per-layer routed expert limit; enables serial prefill and incremental decode')
    p.add_argument('--incremental',action='store_true')
    p.add_argument('--architecture',type=Path,default=DEFAULT_ARCHITECTURE)
    p.add_argument('--expert-labels',type=Path,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--policy',choices=['prefetch','restrict'],default='prefetch')
    p.add_argument('--cache-strategy', choices=['semantic', 'lru', 'popularity', 'learned'], default='semantic')
    p.add_argument('--usage-predictor', type=Path)
    p.add_argument('--prefetch-experts', type=int, help='Candidate budget for learned/popularity; default min(32, capacity)')
    p.add_argument('--learn-online', action='store_true', help='Update predictor only after each completed turn')
    p.add_argument('--predictor-output', type=Path, help='Separate output file for online updates')
    p.add_argument('--max-hot-experts',type=int,default=52)
    p.add_argument('--max-labels',type=int,default=3)
    p.add_argument('--pin-memory',action='store_true')
    p.add_argument('--classifier-device',default='cpu')
    p.add_argument('--no-classifier',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--session-context',action='store_true')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--prompt')
    source.add_argument('--prompts',type=Path,help='JSONL with prompt fields, optionally expected_pools')
    p.add_argument('--output',type=Path,default=Path('results/posthoc_chat.json'))
    a=p.parse_args()
    if a.max_new_tokens<1: p.error('max-new-tokens must be positive')
    if a.prefetch_experts is None:
        a.prefetch_experts = min(32, a.max_hot_experts)
    if not 0 <= a.prefetch_experts <= a.max_hot_experts:
        p.error('prefetch-experts must be between zero and max-hot-experts')
    if a.policy == 'restrict' and (a.cache_strategy != 'semantic' or a.learn_online):
        p.error('Learned caching requires unchanged native routing: --policy prefetch')
    if a.cache_strategy in ('learned', 'popularity') and not a.usage_predictor:
        p.error('This cache strategy requires --usage-predictor')
    if a.learn_online and (not a.usage_predictor or not a.predictor_output):
        p.error('--learn-online requires --usage-predictor and --predictor-output')
    if a.learn_online and a.predictor_output.resolve() == a.usage_predictor.resolve():
        p.error('Use a separate predictor-output to preserve the frozen training artifact')
    mapping=json.loads(a.expert_labels.read_text(encoding='utf-8-sig'))
    backing_store = None
    if a.expert_store:
        from asi.runtime.storage import ExpertDiskStore
        backing_store = ExpertDiskStore(a.expert_store, int(a.ram_expert_mib*1024**2))
        model,metadata = load_streamed_model(a.expert_store,a.architecture)
    else:
        model,metadata=load_original_model(a.checkpoint,a.architecture)
    if a.experts_per_layer is not None:
        a.incremental = True
        a.max_hot_experts = min(a.max_hot_experts, len(native_moes(model))*a.experts_per_layer)
    expected=mapping.get('provenance',{})
    for key in ('checkpoint_sha256','architecture_sha256'):
        if expected.get(key)!=metadata[key]:
            raise ValueError(f'Mapping provenance mismatch: {key}. Recalibrate this exact checkpoint and architecture.')
    predictor = None
    if a.usage_predictor:
        predictor = ExpertUsagePredictor.from_dict(json.loads(a.usage_predictor.read_text(encoding='utf-8')))
        for key in ('checkpoint_sha256', 'architecture_sha256'):
            if predictor.provenance.get(key) != metadata[key]:
                raise ValueError('Predictor provenance mismatch: ' + key)
        if predictor.provenance.get('classifier_enabled') != (not a.no_classifier) or predictor.provenance.get('max_labels') != a.max_labels:
            raise ValueError('Predictor classifier settings differ from this session')
        universe = {(layer, eid) for layer, moe in native_moes(model).items() for eid in range(len(moe.experts))}
        if set(predictor.experts) != universe:
            raise ValueError('Predictor expert universe differs from the loaded model')
    pool_names=sorted({label for layer in mapping['layers'].values() for label in layer})
    router=DomainSessionRouter(pool_names,device=a.classifier_device,max_pools=a.max_labels,
                               load_classifier=not a.no_classifier)
    move_backbone(model,a.device)
    cache=NativeExpertSessionCache(model,mapping,device=a.device,max_hot_experts=a.max_hot_experts,
                                   pin_memory=a.pin_memory,policy=a.policy,
                                   max_experts_per_layer=a.experts_per_layer,backing_store=backing_store)
    enc=tiktoken.get_encoding('gpt2'); history=''; turns=[]; previous_labels=[]
    rows=[{'prompt':a.prompt}] if a.prompt else (
        [json.loads(line) for line in a.prompts.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if a.prompts else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        cursor=0
        while True:
            if rows is not None:
                if cursor>=len(rows): break
                sample=rows[cursor]; prompt=sample['prompt']; cursor+=1
            else:
                try: prompt=input('\nYou> ').strip()
                except (EOFError,KeyboardInterrupt): break
                if prompt.lower() in ('exit','quit'): break
                if not prompt: continue
                sample={'prompt':prompt}
            context=(history+'\n'+prompt).strip() if a.session_context else prompt
            context=enc.decode(enc.encode(context)[-(model.config.block_size-1):])
            started=time.perf_counter(); route=router.route(context); classifier_seconds=time.perf_counter()-started
            before=cache.snapshot(); sync(a.device); started=time.perf_counter()
            if a.cache_strategy == 'semantic':
                cache.set_context_labels(route.ranked_pools)
            elif a.cache_strategy == 'lru':
                cache.prefetch_experts([])
            else:
                cache.prefetch_experts(predictor.rank(route.ranked_pools, previous_labels,
                                      mode=a.cache_strategy)[:a.prefetch_experts])
            if a.learn_online:
                cache.begin_turn()
            sync(a.device); prefetch_seconds=time.perf_counter()-started
            started=time.perf_counter()
            generate_fn = generate_incremental if a.incremental else generate_original
            output=generate_fn(model,enc,context,a.device,a.max_new_tokens,a.temperature)
            sync(a.device); generation_seconds=time.perf_counter()-started
            after=cache.snapshot()
            if a.learn_online:
                predictor.observe(previous_labels, route.ranked_pools, cache.turn_demands)
                predictor.provenance['online_adapted'] = True
                a.predictor_output.parent.mkdir(parents=True, exist_ok=True)
                a.predictor_output.write_text(json.dumps(predictor.to_dict(), indent=2), encoding='utf-8')
            previous_labels = route.ranked_pools
            record={'prompt':prompt,'expected_pools':sample.get('expected_pools'),
                    'classified_labels':route.ranked_pools,'scores':route.scores,'broad_scores':route.broad_scores,
                    'completion':output,'classifier_seconds':classifier_seconds,'prefetch_seconds':prefetch_seconds,
                    'generation_seconds':generation_seconds,
                    'cache_delta':{k:after.get(k,0)-before.get(k,0) for k in ('hits','misses','evictions','host_to_device_bytes','prefetch_loads')},
                    'cache':after,'memory':cache.inventory()}
            turns.append(record); history=context+output
            print('Labels:',route.ranked_pools,flush=True); print('Model>',output,flush=True)
            print('Cache delta:',record['cache_delta'],flush=True)
            a.output.write_text(json.dumps({'metadata':metadata,'policy':a.policy,'max_hot_experts':a.max_hot_experts,
                'cache_strategy':a.cache_strategy, 'learn_online':a.learn_online,
                'incremental':a.incremental, 'experts_per_layer':a.experts_per_layer,
                'usage_predictor':str(a.usage_predictor) if a.usage_predictor else None,
                'classifier_enabled':not a.no_classifier,'classifier_memory':module_memory(router.classifier),
                'note':'Serial prefill and incremental KV decode.' if a.incremental else 'Original full-context forward per generated token.',
                'turns':turns},indent=2),encoding='utf-8')
    finally: cache.close()

