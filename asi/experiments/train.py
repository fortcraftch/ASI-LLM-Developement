#!/usr/bin/env python3
"""Train the pool-restricted DeepSeek-V3 prototype.

Each optimizer step samples exactly one semantic expert pool. Routed experts are
hard-masked to that pool, while shared experts remain active on every batch.
This is the first controlled experiment for the specialization hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import sys
import bisect
from dataclasses import asdict
from pathlib import Path
from asi import DATA_ROOT
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from asi.models.domain import GPT, GPTConfig
from asi.data.identity import manifest_matches


class PoolManifest:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.pools = payload.get("pools", {})
        if not self.pools:
            raise ValueError(f"No pools found in {path}")
        self.names = list(self.pools.keys())
        self.name_to_id = {name: i for i, name in enumerate(self.names)}

    def pool_id(self, name: str) -> int:
        return self.name_to_id[name]

    def categories(self, name: str) -> List[str]:
        return list(self.pools[name].get("categories", []))

    def token_weights(self) -> Dict[str, int]:
        return {name: max(1, int(info.get("tokens", 1))) for name, info in self.pools.items()}


class TokenShardStream:
    """Circular stream across nonempty shards of one pool, with one-token overlap."""
    def __init__(self, shard_paths: List[Path], B: int, T: int):
        if B < 1 or T < 1:
            raise ValueError("Batch size and sequence length must be positive")
        self.shards = []
        for path in sorted(shard_paths):
            tokens = np.load(path, mmap_mode="r")
            valid = tokens.ndim == 1 and tokens.dtype == np.uint16
            size = tokens.size
            tokens._mmap.close()
            if not valid:
                raise ValueError(f"Expected 1-D uint16 tokens in {path}")
            if size:
                self.shards.append(path)
        if not self.shards:
            raise ValueError("Pool has no nonempty token shards")
        self.B, self.T = B, T
        self.current_shard = 0
        self.position = 0
        self.tokens = None
        self._open_current()

    def _open_current(self):
        if self.tokens is not None:
            self.tokens._mmap.close()
        self.tokens = np.load(self.shards[self.current_shard], mmap_mode="r")

    def _advance_if_end(self):
        if self.position == len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.position = 0
            self._open_current()

    def state(self):
        return {"shard": self.current_shard, "position": int(self.position)}

    def load_state(self, state):
        shard = int(state.get("shard", 0))
        position = int(state.get("position", 0))
        if not 0 <= shard < len(self.shards):
            raise ValueError("Saved shard index is out of range")
        self.current_shard = shard
        self._open_current()
        if not 0 <= position <= len(self.tokens):
            raise ValueError("Saved token position is out of range")
        self.position = position

    def next_batch(self):
        count = self.B * self.T
        buf = np.empty(count + 1, dtype=np.int64)
        written = 0
        while written < count:
            self._advance_if_end()
            take = min(count - written, len(self.tokens) - self.position)
            buf[written:written + take] = self.tokens[self.position:self.position + take]
            written += take
            self.position += take
        self._advance_if_end()
        # Peek, do not consume: this target starts the next batch as an input.
        buf[-1] = self.tokens[self.position]
        x = torch.from_numpy(buf[:-1].copy()).view(self.B, self.T)
        y = torch.from_numpy(buf[1:].copy()).view(self.B, self.T)
        return x, y


class RandomWindowStream:
    """Sample windows across a pool by available token positions, with replacement."""
    def __init__(self, paths, B, T, seed):
        if min(B,T)<1: raise ValueError('Positive batch and context required')
        self.B,self.T=B,T
        self.rng=random.Random(seed)
        self.paths=[];self.ends=[];self.identity=[]
        total=0
        for path in sorted(paths):
            arr=np.load(path,mmap_mode='r')
            valid=arr.ndim==1 and arr.dtype==np.uint16;size=arr.size;arr._mmap.close()
            if not valid: raise ValueError('Expected one-dimensional uint16 shards')
            if size>T:
                total+=size-T;self.paths.append(path);self.ends.append(total)
                self.identity.append((path.parent.name,path.name,int(size)))
        if not total: raise ValueError('No shard can supply a full random window')

    def next_batch(self):
        rows=[]
        for _ in range(self.B):
            offset=self.rng.randrange(self.ends[-1]);idx=bisect.bisect_right(self.ends,offset)
            start=offset-(self.ends[idx-1] if idx else 0)
            arr=np.load(self.paths[idx],mmap_mode='r')
            rows.append(np.array(arr[start:start+self.T+1],dtype=np.int64,copy=True));arr._mmap.close()
        tokens=torch.from_numpy(np.stack(rows))
        return tokens[:,:-1].contiguous(),tokens[:,1:].contiguous()

    def state(self):
        return {'kind':'random_windows','identity':self.identity,'B':self.B,'T':self.T,'rng':self.rng.getstate()}

    def load_state(self,state):
        if any(state.get(k)!=v for k,v in (('identity',self.identity),('B',self.B),('T',self.T))):
            raise ValueError('Random stream dimensions or shard inventory changed on resume')
        self.rng.setstate(state['rng'])


def activate_training_pool(model, pool_id):
    recipe=getattr(model,'training_recipe',None)
    family=recipe['family'] if recipe else 'domain'
    active=list(range(model.config.n_pools)) if family in ('base','router') else [pool_id]
    model.set_active_pools(active)


class PoolData:
    def __init__(self, data_root: Path, manifest: PoolManifest, split: str, B: int, T: int, order='sequential', seed=1337):
        self.streams = {}
        for pool_name in manifest.names:
            shards = []
            for category in manifest.categories(pool_name):
                category_dir = data_root / category
                shards.extend(p for p in sorted(category_dir.glob(f"{split}_*.npy")) if not p.name.endswith(".part.npy"))
            if shards:
                self.streams[pool_name] = (RandomWindowStream(shards,B,T,seed+manifest.pool_id(pool_name))
                                           if order=='random' else TokenShardStream(shards,B,T))
        if not self.streams:
            raise RuntimeError(f"No '{split}' shards were found under {data_root}")

    def names(self):
        return list(self.streams.keys())

    def next_batch(self, pool_name):
        return self.streams[pool_name].next_batch()

    def state(self):
        return {name: stream.state() for name, stream in self.streams.items()}

    def load_state(self, state):
        for name, stream_state in state.items():
            if name in self.streams:
                self.streams[name].load_state(stream_state)


class PoolScheduler:
    def __init__(self, manifest: PoolManifest, mode: str, seed: int):
        self.rng = random.Random(seed)
        self.mode = mode
        self.names = manifest.names
        weights = manifest.token_weights()
        total = float(sum(weights.values()))
        self.probs = [math.sqrt(weights[n]) if mode=='sqrt' else weights[n] / total for n in self.names]

    def sample(self, available):
        if not available:
            raise RuntimeError("No trainable expert pools available")
        if self.mode == "uniform":
            return self.rng.choice(available)
        # token-weighted, restricted to pools that actually have shards
        weights = [self.probs[self.names.index(n)] for n in available]
        total = sum(weights)
        weights = [w / total for w in weights]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def state(self):
        return self.rng.getstate()

    def load_state(self, state):
        self.rng.setstate(state)


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it >= max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def autocast_context(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def validate_pools(model: GPT, manifest: PoolManifest):
    if len(manifest.names) != model.config.n_pools:
        raise ValueError(
            f"Pool manifest has {len(manifest.names)} pools, model expects {model.config.n_pools}."
        )
    if model.config.n_routed_experts != len(manifest.names) * model.config.experts_per_pool:
        raise ValueError("Model expert count and pool manifest are inconsistent.")


def validate_loader_against_model(loader_names: List[str], manifest: PoolManifest):
    missing = [n for n in manifest.names if n not in loader_names]
    if missing:
        print(f"Warning: pools without shards in this split: {missing}")


def evaluate_by_pool(model, val_data: PoolData, manifest: PoolManifest, device: str, steps_per_pool: int):
    model.eval()
    results = {}
    total = 0.0
    count = 0
    original_state = val_data.state()
    with torch.no_grad():
        for pool_name in manifest.names:
            if pool_name not in val_data.streams:
                continue
            pool_id = manifest.pool_id(pool_name)
            activate_training_pool(model, pool_id)
            acc = 0.0
            for _ in range(steps_per_pool):
                x, y = val_data.next_batch(pool_name)
                x, y = x.to(device), y.to(device)
                with autocast_context(device):
                    _, loss = model(x, y)
                acc += float(loss.detach().item())
            acc /= steps_per_pool
            results[pool_name] = acc
            total += acc
            count += 1
    overall = total / count if count else float("nan")
    val_data.load_state(original_state)
    model.train()
    return overall, results


def save_checkpoint(path, model, optimizer, step, train_data, scheduler, val_loss, pool_names, val_data=None):
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "train_state": train_data.state(),
        "val_state": val_data.state() if val_data is not None else None,
        "pool_rng_state": scheduler.state(),
        "pool_names": pool_names,
        "experiment": ((getattr(model, 'training_recipe', None) or {}).get('family', 'domain') + "_training_v1"),
        "checkpoint_timing": "after_optimizer_step",
        "val_loss_timing": "before_this_step_update" if val_loss is not None else None,
        "training_recipe": getattr(model,'training_recipe',None),
        "training_settings": getattr(model,'training_settings',None),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    early=argparse.ArgumentParser(add_help=False)
    early.add_argument('--recipe',type=Path)
    selected,_=early.parse_known_args()
    recipe=json.loads(selected.recipe.read_text(encoding='utf-8')) if selected.recipe else None
    p = argparse.ArgumentParser()
    p.add_argument('--recipe',type=Path,help='Portable model/training configuration produced by design')
    p.add_argument('--dry-run',action='store_true',help='Validate and count on meta; no weights allocated and no training')
    p.add_argument('--data-order',choices=['sequential','random'],default='sequential')
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--pool-manifest", type=Path, default=(DATA_ROOT / "expert_pools.json"))
    p.add_argument("--log-dir", type=Path, default=Path("log_domain"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument('--auto-restart', action='store_true', help='Restart a fresh process after CUDA errors')
    p.add_argument('--max-restarts', type=int, default=5)
    p.add_argument('--restart-delay', type=float, default=15)
    p.add_argument('--keep-checkpoints', type=int, default=0, help='Keep newest N model_*.pt files in log-dir; 0 keeps all')
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--total-batch-size", type=int, default=524288)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--val-steps-per-pool", type=int, default=1)
    p.add_argument("--pool-sampling", choices=["uniform", "token", "sqrt"], default="uniform")

    p.add_argument("--n-routed-experts", type=int, default=16)
    p.add_argument("--experts-per-pool", type=int, default=2)
    p.add_argument("--n-activated-experts", type=int, default=2)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)

    p.add_argument("--max-lr", type=float, default=6e-4)
    p.add_argument("--min-lr-factor", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=715)
    p.add_argument("--weight-decay", type=float, default=0.1)
    if recipe:
        p.set_defaults(**recipe['training'])
    args = p.parse_args()
    explicit = {arg.split('=')[0] for arg in sys.argv[1:]}
    if args.auto_restart:
        if '--save-interval' not in explicit:
            args.save_interval = 10
        if '--keep-checkpoints' not in explicit:
            args.keep_checkpoints = 3
    if args.max_restarts < 0 or args.restart_delay < 0 or not math.isfinite(args.restart_delay) or args.keep_checkpoints < 0:
        p.error('Restart limits, delay and checkpoint retention must be nonnegative')
    if recipe and any(arg.split('=')[0] in ('--n-layer','--n-head','--n-embd','--n-routed-experts','--experts-per-pool','--n-activated-experts') for arg in sys.argv[1:]):
        p.error('Set architecture dimensions in the recipe, not conflicting CLI overrides')

    if min(args.batch_size,args.seq_len,args.total_batch_size,args.max_steps,args.val_interval,args.save_interval,args.val_steps_per_pool)<1:
        p.error('Training sizes and intervals must be positive')
    if args.total_batch_size % (args.batch_size * args.seq_len) != 0:
        raise ValueError("total-batch-size must be divisible by batch-size * seq-len")

    if args.auto_restart and not args.dry_run:
        from asi.experiments.train_restart import supervise
        supervise(args, sys.argv[1:])
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    manifest = PoolManifest(args.pool_manifest)
    if recipe:
        from asi.models.design import validate_recipe, count_parameters
        if not manifest_matches(args.pool_manifest, recipe.get('pool_manifest_sha256')):
            p.error('Pool manifest differs from the frozen recipe')
        recipe_config=validate_recipe(recipe,manifest.names)
        recipe_config.block_size=recipe_config.max_seq_len=recipe_config.original_seq_len=args.seq_len
        recipe_config.max_batch_size=args.batch_size
        counts=count_parameters(asdict(recipe_config))
        if counts['total'] > recipe['parameter_budget']:
            p.error('Actual model exceeds total parameter budget')
        if args.dry_run:
            print(json.dumps({'family':recipe['family'],'parameters':counts,'pool_order':manifest.names,
                              'effective_training':vars(args)},indent=2,default=str)); return
    elif args.dry_run:
        p.error('--dry-run requires --recipe')
    if recipe:
        dataset_manifest=args.data_root/'manifest.json'
        if not manifest_matches(dataset_manifest, recipe['source_manifest_sha256']):
            p.error('data-root must contain the exact full dataset manifest used to design this recipe')
    if min(args.batch_size,args.seq_len,args.total_batch_size,args.max_steps,args.val_interval,args.save_interval,args.val_steps_per_pool)<1:
        p.error('Training sizes and intervals must be positive')
    grad_accum = args.total_batch_size // (args.batch_size * args.seq_len)
    print(f"device={args.device} | pools={len(manifest.names)} | experts/pool={recipe_config.experts_per_pool if recipe else args.experts_per_pool}")
    print(f"grad_accumulation_steps={grad_accum}")

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = GPTConfig(**resume_ckpt["config"])
        model = GPT(config)
        model.load_state_dict(resume_ckpt["model"])
        start_step = int(resume_ckpt["step"]) + 1
        if resume_ckpt.get('training_recipe') != recipe:
            p.error('Resume requires the same training recipe (including family and pool)')
        if recipe and asdict(config) != asdict(recipe_config):
            p.error('Resume model/context/batch configuration differs from the recipe')
    elif recipe:
        config=recipe_config
        model=GPT(config)
        start_step=0
    else:
        config = GPTConfig(
            block_size=args.seq_len,
            vocab_size=50304,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            inter_dim=1536,
            moe_inter_dim=224,
            n_dense_layers=1,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=2,
            n_activated_experts=args.n_activated_experts,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="softmax",
            route_scale=1.0,
            q_lora_rank=0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            max_seq_len=args.seq_len,
            original_seq_len=args.seq_len,
            n_pools=len(manifest.names),
            experts_per_pool=args.experts_per_pool,
        )
        model = GPT(config)
        start_step = 0

    model.training_recipe=recipe
    model.training_settings={k:getattr(args,k) for k in ('seed','pool_sampling','data_order','total_batch_size','batch_size','seq_len',
        'max_steps','max_lr','min_lr_factor','warmup_steps','weight_decay')}
    if resume_ckpt and resume_ckpt.get('training_settings') is not None and resume_ckpt['training_settings']!=model.training_settings:
        p.error('Resume must preserve sampling, token budget, batch, seed and learning-rate schedule')
    validate_pools(model, manifest)
    if resume_ckpt is not None and resume_ckpt.get("pool_names") != manifest.names:
        raise ValueError("Resume pool identity/order differs from the checkpoint")
    model.to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.max_lr, args.device)

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    train_data = PoolData(args.data_root, manifest, "train", args.batch_size, args.seq_len,args.data_order,args.seed)
    val_data = PoolData(args.data_root, manifest, "val", args.batch_size, args.seq_len,args.data_order,args.seed+100000)
    if recipe and recipe['family']=='router':
        name=recipe['pool']
        train_data.streams={k:v for k,v in train_data.streams.items() if k==name}
        val_data.streams={k:v for k,v in val_data.streams.items() if k==name}
    if recipe:
        required=[recipe['pool']] if recipe['family']=='router' else manifest.names
        if set(required)-set(train_data.names()) or set(required)-set(val_data.names()):
            raise ValueError('Required train/val pool shards are missing; verify the transferred dataset')
    validate_loader_against_model(train_data.names(), manifest)
    validate_loader_against_model(val_data.names(), manifest)

    scheduler = PoolScheduler(manifest, args.pool_sampling, args.seed)
    if resume_ckpt is not None:
        train_data.load_state(resume_ckpt.get("train_state", {}))
        if resume_ckpt.get('val_state') is not None:
            val_data.load_state(resume_ckpt['val_state'])
        if resume_ckpt.get("pool_rng_state") is not None:
            scheduler.load_state(resume_ckpt["pool_rng_state"])
        if resume_ckpt.get('torch_rng_state') is not None:
            torch.set_rng_state(resume_ckpt['torch_rng_state'])
        if torch.cuda.is_available() and resume_ckpt.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(resume_ckpt['cuda_rng_states'])

    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / 'run_metadata.json').write_text(json.dumps({
        'moe_backend': os.environ.get('ASI_MOE_BACKEND', 'auto'),
        'recipe': recipe, 'settings': model.training_settings, 'arguments': vars(args),
        'model': asdict(config), 'parameters': sum(p.numel() for p in model.parameters()),
        'validation_aggregation': 'equal mean over pools',
    }, indent=2, default=str), encoding='utf-8')
    log_file = args.log_dir / "log.txt"
    if start_step == 0:
        log_file.write_text("", encoding="utf-8")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
    print("Pool order:")
    for i, name in enumerate(manifest.names):
        print(f"  {i:02d} -> {name}")

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = step == args.max_steps - 1

        if step % args.val_interval == 0 or last_step:
            val_loss, by_pool = evaluate_by_pool(
                model, val_data, manifest, args.device, args.val_steps_per_pool
            )
            print(f"step {step}: overall val loss {val_loss:.4f}")
            print("  " + " | ".join(f"{k}={v:.4f}" for k, v in by_pool.items()))
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val_loss": val_loss, "val_by_pool": by_pool}) + "\n")
        else:
            val_loss = None

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        active_pool_name = scheduler.sample(train_data.names())
        active_pool_id = manifest.pool_id(active_pool_name)
        activate_training_pool(model, active_pool_id)

        for micro in range(grad_accum):
            x, y = train_data.next_batch(active_pool_name)
            x, y = x.to(args.device), y.to(args.device)
            with autocast_context(args.device):
                _, loss = model(x, y)
            loss = loss / grad_accum
            loss_accum += float(loss.detach().item())
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, args.max_lr, args.max_lr * args.min_lr_factor, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()

        if (step + 1) % args.save_interval == 0 or last_step:
            path = args.log_dir / f"model_{step:05d}.pt"
            save_checkpoint(
                path, model, optimizer, step, train_data, scheduler,
                val_loss, manifest.names, val_data
            )
            from asi.experiments.train_restart import publish_checkpoint
            publish_checkpoint(path, args.keep_checkpoints, os.environ.get('ASI_TRAIN_CHECKPOINT_STATUS'))
            print(f"saved {path}")

        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_sec = args.batch_size * args.seq_len * grad_accum / max(dt, 1e-9)
        print(
            f"step {step}, pool={active_pool_name}, loss={loss_accum:.4f}, "
            f"norm={float(norm):.4f}, lr={lr:.6f}, time={dt:.2f}s, tokens/sec={tokens_sec:.2f}"
        )
        with log_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "step": step,
                    "pool": active_pool_name,
                    "loss": loss_accum,
                    "norm": float(norm),
                    "lr": lr,
                    "time_sec": dt,
                    "tokens_sec": tokens_sec,
                }) + "\n"
            )


if __name__ == "__main__":
    main()
