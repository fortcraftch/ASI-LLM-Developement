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
from dataclasses import asdict
from pathlib import Path
from asi import DATA_ROOT
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from asi.models.domain import GPT, GPTConfig


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


class PoolData:
    def __init__(self, data_root: Path, manifest: PoolManifest, split: str, B: int, T: int):
        self.streams = {}
        for pool_name in manifest.names:
            shards = []
            for category in manifest.categories(pool_name):
                category_dir = data_root / category
                shards.extend(p for p in sorted(category_dir.glob(f"{split}_*.npy")) if not p.name.endswith(".part.npy"))
            if shards:
                self.streams[pool_name] = TokenShardStream(shards, B, T)
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
        self.probs = [weights[n] / total for n in self.names]

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
    with torch.no_grad():
        for pool_name in manifest.names:
            if pool_name not in val_data.streams:
                continue
            pool_id = manifest.pool_id(pool_name)
            model.set_active_pools([pool_id])
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
    model.train()
    return overall, results


def save_checkpoint(path, model, optimizer, step, train_data, scheduler, val_loss, pool_names):
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "train_state": train_data.state(),
        "pool_rng_state": scheduler.state(),
        "pool_names": pool_names,
        "experiment": "domain_pool_restricted_moe_v1",
        "checkpoint_timing": "after_optimizer_step",
        "val_loss_timing": "before_this_step_update" if val_loss is not None else None,
    }
    torch.save(payload, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--pool-manifest", type=Path, default=(DATA_ROOT / "expert_pools.json"))
    p.add_argument("--log-dir", type=Path, default=Path("log_domain"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--total-batch-size", type=int, default=524288)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--val-steps-per-pool", type=int, default=1)
    p.add_argument("--pool-sampling", choices=["uniform", "token"], default="uniform")

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
    args = p.parse_args()

    if args.total_batch_size % (args.batch_size * args.seq_len) != 0:
        raise ValueError("total-batch-size must be divisible by batch-size * seq-len")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    manifest = PoolManifest(args.pool_manifest)
    grad_accum = args.total_batch_size // (args.batch_size * args.seq_len)
    print(f"device={args.device} | pools={len(manifest.names)} | experts/pool={args.experts_per_pool}")
    print(f"grad_accumulation_steps={grad_accum}")

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = GPTConfig(**resume_ckpt["config"])
        model = GPT(config)
        model.load_state_dict(resume_ckpt["model"])
        start_step = int(resume_ckpt["step"]) + 1
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

    validate_pools(model, manifest)
    if resume_ckpt is not None and resume_ckpt.get("pool_names") != manifest.names:
        raise ValueError("Resume pool identity/order differs from the checkpoint")
    model.to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.max_lr, args.device)

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    train_data = PoolData(args.data_root, manifest, "train", args.batch_size, args.seq_len)
    val_data = PoolData(args.data_root, manifest, "val", args.batch_size, args.seq_len)
    validate_loader_against_model(train_data.names(), manifest)
    validate_loader_against_model(val_data.names(), manifest)

    scheduler = PoolScheduler(manifest, args.pool_sampling, args.seed)
    if resume_ckpt is not None:
        train_data.load_state(resume_ckpt.get("train_state", {}))
        if resume_ckpt.get("pool_rng_state") is not None:
            scheduler.load_state(resume_ckpt["pool_rng_state"])

    args.log_dir.mkdir(parents=True, exist_ok=True)
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
        model.set_active_pools([active_pool_id])

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

        if step % args.save_interval == 0 and step > 0 or last_step:
            path = args.log_dir / f"model_{step:05d}.pt"
            save_checkpoint(
                path, model, optimizer, step, train_data, scheduler,
                val_loss, manifest.names
            )
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
