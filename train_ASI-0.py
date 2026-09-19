"""
========

CED-124M: DeepSeek-V4.1-Flash-inspired Causal Encoder-Decoder

Architecture
------------
6 causal encoder layers
6 decoder layers

Hidden size:       768
Attention heads:   12
Head dimension:    64
FFN:               SwiGLU, 1792
Normalization:     RMSNorm
Position encoding: RoPE

Decoder:
    Q       <- decoder hidden state
    global K/V <- final encoder hidden state
    local K/V  <- decoder hidden state
    local attention uses a sliding window

This is intentionally a simplified CED architecture.
CSA2, mHC, Engram and other DeepSeek-V4.1 components
are NOT included yet. They should be introduced later
as separate experiments.

Approximate parameters:
    123.55M with GPT-2 vocabulary (50257)
"""

import os
import math
import time
import glob
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken


# ============================================================
# Configuration
# ============================================================

@dataclass
class CEDConfig:

    # Vocabulary / sequence
    vocab_size: int = 50257
    block_size: int = 1024

    # Architecture
    n_embd: int = 768
    n_head: int = 12

    encoder_layers: int = 6
    decoder_layers: int = 6

    # SwiGLU
    ffn_hidden: int = 1792

    # Sliding window
    window_size: int = 256

    # Dropout
    dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0

    # Training
    learning_rate: float = 3e-4
    min_lr: float = 3e-5

    warmup_steps: int = 715
    max_steps: int = 19073

    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Effective batch size
    total_batch_size: int = 524288

    # Micro batch
    batch_size: int = 4

    # Evaluation
    eval_interval: int = 250
    eval_iters: int = 50

    # Checkpoints
    checkpoint_interval: int = 1000

    # Data
    data_dir: str = "edu_fineweb10B"

    # Output
    out_dir: str = "log"


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("using device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

    torch.set_float32_matmul_precision("high")


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):

        norm = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(norm + self.eps)

        return self.weight * x


# ============================================================
# Rotary Positional Embeddings
# ============================================================

def precompute_rope(dim, max_seq_len, theta=10000.0, device=None):

    # Half dimension because we rotate pairs.
    freqs = torch.arange(
        0,
        dim,
        2,
        dtype=torch.float32,
        device=device,
    )

    freqs = 1.0 / (
        theta ** (freqs / dim)
    )

    positions = torch.arange(
        max_seq_len,
        dtype=torch.float32,
        device=device,
    )

    angles = torch.outer(
        positions,
        freqs,
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):

    """
    x:
        [B, H, T, D]

    cos/sin:
        [T, D/2]
    """

    B, H, T, D = x.shape

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    out = torch.empty_like(x)

    out[..., ::2] = rotated_x1
    out[..., 1::2] = rotated_x2

    return out


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(self, n_embd, hidden):

        super().__init__()

        self.gate = nn.Linear(
            n_embd,
            hidden,
            bias=False,
        )

        self.up = nn.Linear(
            n_embd,
            hidden,
            bias=False,
        )

        self.down = nn.Linear(
            hidden,
            n_embd,
            bias=False,
        )

    def forward(self, x):

        return self.down(
            F.silu(self.gate(x)) * self.up(x)
        )


# ============================================================
# Causal Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.dropout = config.dropout

        self.register_buffer(
            "rope_cos",
            precompute_rope(
                self.head_dim,
                config.block_size,
                config.rope_theta,
            ),
            persistent=False,
        )

        self.register_buffer(
            "rope_sin",
            precompute_rope(
                self.head_dim,
                config.block_size,
                config.rope_theta,
            )[1],
            persistent=False,
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
        )

        # PyTorch SDPA automatically applies causal masking.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C,
        )

        return self.out_proj(y)


# ============================================================
# Sliding Window Attention
# ============================================================

class SlidingWindowAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.window_size = config.window_size

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.dropout = config.dropout

        cos, sin = precompute_rope(
            self.head_dim,
            config.block_size,
            config.rope_theta,
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False,
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
        )

        # ----------------------------------------------------
        # Causal sliding-window mask
        #
        # Position i can attend to:
        #
        # max(0, i-window+1) ... i
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=x.device,
        )

        mask = (
            positions[None, :]
            <= positions[:, None]
        )

        mask = mask & (
            positions[None, :]
            >= (
                positions[:, None]
                - self.window_size
                + 1
            )
        )

        # [T,T] -> [1,1,T,T]
        mask = mask.unsqueeze(0).unsqueeze(0)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C,
        )

        return self.out_proj(y)


# ============================================================
# Encoder Block
# ============================================================

class EncoderBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.norm1 = RMSNorm(
            config.n_embd
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.norm2 = RMSNorm(
            config.n_embd
        )

        self.ffn = SwiGLU(
            config.n_embd,
            config.ffn_hidden,
        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# Decoder Attention
# ============================================================

class CEDDecoderAttention(nn.Module):

    """
    CED attention.

    Query:
        decoder hidden state

    Global K/V:
        final encoder hidden state

    Local K/V:
        decoder hidden state

    The global branch is causally masked so that position t
    cannot attend to encoder positions > t.

    This prevents future-token leakage during autoregressive
    training.
    """

    def __init__(self, config):

        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Decoder Q
        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        # Global encoder KV
        self.global_k_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.global_v_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        # Local decoder KV
        self.local_k_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.local_v_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.window_size = config.window_size
        self.dropout = config.dropout

        cos, sin = precompute_rope(
            self.head_dim,
            config.block_size,
            config.rope_theta,
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False,
        )

    def forward(
        self,
        x,
        encoder_output,
    ):

        B, T, C = x.shape

        # ====================================================
        # Query
        # ====================================================

        q = self.q_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # ====================================================
        # Global K/V
        #
        # Generated from final encoder state.
        # ====================================================

        gk = self.global_k_proj(
            encoder_output
        )

        gv = self.global_v_proj(
            encoder_output
        )

        gk = gk.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        gv = gv.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # ====================================================
        # Local K/V
        # ====================================================

        lk = self.local_k_proj(x)
        lv = self.local_v_proj(x)

        lk = lk.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        lv = lv.view(
            B,
            T,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # ====================================================
        # RoPE
        # ====================================================

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        gk = apply_rope(
            gk,
            self.rope_cos,
            self.rope_sin,
        )

        lk = apply_rope(
            lk,
            self.rope_cos,
            self.rope_sin,
        )

        # ====================================================
        # GLOBAL ATTENTION
        #
        # Decoder position i can only see encoder positions
        # <= i.
        # ====================================================

        positions = torch.arange(
            T,
            device=x.device,
        )

        global_mask = (
            positions[None, :]
            <= positions[:, None]
        )

        global_mask = (
            global_mask
            .unsqueeze(0)
            .unsqueeze(0)
        )

        global_y = F.scaled_dot_product_attention(
            q,
            gk,
            gv,
            attn_mask=global_mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=False,
        )

        # ====================================================
        # LOCAL SWA
        # ====================================================

        local_mask = (
            positions[None, :]
            <= positions[:, None]
        )

        local_mask = local_mask & (
            positions[None, :]
            >= (
                positions[:, None]
                - self.window_size
                + 1
            )
        )

        local_mask = (
            local_mask
            .unsqueeze(0)
            .unsqueeze(0)
        )

        local_y = F.scaled_dot_product_attention(
            q,
            lk,
            lv,
            attn_mask=local_mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=False,
        )

        # ====================================================
        # Combine global + local branches
        # ====================================================

        y = global_y + local_y

        y = y.transpose(
            1,
            2,
        ).contiguous()

        y = y.view(
            B,
            T,
            C,
        )

        return self.out_proj(y)


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.norm1 = RMSNorm(
            config.n_embd
        )

        self.attn = CEDDecoderAttention(
            config
        )

        self.norm2 = RMSNorm(
            config.n_embd
        )

        self.ffn = SwiGLU(
            config.n_embd,
            config.ffn_hidden,
        )

    def forward(
        self,
        x,
        encoder_output,
    ):

        x = x + self.attn(
            self.norm1(x),
            encoder_output,
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# CED Model
# ============================================================

class CEDModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.ModuleList([
            EncoderBlock(config)
            for _ in range(
                config.encoder_layers
            )
        ])

        self.encoder_norm = RMSNorm(
            config.n_embd
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.ModuleList([
            DecoderBlock(config)
            for _ in range(
                config.decoder_layers
            )
        ])

        self.decoder_norm = RMSNorm(
            config.n_embd
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    # ========================================================
    # Initialization
    # ========================================================

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        idx,
        targets=None,
    ):

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block size {self.config.block_size}"
            )

        # Shared token embeddings.
        x = self.token_embedding(idx)

        # ====================================================
        # ENCODER
        # ====================================================

        encoder_output = x

        for block in self.encoder:

            encoder_output = block(
                encoder_output
            )

        encoder_output = self.encoder_norm(
            encoder_output
        )

        # ====================================================
        # DECODER
        # ====================================================

        decoder_output = x

        for block in self.decoder:

            decoder_output = block(
                decoder_output,
                encoder_output,
            )

        decoder_output = self.decoder_norm(
            decoder_output
        )

        # ====================================================
        # Language modeling head
        # ====================================================

        logits = self.lm_head(
            decoder_output
        )

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    # ========================================================
    # Parameter count
    # ========================================================

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


# ============================================================
# Data loader
# ============================================================

def get_shard_files():

    files = sorted(
        glob.glob(
            os.path.join(
                C.data_dir,
                "*.bin",
            )
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No .bin files found in {C.data_dir}"
        )

    return files


def load_tokens(filename):

    return torch.from_numpy(
        __import__("numpy").fromfile(
            filename,
            dtype=__import__("numpy").uint16,
        ).astype("int64")
    )


def get_batch(
    split,
    batch_size,
    block_size,
):

    if split == "train":

        shard_files = train_shards

    else:

        shard_files = val_shards

    # Randomly select a shard.
    filename = shard_files[
        torch.randint(
            0,
            len(shard_files),
            (1,),
        ).item()
    ]

    tokens = load_tokens(
        filename
    )

    if len(tokens) <= block_size + 1:

        raise RuntimeError(
            f"Shard too small: {filename}"
        )

    max_start = (
        len(tokens)
        - block_size
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
    )

    x = torch.stack([
        tokens[
            start:
            start + block_size
        ]
        for start in starts
    ])

    y = torch.stack([
        tokens[
            start + 1:
            start + block_size + 1
        ]
        for start in starts
    ])

    return (
        x.to(device),
        y.to(device),
    )


# ============================================================
# Learning rate schedule
# ============================================================

def get_lr(step):

    if step < C.warmup_steps:

        return (
            C.learning_rate
            * (step + 1)
            / C.warmup_steps
        )

    if step >= C.max_steps:

        return C.min_lr

    decay_ratio = (
        step - C.warmup_steps
    ) / (
        C.max_steps
        - C.warmup_steps
    )

    coeff = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * decay_ratio
            )
        )
    )

    return (
        C.min_lr
        + coeff
        * (
            C.learning_rate
            - C.min_lr
        )
    )


# ============================================================
# Optimizer
# ============================================================

def configure_optimizer(model):

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": C.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=C.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(model):

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(
            C.eval_iters,
            device=device,
        )

        for k in range(C.eval_iters):

            X, Y = get_batch(
                split,
                C.batch_size,
                C.block_size,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    device == "cuda"
                    and torch.cuda.is_bf16_supported()
                ),
            ):

                _, loss = model(
                    X,
                    Y,
                )

            losses[k] = loss.detach()

        results[split] = (
            losses.mean().item()
        )

    model.train()

    return results


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    best_val,
):

    os.makedirs(
        C.out_dir,
        exist_ok=True,
    )

    path = os.path.join(
        C.out_dir,
        f"model_{step:05d}.pt",
    )

    checkpoint = {

        "model": model.state_dict(),

        "config": {
            k: v
            for k, v in C.__dict__.items()
        },

        "step": step,

        "best_val": best_val,

        "optimizer": optimizer.state_dict(),

    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"saved checkpoint: {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    global C
    global train_shards
    global val_shards

    C = CEDConfig()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    shards = get_shard_files()

    # Last shard is treated as validation.
    #
    # If your current dataset already has explicit train/val
    # shard naming, change this section to match it.
    #
    # For example:
    # train_shards = [...]
    # val_shards = [...]
    #
    # --------------------------------------------------------

    train_shards = [
        f
        for f in shards
        if "val" not in os.path.basename(f).lower()
    ]

    val_shards = [
        f
        for f in shards
        if "val" in os.path.basename(f).lower()
    ]

    if not val_shards:

        # Fallback: use the last shard for validation.
        val_shards = [train_shards[-1]]
        train_shards = train_shards[:-1]

    print(
        f"train shards: {len(train_shards)}"
    )

    print(
        f"validation shards: {len(val_shards)}"
    )

    # --------------------------------------------------------
    # Effective batch size
    # --------------------------------------------------------

    tokens_per_microbatch = (
        C.batch_size
        * C.block_size
    )

    grad_accum_steps = (
        C.total_batch_size
        // tokens_per_microbatch
    )

    assert (
        C.total_batch_size
        % tokens_per_microbatch
        == 0
    )

    print()
    print(
        "tokens per microbatch:",
        tokens_per_microbatch,
    )

    print(
        "gradient accumulation:",
        grad_accum_steps,
    )

    print(
        "effective batch:",
        C.total_batch_size,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CEDModel(
        C
    ).to(device)

    n_params = model.num_parameters()

    print()
    print("================================================")
    print("CED-124M")
    print("================================================")
    print(
        f"parameters: {n_params:,}"
    )
    print(
        f"parameters (M): {n_params / 1e6:.3f}"
    )

    print(
        f"encoder layers: {C.encoder_layers}"
    )

    print(
        f"decoder layers: {C.decoder_layers}"
    )

    print(
        f"hidden size: {C.n_embd}"
    )

    print(
        f"heads: {C.n_head}"
    )

    print(
        f"head dimension: "
        f"{C.n_embd // C.n_head}"
    )

    print(
        f"FFN hidden: {C.ffn_hidden}"
    )

    print(
        f"SWA window: {C.window_size}"
    )

    print("================================================")
    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = configure_optimizer(
        model
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    use_bf16 = (
        device == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print(
        "bfloat16:",
        use_bf16,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val = float("inf")

    model.train()

    for step in range(
        C.max_steps
    ):

        t0 = time.time()

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_lr(step)

        for param_group in optimizer.param_groups:

            param_group["lr"] = lr

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_accum = 0.0

        for micro_step in range(
            grad_accum_steps
        ):

            X, Y = get_batch(
                "train",
                C.batch_size,
                C.block_size,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):

                _, loss = model(
                    X,
                    Y,
                )

                loss = (
                    loss
                    / grad_accum_steps
                )

            loss_accum += loss.detach()

            loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            C.grad_clip,
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        if device == "cuda":

            torch.cuda.synchronize()

        dt = time.time() - t0

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        tokens_per_second = (
            C.total_batch_size
            / dt
        )

        print(
            f"step {step:5d} | "
            f"loss {loss_accum.item():.4f} | "
            f"lr {lr:.3e} | "
            f"grad {grad_norm:.2f} | "
            f"{tokens_per_second:,.0f} tok/s | "
            f"{dt:.2f}s"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            step % C.eval_interval == 0
            or step == C.max_steps - 1
        ):

            losses = estimate_loss(
                model
            )

            print(
                f"validation | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            if losses["val"] < best_val:

                best_val = losses["val"]

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    best_val,
                )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        elif (
            step > 0
            and step % C.checkpoint_interval == 0
        ):

            save_checkpoint(
                model,
                optimizer,
                step,
                best_val,
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        model,
        optimizer,
        C.max_steps - 1,
        best_val,
    )

    print()
    print("Training finished.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()