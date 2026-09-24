import os
import math
import time
import inspect
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Sequence

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from kernel import act_quant, weight_dequant, fp8_gemm
except ImportError:
    act_quant = weight_dequant = fp8_gemm = None


world_size = 1
rank = 0
kernel_block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class GPTConfig:
    # Common GPT-2 training fields
    block_size: int = 1024
    vocab_size: int = 50304  # padded GPT-2 tokenizer vocabulary
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512

    # DeepSeek-V3-inspired fields
    max_batch_size: int = 8
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None

    # Dense/MoE FFN
    inter_dim: int = 1536
    moe_inter_dim: int = 224
    n_dense_layers: int = 1
    n_routed_experts: int = 16
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    n_pools: int = 8
    experts_per_pool: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # RoPE / YaRN-compatible settings
    max_seq_len: int = 1024
    original_seq_len: int = 1024
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale, kernel_block_size)
        return F.linear(x, weight, bias)
    else:
        if act_quant is None or fp8_gemm is None:
            raise RuntimeError("FP8 GEMM requested but kernel.py is unavailable")
        x, scale = act_quant(x, kernel_block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.float32
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + kernel_block_size - 1) // kernel_block_size
            scale_in_features = (in_features + kernel_block_size - 1) // kernel_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: GPTConfig) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (GPTConfig): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: GPTConfig):
        super().__init__()
        self.dim = args.n_embd
        self.n_heads = args.n_head
        self.n_local_heads = args.n_head // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.wo.NANOGPT_SCALE_INIT = 1
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        # Training uses a direct, differentiable MLA path without KV-cache
        # assignments. The original inference implementation stores K/V in
        # buffers, which would sever the autograd graph during pretraining.
        if self.training:
            bsz, seqlen, _ = x.size()
            if freqs_cis is None:
                raise ValueError("freqs_cis is required during training")

            if self.q_lora_rank == 0:
                q = self.wq(x)
            else:
                q = self.wq_b(self.q_norm(self.wq_a(x)))
            q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_pe = apply_rotary_emb(q_pe, freqs_cis)

            kv = self.wkv_a(x)
            kv, k_pe = torch.split(
                kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(
                bsz,
                seqlen,
                self.n_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )

            k = torch.cat(
                [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1
            )
            q = torch.cat([q_nope, q_pe], dim=-1)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=(mask is None),
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(y)

        # Inference path: preserve the repository's original cache-aware MLA.
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if freqs_cis is None:
            freqs_cis = self._freqs_from_length(seqlen, x.device)

        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(
                bsz, seqlen, self.n_local_heads,
                self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat(
                [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1
            )
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum(
                "bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]
            ) * self.softmax_scale
        else:
            wkv_b = (
                self.wkv_b.weight
                if self.wkv_b.scale is None
                else weight_dequant(
                    self.wkv_b.weight, self.wkv_b.scale, kernel_block_size
                )
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim]
            )
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (
                torch.einsum(
                    "bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]
                )
                + torch.einsum(
                    "bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos]
                )
            ) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum(
                "bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos]
            )
        else:
            x = torch.einsum(
                "bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos]
            )
            x = torch.einsum(
                "bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:]
            )
        return self.wo(x.flatten(2))



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w2.NANOGPT_SCALE_INIT = 1
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """MoE router with an optional hard expert-pool mask."""

    def __init__(self, args: GPTConfig):
        super().__init__()
        self.dim = args.n_embd
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.routing_observer = None  # Optional diagnostic callback; no checkpoint state.

    def forward(
        self,
        x: torch.Tensor,
        allowed_expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = linear(x, self.weight)
        weights, indices = self.route_logits(logits, allowed_expert_mask)
        if self.routing_observer is not None:
            self.routing_observer(self, x, logits, weights, indices, allowed_expert_mask)
        return weights.type_as(x), indices

    def route_logits(self, logits, allowed_expert_mask=None):

        # IMPORTANT: mask the routing logits before softmax so the selected
        # expert weights are still normalized over the permitted pool(s).
        if allowed_expert_mask is not None:
            if allowed_expert_mask.dtype != torch.bool:
                allowed_expert_mask = allowed_expert_mask.to(torch.bool)
            if allowed_expert_mask.ndim != 1 or allowed_expert_mask.numel() != self.weight.size(0):
                raise ValueError(
                    f"allowed_expert_mask must have shape [{self.weight.size(0)}], "
                    f"got {tuple(allowed_expert_mask.shape)}"
                )
            if int(allowed_expert_mask.sum().item()) < self.topk:
                raise ValueError(
                    f"Active pool set exposes fewer experts ({int(allowed_expert_mask.sum().item())}) "
                    f"than n_activated_experts={self.topk}."
                )
            logits = logits.masked_fill(~allowed_expert_mask.unsqueeze(0), float("-inf"))

        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits.sigmoid()

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        # Bias correction must never resurrect a masked expert (sigmoid(-inf)=0).
        if allowed_expert_mask is not None:
            scores = scores.masked_fill(~allowed_expert_mask.unsqueeze(0), float("-inf"))
        if self.n_groups > 1:
            scores = scores.view(logits.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                top = scores.topk(min(2, scores.size(-1)), dim=-1)[0]
                group_scores = top.masked_fill(~torch.isfinite(top), 0).sum(dim=-1)
                group_scores = group_scores.masked_fill(~torch.isfinite(scores).any(-1), float("-inf"))
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(logits.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
            if (torch.isfinite(scores).sum(-1) < self.topk).any():
                raise ValueError("Group routing leaves fewer allowed experts than top-k")

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Standard SwiGLU expert used by the routed MoE."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Pool-restricted MoE.

    Routed experts are partitioned into fixed semantic pools. The caller passes
    an expert mask for the active pool(s), while shared experts always remain
    active. The grouped-MM implementation from the previous V3 version is
    retained so the experiment changes routing semantics without throwing away
    the performance work already done on the MoE.
    """

    def __init__(self, args: GPTConfig):
        super().__init__()
        self.dim = args.n_embd
        assert args.n_routed_experts % world_size == 0, (
            "Number of routed experts must be divisible by world size "
            f"(world_size={world_size})"
        )
        if args.n_routed_experts != args.n_pools * args.experts_per_pool:
            raise ValueError(
                "For this prototype n_routed_experts must equal "
                "n_pools * experts_per_pool. "
                f"Got {args.n_routed_experts} != {args.n_pools} * {args.experts_per_pool}."
            )

        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.n_pools = args.n_pools
        self.experts_per_pool = args.experts_per_pool
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.n_embd, args.moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(
            args.n_embd,
            args.n_shared_experts * args.moe_inter_dim,
        )

        expert_pool_ids = torch.arange(self.n_routed_experts, dtype=torch.long) // self.experts_per_pool
        self.register_buffer("expert_pool_ids", expert_pool_ids, persistent=False)

        self._grouped_mm_available = hasattr(F, "grouped_mm")
        self._backend_reported = False

    def allowed_mask_for_pools(self, pool_ids: Sequence[int]) -> torch.Tensor:
        ids = torch.as_tensor(list(pool_ids), dtype=torch.long, device=self.expert_pool_ids.device)
        if ids.numel() == 0:
            raise ValueError("At least one pool must be active.")
        mask = torch.zeros(
            self.n_routed_experts,
            dtype=torch.bool,
            device=self.expert_pool_ids.device,
        )
        for pool_id in ids.tolist():
            if pool_id < 0 or pool_id >= self.n_pools:
                raise ValueError(f"Invalid pool id {pool_id}; expected [0, {self.n_pools}).")
            mask |= self.expert_pool_ids == pool_id
        return mask

    def _stack_local_weights(self, active_ids):
        local_experts = [
            self.experts[i]
            for i in active_ids
        ]
        if any(expert is None for expert in local_experts):
            raise RuntimeError("Missing local expert in MoE parameter layout.")
        w1 = torch.stack([expert.w1.weight for expert in local_experts], dim=0)
        w2 = torch.stack([expert.w2.weight for expert in local_experts], dim=0)
        w3 = torch.stack([expert.w3.weight for expert in local_experts], dim=0)
        return w1, w2, w3

    @staticmethod
    def _grouped_linear(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError("grouped_mm is unavailable")
        x = x.contiguous()
        weights = weights.transpose(1, 2).contiguous()
        return F.grouped_mm(x, weights, offs=offsets, out_dtype=x.dtype)

    def _forward_grouped(self, x, weights, indices, w1, w2, w3):
        num_tokens, top_k = indices.shape
        token_ids = torch.arange(num_tokens, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, top_k).reshape(-1)
        expert_ids = indices.reshape(-1)
        route_weights = weights.reshape(-1)
        order = torch.argsort(expert_ids)
        expert_ids = expert_ids[order]
        token_ids = token_ids[order]
        route_weights = route_weights[order]
        local_mask = (expert_ids >= 0) & (expert_ids < w1.size(0))
        expert_ids = expert_ids[local_mask]
        token_ids = token_ids[local_mask]
        route_weights = route_weights[local_mask]
        if expert_ids.numel() == 0:
            return torch.zeros_like(x)

        local_counts = torch.bincount(expert_ids, minlength=w1.size(0))
        active_experts = torch.nonzero(local_counts > 0, as_tuple=False).flatten()
        counts = local_counts[active_experts]
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        x_sorted = x[token_ids]
        w1_active = w1[active_experts]
        w2_active = w2[active_experts]
        w3_active = w3[active_experts]
        if w1_active.dtype != x_sorted.dtype:
            w1_active = w1_active.to(x_sorted.dtype)
            w2_active = w2_active.to(x_sorted.dtype)
            w3_active = w3_active.to(x_sorted.dtype)
        up = self._grouped_linear(x_sorted, w1_active, offsets)
        gate = self._grouped_linear(x_sorted, w3_active, offsets)
        hidden = F.silu(up) * gate
        down = self._grouped_linear(hidden, w2_active, offsets)
        down = down * route_weights.to(down.dtype).unsqueeze(-1)
        y = torch.zeros_like(x)
        y.index_add_(0, token_ids, down)
        return y

    def _forward_bmm(self, x, weights, indices, w1, w2, w3):
        num_tokens, top_k = indices.shape
        token_ids = torch.arange(num_tokens, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, top_k).reshape(-1)
        expert_ids = indices.reshape(-1)
        route_weights = weights.reshape(-1)
        order = torch.argsort(expert_ids)
        expert_ids = expert_ids[order]
        token_ids = token_ids[order]
        route_weights = route_weights[order]
        local_mask = (expert_ids >= 0) & (expert_ids < w1.size(0))
        expert_ids = expert_ids[local_mask]
        token_ids = token_ids[local_mask]
        route_weights = route_weights[local_mask]
        if expert_ids.numel() == 0:
            return torch.zeros_like(x)
        local_counts = torch.bincount(expert_ids, minlength=w1.size(0))
        active_experts = torch.nonzero(local_counts > 0, as_tuple=False).flatten()
        counts = local_counts[active_experts]
        max_count = int(counts.max().item())
        x_batched = torch.zeros(active_experts.numel(), max_count, self.dim, dtype=x.dtype, device=x.device)
        weight_batched = torch.zeros(active_experts.numel(), max_count, dtype=weights.dtype, device=x.device)
        original_tokens = torch.full((active_experts.numel(), max_count), -1, dtype=torch.long, device=x.device)
        pos = 0
        for group_id, expert_id in enumerate(active_experts.tolist()):
            count = int(counts[group_id].item())
            end = pos + count
            ids = token_ids[pos:end]
            x_batched[group_id, :count] = x[ids]
            weight_batched[group_id, :count] = route_weights[pos:end]
            original_tokens[group_id, :count] = ids
            pos = end
        w1b = w1[active_experts].to(x.dtype)
        w2b = w2[active_experts].to(x.dtype)
        w3b = w3[active_experts].to(x.dtype)
        up = torch.bmm(x_batched, w1b.transpose(1, 2))
        gate = torch.bmm(x_batched, w3b.transpose(1, 2))
        hidden = F.silu(up) * gate
        down = torch.bmm(hidden, w2b.transpose(1, 2))
        down *= weight_batched.unsqueeze(-1).to(down.dtype)
        y = torch.zeros_like(x)
        flat_ids = original_tokens.reshape(-1)
        flat_down = down.reshape(-1, self.dim)
        valid = flat_ids >= 0
        y.index_add_(0, flat_ids[valid], flat_down[valid])
        return y

    def forward(self, x: torch.Tensor, allowed_expert_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape = x.size()
        x = x.reshape(-1, self.dim)
        if allowed_expert_mask is not None:
            allowed_expert_mask = allowed_expert_mask.to(device=x.device)
        weights, indices = self.gate(x, allowed_expert_mask=allowed_expert_mask)
        active_ids = indices.unique()
        active_ids = active_ids[(active_ids >= self.experts_start_idx) & (active_ids < self.experts_end_idx)]
        if active_ids.numel() == 0:
            y = torch.zeros_like(x)
            if world_size > 1:
                dist.all_reduce(y)
            return (y + self.shared_experts(x)).view(shape)
        # Stack only executed experts. Cold experts may reside in host RAM;
        # inactive training parameters must keep grad=None so AdamW skips them.
        w1, w2, w3 = self._stack_local_weights(active_ids.tolist())
        remap = torch.full((self.n_routed_experts,), -1, device=x.device, dtype=torch.long)
        remap[active_ids] = torch.arange(active_ids.numel(), device=x.device)
        indices = remap[indices]
        use_grouped = x.is_cuda and self._grouped_mm_available
        if not self._backend_reported:
            print(f"MoE backend: {'grouped_mm' if use_grouped else 'batched_bmm'} | pool-restricted routing enabled")
            self._backend_reported = True
        if use_grouped:
            try:
                y = self._forward_grouped(x, weights, indices, w1, w2, w3)
            except (RuntimeError, NotImplementedError):
                y = self._forward_bmm(x, weights, indices, w1, w2, w3)
        else:
            y = self._forward_bmm(x, weights, indices, w1, w2, w3)
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: GPTConfig):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (GPTConfig): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.n_embd, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.n_embd)
        self.ffn_norm = RMSNorm(args.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        active_expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        ffn_input = self.ffn_norm(x)
        if isinstance(self.ffn, MoE):
            x = x + self.ffn(ffn_input, allowed_expert_mask=active_expert_mask)
        else:
            x = x + self.ffn(ffn_input)
        return x


class GPT(nn.Module):
    """DeepSeek-V3-inspired decoder with pool-restricted routed experts."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if config.n_routed_experts != config.n_pools * config.experts_per_pool:
            raise ValueError(
                f"n_routed_experts={config.n_routed_experts} must equal "
                f"n_pools({config.n_pools}) * experts_per_pool({config.experts_per_pool})"
            )

        Linear.dtype = torch.float32
        Linear.scale_fmt = config.scale_fmt
        self.max_seq_len = config.block_size
        self.embed = ParallelEmbedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(layer_id, config) for layer_id in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.head = ColumnParallelLinear(config.n_embd, config.vocab_size, dtype=torch.float32)
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)
        expert_pool_ids = torch.arange(config.n_routed_experts, dtype=torch.long) // config.experts_per_pool
        self.register_buffer("expert_pool_ids", expert_pool_ids, persistent=False)
        self.register_buffer("active_expert_mask", torch.ones(config.n_routed_experts, dtype=torch.bool), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Linear)):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, ParallelEmbedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_expert_mask(self, pool_ids: Sequence[int]) -> torch.Tensor:
        if not pool_ids:
            raise ValueError("pool_ids must not be empty")
        mask = torch.zeros(self.config.n_routed_experts, dtype=torch.bool, device=self.expert_pool_ids.device)
        for pool_id in pool_ids:
            if not 0 <= int(pool_id) < self.config.n_pools:
                raise ValueError(f"Invalid pool id {pool_id}")
            mask |= self.expert_pool_ids == int(pool_id)
        if int(mask.sum().item()) < self.config.n_activated_experts:
            raise ValueError("Active pools expose fewer experts than n_activated_experts")
        return mask

    @torch.no_grad()
    def set_active_pools(self, pool_ids: Sequence[int]) -> torch.Tensor:
        mask = self.build_expert_mask(pool_ids)
        self.active_expert_mask = mask.to(device=self.embed.weight.device)
        return self.active_expert_mask

    @torch.no_grad()
    def set_all_experts_active(self) -> None:
        self.active_expert_mask = torch.ones(
            self.config.n_routed_experts,
            dtype=torch.bool,
            device=self.embed.weight.device,
        )

    def forward(
        self,
        idx,
        targets=None,
        start_pos: int = 0,
        active_expert_mask: Optional[torch.Tensor] = None,
    ):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        x = self.embed(idx)
        freqs_cis = self.freqs_cis[start_pos:start_pos + T].to(idx.device)
        mask = None
        if not self.training and T > 1:
            mask = torch.full((T, start_pos + T), float("-inf"), device=idx.device).triu_(start_pos + 1)

        if active_expert_mask is None:
            active_expert_mask = self.active_expert_mask.to(idx.device)
        else:
            active_expert_mask = active_expert_mask.to(idx.device)

        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask, active_expert_mask=active_expert_mask)

        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
