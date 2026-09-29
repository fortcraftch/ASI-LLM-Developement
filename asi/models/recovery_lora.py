"""Persistent FP32 low-rank updates for disk-backed frozen OLMoE experts."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ExpertLoRA(nn.Module):
    """One down-projection update per layer/expert, shared across all categories.

    These small parameters stay on the compute device independently of the base
    expert LRU. Never attach trainable tensors to evictable cache entries.
    """
    def __init__(self, model, mappings, rank=8):
        super().__init__()
        if rank < 1:
            raise ValueError('Positive LoRA rank required')
        self.rank = rank
        keys = sorted({(int(layer), expert)
                       for mapping in mappings.values() for layers in mapping.values()
                       for layer, entry in layers.items() for expert in entry['experts']})
        self.a, self.b = nn.ParameterDict(), nn.ParameterDict()
        device = model.model.layers[0].mlp.gate.weight.device
        for layer, expert in keys:
            key = self.key(layer, expert)
            a = torch.empty(rank, model.config.intermediate_size, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.a[key] = nn.Parameter(a)
            self.b[key] = nn.Parameter(torch.zeros(model.config.hidden_size, rank, device=device, dtype=torch.float32))

    @staticmethod
    def key(layer, expert):
        return f'l{layer}_e{expert}'

    def forward(self, layer, expert, hidden):
        key = self.key(layer, expert)
        if key not in self.a:
            raise ValueError('Expert absent from the trained adaptation bank: '+key)
        # LoRA alpha == rank, so the scale is one. Autocast uses BF16 compute.
        return F.linear(F.linear(hidden, self.a[key]), self.b[key])

    def cpu_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
