"""Frozen base-expert cache with optional persistent LoRA recovery (5.16 layout)."""
from collections import Counter, OrderedDict
from types import MethodType
import torch
from torch.nn import functional as F


class OlmoeExpertCache:
    def __init__(self, model, device='cuda', capacity=128, store=None):
        if model.config.model_type != 'olmoe' or model.training or capacity < 1:
            raise ValueError('Expected eval-mode OLMoE and a positive expert capacity')
        self.model, self.device, self.capacity = model, torch.device(device), capacity
        self.store = store
        self.cache = OrderedDict()
        self.stats = Counter()
        self.fixed = None
        self.mixture = 'constant'
        self.active = model.config.num_experts_per_tok
        self.callbacks = []
        self.expert_adapter = None
        self.originals = []
        self.blocks = [layer.mlp for layer in model.model.layers]
        for block in self.blocks:
            if not hasattr(block.experts, 'gate_up_proj') or block.experts.gate_up_proj.ndim != 3:
                raise ValueError('Unsupported OLMoE layout; use the tested Transformers version')
        # Keep packed backing in RAM. Move only common tensors to GPU.
        expert_parameters = {id(p) for block in self.blocks for p in block.experts.parameters()}
        with torch.no_grad():
            for p in model.parameters():
                if store is not None and id(p) in expert_parameters:
                    continue
                p.data = p.data.to('cpu' if id(p) in expert_parameters else self.device)
            for module in model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    setattr(module, name, buffer.to(self.device))
        for layer, block in enumerate(self.blocks):
            self.originals.append((block.experts, block.experts.forward, block.gate, block.gate.forward))
            native = block.gate.forward
            def gate_forward(gate, hidden, lid=layer, original=native):
                if self.fixed is None:
                    logits, weights, indices = original(hidden)
                elif self.mixture == 'constant':
                    entry = self.fixed[str(lid)]
                    count = hidden.reshape(-1, hidden.shape[-1]).shape[0]
                    indices = torch.tensor(entry['experts'], device=hidden.device).expand(count, -1)
                    weights = torch.tensor(entry['weights'], device=hidden.device, dtype=hidden.dtype).expand(count, -1)
                    logits = hidden.new_zeros((count, model.config.num_experts))
                else:
                    # Keep the original full softmax scale, but restrict execution
                    # to resident experts. No expert weights are needed by the gate.
                    logits, native_weights, _ = original(hidden)
                    probabilities = torch.softmax(logits.float(), dim=-1)
                    allowed = torch.tensor(self.fixed[str(lid)]['experts'], device=hidden.device)
                    weights, positions = probabilities[:, allowed].topk(self.active, dim=-1)
                    indices = allowed[positions]
                    if self.mixture == 'restricted_native_mass':
                        weights = weights * native_weights.float().sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp_min(1e-20)
                    elif model.config.norm_topk_prob:
                        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
                    weights = weights.to(hidden.dtype)
                for callback in self.callbacks:
                    callback(lid, indices, weights)
                return logits, weights, indices
            def expert_forward(experts, hidden, indices, weights, lid=layer):
                out = torch.zeros_like(hidden)
                for expert_id in indices.unique().tolist():
                    tokens, slots = torch.where(indices == expert_id)
                    gate_up, down = self.get((lid, expert_id))
                    gate, up = F.linear(hidden[tokens], gate_up).chunk(2, dim=-1)
                    intermediate = experts.act_fn(gate) * up
                    values = F.linear(intermediate, down)
                    if self.expert_adapter is not None:
                        values = values + self.expert_adapter(lid, expert_id, intermediate)
                    values *= weights[tokens, slots, None]
                    out.index_add_(0, tokens, values.to(out.dtype))
                return out
            block.gate.forward = MethodType(gate_forward, block.gate)
            block.experts.forward = MethodType(expert_forward, block.experts)

    def get(self, key):
        self.stats['requests'] += 1
        if key in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.fixed is not None:
            raise RuntimeError('Fixed context missed an expert: refusing a transfer during generation')
        self.stats['misses'] += 1
        return self.load(key)

    def load(self, key):
        if key in self.cache:
            return self.cache[key]
        while len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
        layer, expert = key
        source = self.blocks[layer].experts
        backing = self.store.get(key) if self.store else (source.gate_up_proj[expert].detach(), source.down_proj[expert].detach())
        tensors = tuple(t.to(self.device) for t in backing)
        self.cache[key] = tensors
        self.stats['loads'] += 1
        if self.device.type == 'cuda':
            self.stats['host_to_device_bytes'] += sum(t.numel() * t.element_size() for t in tensors)
        return tensors

    def set_context(self, layers=None, mixture='constant', active=None):
        if mixture not in ('constant', 'restricted', 'restricted_native_mass'):
            raise ValueError('Unknown mixture policy')
        active = self.model.config.num_experts_per_tok if active is None else active
        if type(active) is not int or active < 1:
            raise ValueError('Active experts must be positive')
        if layers is None:
            self.fixed = None
            return
        required = set()
        if set(layers) != {str(i) for i in range(len(self.blocks))}:
            raise ValueError('Mapping must cover every layer')
        for layer, entry in layers.items():
            ids, weights = entry['experts'], entry['weights']
            if mixture != 'constant' and active > len(ids):
                raise ValueError('Active experts exceed resident candidates')
            if not ids or len(set(ids)) != len(ids) or len(weights) != len(ids):
                raise ValueError('Invalid fixed expert list')
            if any(type(e) is not int or not 0 <= e < self.model.config.num_experts for e in ids):
                raise ValueError('Invalid expert ID')
            if any(not isinstance(w, (int, float)) or not 0 <= w < float('inf') for w in weights) or sum(weights) <= 0:
                raise ValueError('Invalid mixture weights')
            required.update((int(layer), expert) for expert in ids)
        if len(required) > self.capacity:
            raise ValueError('Context does not fit hardware-selected expert capacity')
        if self.fixed != layers:
            self.stats['context_changes'] += 1
        for key in list(self.cache):
            if key not in required:
                del self.cache[key]
                self.stats['evictions'] += 1
        self.fixed = None
        for key in sorted(required):
            self.load(key)
        self.fixed = layers
        self.mixture, self.active = mixture, active
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def snapshot(self):
        return {**dict(self.stats), 'disk_ram': self.store.snapshot() if self.store else None, 'resident_experts': len(self.cache),
                'adapter_bytes': sum(p.numel()*p.element_size() for p in self.expert_adapter.parameters()) if self.expert_adapter is not None else 0,
                'resident_per_layer': {str(layer): sum(lid == layer for lid, _ in self.cache) for layer in range(len(self.blocks))},
                'resident_expert_bytes': sum(t.numel() * t.element_size() for pair in self.cache.values() for t in pair),
                'mixture': self.mixture, 'active_experts': self.active if self.mixture != 'constant' else None,
                'unit': 'one layer/expert, not one complete pool', 'fixed_context': self.fixed is not None}

    def close(self):
        for experts, expert_forward, gate, gate_forward in self.originals:
            experts.forward = expert_forward
            gate.forward = gate_forward
        self.originals.clear()
        self.cache.clear()
        self.callbacks.clear()
        self.expert_adapter = None


class CategoryCalibration:
    def __init__(self, layers, experts):
        self.layers, self.experts = layers, experts
        self.rows = {}
        self.labels = []

    def observe(self, layer, indices, weights):
        for label in self.labels:
            row = self.rows.setdefault((label, layer), {'tokens': 0, 'counts': torch.zeros(self.experts), 'mass': torch.zeros(self.experts)})
            ids = indices.detach().cpu().flatten()
            row['tokens'] += indices.shape[0]
            row['counts'] += torch.bincount(ids, minlength=self.experts)
            row['mass'].scatter_add_(0, ids, weights.detach().float().cpu().flatten())

    def mapping(self, labels, k):
        if not 1 <= k <= self.experts:
            raise ValueError('Invalid experts per category')
        output = {}
        for label in labels:
            output[label] = {}
            for layer in range(self.layers):
                row = self.rows[(label, layer)]
                # Preserve measured mean routing mass; OLMoE does not normally renormalize top-k.
                ids = sorted(range(self.experts), key=lambda e: (-float(row['counts'][e]), e))[:k]
                mass = row['mass'][ids]
                if float(mass.sum()) <= 0:
                    raise ValueError('No calibration evidence for selected experts')
                weights = mass / mass.sum() * row['mass'].sum() / row['tokens']
                output[label][str(layer)] = {'experts': ids, 'weights': weights.tolist(), 'calibration_tokens': row['tokens']}
        return output
