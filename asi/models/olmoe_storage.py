"""Direct safetensors backing, bounded adaptive RAM cache and meta initialization."""
from collections import Counter, OrderedDict
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open


class OlmoeDiskStore:
    def __init__(self, directory, ram_bytes, process_budget, reserve_bytes=512 * 1024**2):
        self.root = Path(directory).resolve()
        index = self.root / 'model.safetensors.index.json'
        if index.exists():
            self.index = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(str(self.root / 'model.safetensors'), framework='pt') as f:
                self.index = {key: 'model.safetensors' for key in f.keys()}
        for name in self.index.values():
            if not (self.root / name).resolve().is_relative_to(self.root):
                raise ValueError('Checkpoint shard escapes model directory')
        self.limit, self.process_budget, self.reserve = ram_bytes, process_budget, reserve_bytes
        self.cache, self.stats, self.bytes = OrderedDict(), Counter(), 0

    def tensor(self, name):
        with safe_open(str(self.root / self.index[name]), framework='pt', device='cpu') as f:
            # Own storage: closing the mapping must release unrelated file pages.
            value = f.get_tensor(name).clone()
        self.stats['logical_read_bytes'] += value.numel() * value.element_size()
        return value

    def trim(self):
        import psutil
        available = psutil.virtual_memory().available
        rss = psutil.Process().memory_info().rss
        target = max(0, min(self.limit, self.bytes + available - self.reserve,
                            self.bytes + self.process_budget - rss - self.reserve))
        while self.cache and self.bytes > target:
            _, pair = self.cache.popitem(last=False)
            self.bytes -= sum(t.numel() * t.element_size() for t in pair)
            self.stats['ram_evictions'] += 1
        return target

    def get(self, key):
        target = self.trim()
        if key in self.cache:
            self.stats['ram_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.stats['ram_misses'] += 1
        layer, expert = key
        prefix = f'model.layers.{layer}.mlp.experts.{expert}'
        start = time.perf_counter()
        gate, up = self.tensor(prefix + '.gate_proj.weight'), self.tensor(prefix + '.up_proj.weight')
        pair = (torch.cat((gate, up), dim=0), self.tensor(prefix + '.down_proj.weight'))
        self.stats['read_seconds'] += time.perf_counter() - start
        size = sum(t.numel() * t.element_size() for t in pair)
        while self.cache and self.bytes + size > target:
            _, old = self.cache.popitem(last=False)
            self.bytes -= sum(t.numel() * t.element_size() for t in old)
            self.stats['ram_evictions'] += 1
        if size <= target:
            self.cache[key] = pair
            self.bytes += size
        self.stats['peak_ram_weight_bytes'] = max(self.stats['peak_ram_weight_bytes'], self.bytes)
        return pair

    def snapshot(self):
        return {**self.stats, 'ram_weight_bytes': self.bytes, 'ram_limit_bytes': self.limit,
                'ram_resident_experts': len(self.cache), 'policy': 'LRU with available-memory pressure trimming',
                'note': 'Logical reads, not physical disk IO; cached tensors exclude transient expert buffers and OS page cache.'}


def load_disk_backed(directory, ram_bytes, process_budget, device='cuda'):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'olmoe':
        raise ValueError('Expected OLMoE checkpoint')
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager').eval()
        # Cast parameters only: rotary-frequency buffers must retain FP32 precision.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    store = OlmoeDiskStore(directory, ram_bytes, process_budget)
    for name, parameter in model.named_parameters():
        if '.mlp.experts.' not in name:
            set_module_tensor_to_device(model, name, device, value=store.tensor(name), dtype=torch.bfloat16)
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.to(device))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    return model, tokenizer, store
