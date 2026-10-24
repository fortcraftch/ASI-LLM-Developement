"""Synchronous profiling cache; timings are diagnostic, not deployment throughput."""
import time
import torch
from asi.models.hf_olmoe import OlmoeExpertCache


class TimedOlmoeCache(OlmoeExpertCache):
    def load(self, key):
        if key in self.cache:
            return self.cache[key]
        # Exclude preceding queued model compute from the measured load interval.
        torch.cuda.synchronize(self.device)
        total_start = time.perf_counter()
        while len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
        layer, expert = key
        source = self.blocks[layer].experts
        start = time.perf_counter()
        backing = self.store.get(key) if self.store else (source.gate_up_proj[expert].detach(), source.down_proj[expert].detach())
        self.stats['weight_fetch_seconds'] += time.perf_counter() - start
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        tensors = tuple(t.to(self.device) for t in backing)
        end.record()
        end.synchronize()
        self.stats['h2d_cuda_seconds'] += begin.elapsed_time(end) / 1000
        self.stats['h2d_wall_seconds'] += time.perf_counter() - wall
        self.cache[key] = tensors
        self.stats['loads'] += 1
        self.stats['host_to_device_bytes'] += sum(t.numel() * t.element_size() for t in tensors)
        self.stats['profiled_load_seconds'] += time.perf_counter() - total_start
        return tensors
