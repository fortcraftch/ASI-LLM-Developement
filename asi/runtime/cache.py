"""Inference-only expert cache: immutable RAM backing and bounded GPU residency."""
from __future__ import annotations

from collections import Counter, OrderedDict
import time
import torch
import torch.nn.functional as F
from asi.models.domain import Gate
from asi.models.original import native_moes
from asi.models.domain import MoE


def tensor_bytes(t):
    return t.numel() * t.element_size()


def module_memory(module):
    """Tensor inventory, not process RSS or Python/tokenizer overhead."""
    weights, buffers = Counter(), Counter()
    if module is not None:
        for p in module.parameters():
            weights[str(p.device)] += tensor_bytes(p)
        for b in module.buffers():
            buffers[str(b.device)] += tensor_bytes(b)
    return {"weight_bytes_by_device": dict(weights), "buffer_bytes_by_device": dict(buffers)}


def move_backbone(model, device):
    """Move non-expert tensors without ever staging the full model on GPU."""
    expert_modules = {id(m) for layer in model.layers if hasattr(layer.ffn, "experts")
                      for expert in layer.ffn.experts if expert is not None
                      for m in expert.modules()}
    with torch.no_grad():
        for module in model.modules():
            if id(module) in expert_modules:
                continue
            for p in module.parameters(recurse=False):
                p.data = p.data.to(device)
            for name, b in module.named_buffers(recurse=False):
                setattr(module, name, b.to(device))


class ExpertCacheManager:
    def __init__(self, device: str, max_hot_pools: int = 3, pin_memory: bool = False,
                 max_pinned_bytes: int = 512 * 1024**2):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if max_hot_pools < 1 or max_pinned_bytes < 0:
            raise ValueError("Cache capacity must be positive; pinned budget nonnegative")
        self.max_hot_pools = max_hot_pools
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.max_pinned_bytes = max_pinned_bytes
        self.resident_pools = set()
        self.stats = Counter()
        self.entries = {}
        self.backing = {}
        self.model = None

    @torch.no_grad()
    def initialize(self, model):
        if model.training:
            raise ValueError("Expert cache is inference-only; call model.eval() first")
        self.model = model
        self.entries.clear()
        self.backing.clear()
        self.stats.clear()
        self.resident_pools.clear()
        pinned = 0
        for layer_id, layer in enumerate(model.layers):
            if not isinstance(layer.ffn, MoE):
                continue
            for expert_id, expert in enumerate(layer.ffn.experts):
                if expert is None:
                    continue
                key = (layer_id, expert_id)
                pool = int(layer.ffn.expert_pool_ids[expert_id])
                expert.to("cpu")
                host = {}
                for name, p in expert.named_parameters():
                    t = p.detach()
                    size = tensor_bytes(t)
                    if self.pin_memory and pinned + size <= self.max_pinned_bytes:
                        t = t.pin_memory()
                        pinned += size
                    elif t.is_pinned():
                        t = torch.empty_like(t, device="cpu", pin_memory=False).copy_(t)
                    host[name] = t
                    p.data = t
                self.entries[key] = (pool, expert)
                self.backing[key] = host
        move_backbone(model, self.device)
        self.stats["pinned_bytes"] = pinned

    def _check(self, model):
        if model is not self.model or model.training:
            raise ValueError("Initialize this cache with the same model in eval mode")

    def _place(self, key, device):
        _, expert = self.entries[key]
        for name, p in expert.named_parameters():
            host = self.backing[key][name]
            p.data = host if device.type == "cpu" else host.to(device, non_blocking=host.is_pinned())

    @torch.no_grad()
    def prepare(self, model, required_pool_ids, ranked_pool_ids=None):
        self._check(model)
        required = list(dict.fromkeys(int(p) for p in required_pool_ids))
        if not required or len(required) > self.max_hot_pools:
            raise ValueError("Active pools must be nonempty and fit max_hot_pools")
        candidates = list(dict.fromkeys(required + list(ranked_pool_ids or [])))
        if any(p < 0 or p >= model.config.n_pools for p in candidates):
            raise ValueError("Invalid pool id")
        desired = candidates[:self.max_hot_pools]
        # Retain useful old pools in spare slots, without making them routable.
        desired += [p for p in sorted(self.resident_pools) if p not in desired][:self.max_hot_pools-len(desired)]
        hot = set(desired)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        # Evict before loading, so a domain change cannot double GPU residency.
        for key, (pool, _) in self.entries.items():
            if pool in self.resident_pools and pool not in hot:
                self._place(key, torch.device("cpu"))
                self.stats["evictions"] += 1
        for key, (pool, _) in self.entries.items():
            if pool not in hot:
                continue
            if pool in required:
                self.stats["requests"] += 1
                self.stats["hits" if pool in self.resident_pools else "misses"] += 1
            if pool not in self.resident_pools:
                self._place(key, self.device)
                if self.device.type == "cuda":
                    self.stats["host_to_device_bytes"] += sum(tensor_bytes(t) for t in self.backing[key].values())
                    self.stats["loads"] += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats["prepare_seconds"] += time.perf_counter() - start
        self.resident_pools = hot
        return desired

    @property
    def hit_rate(self):
        return self.stats["hits"] / self.stats["requests"] if self.stats["requests"] else 0.0

    def snapshot(self):
        return {**dict(self.stats), "hits": self.stats["hits"], "misses": self.stats["misses"],
                "evictions": self.stats["evictions"], "hit_rate": self.hit_rate,
                "resident_pools": sorted(self.resident_pools),
                "device": str(self.device),
                "counter_unit": "layer-expert requested at session preparation",
                "cpu_mode": "logical residency only" if self.device.type == "cpu" else None}

    def inventory(self):
        self._check(self.model)
        experts = []
        expert_params = set()
        for key, (pool, expert) in self.entries.items():
            params = list(expert.parameters())
            expert_params.update(id(p) for p in params)
            experts.append({"layer": key[0], "expert": key[1], "pool": pool,
                            "device": str(params[0].device),
                            "execution_weight_bytes": sum(tensor_bytes(p) for p in params),
                            "ram_backing_bytes": sum(tensor_bytes(t) for t in self.backing[key].values()),
                            "pinned_bytes": sum(tensor_bytes(t) for t in self.backing[key].values() if t.is_pinned())})
        base = Counter()
        for p in self.model.parameters():
            if id(p) not in expert_params:
                base[str(p.device)] += tensor_bytes(p)
        buffers = Counter()
        for b in self.model.buffers():
            buffers[str(b.device)] += tensor_bytes(b)
        report = {"experts": experts, "backbone_and_shared_weight_bytes": dict(base),
                  "buffers_including_kv_bytes": dict(buffers),
                  "ram_backing_bytes": sum(e["ram_backing_bytes"] for e in experts),
                  "note": "Cold execution weights alias RAM backing; do not sum them twice. CUDA allocator includes temporary tensors; classifier is separate."}
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            report["cuda"] = {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                              "reserved_bytes": torch.cuda.memory_reserved(self.device),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                              "device_free_bytes": free, "device_total_bytes": total}
        return report

    @torch.no_grad()
    def offload_all(self, model):
        self._check(model)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for key in self.entries:
            self._place(key, torch.device("cpu"))
        self.resident_pools.clear()


class NativeExpertSessionCache:
    """Bounded global LRU for native BF16 experts, with exact routing by default.

    Reviewed mapping: {"layers":{"3":{"programming":[0,7],"math":[7,12]}}}.
    In prefetch mode labels are hints; native router misses load on demand.
    In restrict mode labels define a hard per-layer mask (quality may change).
    Backbone must fit the execution device. Optional disk backing bounds cached RAM weights.
    Shared experts remain on their original device. No quantized/distributed support.
    """
    def __init__(self,model,mapping,device='cuda',max_hot_experts=256,
                 pin_memory=False,max_pinned_bytes=512*1024**2,policy='prefetch',
                 max_experts_per_layer=None, backing_store=None, fixed_weight_mode='uniform'):
        if policy not in ('prefetch','restrict','fixed'): raise ValueError('Unknown routing policy')
        if max_hot_experts<1: raise ValueError('Positive expert capacity required')
        self.moes=native_moes(model)
        self.mapping=mapping['layers']
        self.device=torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        self.capacity=max_hot_experts
        if max_experts_per_layer is not None and max_experts_per_layer < 1:
            raise ValueError('Positive per-layer capacity required')
        self.layer_capacity = max_experts_per_layer
        self.peak_by_layer = Counter()
        self.policy=policy
        self.labels=[]
        self.hot=OrderedDict()
        self.preferred=set()
        self.backing={}
        self.backing_store = backing_store
        self.experts={}
        self.stats=Counter()
        self.handles=[]
        self.model=model
        self.turn_demands = None
        # Validate everything before moving any tensors.
        for layer,moe in self.moes.items():
            if self.layer_capacity is not None and self.layer_capacity < moe.gate.topk:
                raise ValueError('Per-layer capacity cannot be smaller than router top-k')
            if moe.gate.weight.device != self.device:
                raise ValueError('Place the native backbone/router on the execution device before attaching the expert cache')
            for expert in moe.experts:
                if expert is None or any(p.element_size()==1 for p in expert.parameters()):
                    raise ValueError('Only native, unquantized single-rank experts are supported')
            for ids in self.mapping.get(str(layer),{}).values():
                if any(not isinstance(i,int) or not 0<=i<len(moe.experts) for i in ids):
                    raise ValueError(f'Invalid reviewed expert mapping at layer {layer}')
        if model.training: raise ValueError('Native expert cache requires eval mode')
        self.fixed_router = None
        if policy == 'fixed':
            from asi.runtime.routing import FixedExpertRouting
            self.fixed_router = FixedExpertRouting(self.moes, mapping, fixed_weight_mode)
            if self.capacity < sum(moe.gate.topk for moe in self.moes.values()):
                raise ValueError('Fixed N-of-N requires enough capacity to retain N experts in every layer')
        if backing_store is not None:
            if self.device.type != 'cuda':
                raise ValueError('Disk-backed execution currently requires CUDA to bound cold RAM aliases')
            expected = {(layer,eid) for layer,moe in self.moes.items() for eid in range(len(moe.experts))}
            if set(backing_store.entries) != expected:
                raise ValueError('Disk store expert identities do not match model')
        pinned=0
        with torch.no_grad():
            for layer,moe in self.moes.items():
                for eid,expert in enumerate(moe.experts):
                    key=(layer,eid); self.experts[key]=expert; self.backing[key]={}
                    if backing_store is not None:
                        spec = backing_store.entries[key]['parameters']
                        if set(spec) != {name for name, _ in expert.named_parameters()}:
                            raise ValueError('Disk store parameter names mismatch')
                        for name,p in expert.named_parameters():
                            if str(p.dtype) != spec[name]['dtype'] or (p.numel() and list(p.shape) != spec[name]['shape']):
                                raise ValueError('Disk store parameter shape/dtype mismatch')
                            p.data = torch.empty(0, dtype=p.dtype, device='cpu')
                        continue
                    for name,p in expert.named_parameters():
                        host=p.detach().cpu()
                        size=host.numel()*host.element_size()
                        if pin_memory and self.device.type=='cuda' and pinned+size<=max_pinned_bytes:
                            host=host.pin_memory(); pinned+=size
                        self.backing[key][name]=host; p.data=host
        self.stats['pinned_bytes']=pinned
        for layer,moe in self.moes.items():
            self.handles.append(moe.gate.register_forward_hook(lambda gate,args,out,lid=layer:self._route(lid,gate,args,out)))
        if self.fixed_router:
            self.fixed_router.attach()

    def _place(self,key,device):
        if self.backing_store is not None:
            if device.type == 'cpu':
                for p in self.experts[key].parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device='cpu')
            else:
                weights = self.backing_store.get(key)
                for name,p in self.experts[key].named_parameters():
                    p.data = weights[name].to(device)
            return
        for name,p in self.experts[key].named_parameters():
            host=self.backing[key][name]
            p.data=host if device.type=='cpu' else host.to(device,non_blocking=host.is_pinned())

    @torch.no_grad()
    def _ensure(self,keys,demand):
        keys=list(dict.fromkeys(keys))
        if len(keys)>self.capacity:
            raise ValueError('Selected experts for this forward exceed cache capacity; shorten prefill or increase budget')
        if self.layer_capacity is not None and any(n > self.layer_capacity for n in Counter(k[0] for k in keys).values()):
            raise ValueError('Demand exceeds per-layer capacity; use batch-one incremental decoding and serial prefill')
        protected=set(keys)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        for key in keys:
            if key in self.hot:
                if demand: self.stats['hits']+=1
                self.hot.move_to_end(key)
                continue
            if demand: self.stats['misses']+=1
            else: self.stats['prefetch_loads']+=1
            while len(self.hot)>=self.capacity or (self.layer_capacity is not None and
                    sum(k[0] == key[0] for k in self.hot) >= self.layer_capacity):
                layer_full = self.layer_capacity is not None and sum(k[0] == key[0] for k in self.hot) >= self.layer_capacity
                candidates=[k for k in self.hot if k not in protected and (not layer_full or k[0] == key[0])]
                victim=next((k for k in candidates if k not in self.preferred),candidates[0])
                self._place(victim,torch.device('cpu')); del self.hot[victim]
                self.stats['evictions']+=1
            self._place(key,self.device); self.hot[key]=None
            self.peak_by_layer[key[0]] = max(self.peak_by_layer[key[0]], sum(k[0] == key[0] for k in self.hot))
            if self.device.type=='cuda':
                self.stats['host_to_device_bytes'] += (self.backing_store.entries[key]['weight_bytes'] if self.backing_store else
                    sum(t.numel()*t.element_size() for t in self.backing[key].values()))
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        self.stats['transfer_and_lookup_seconds']+=time.perf_counter()-started

    def set_context_labels(self,labels):
        labels=list(dict.fromkeys(labels))
        if not labels: raise ValueError('Context labels cannot be empty')
        if self.fixed_router:
            self.fixed_router.select(labels)
        # Validate all layers before changing context in restrictive mode.
        for layer,moe in self.moes.items():
            ids=self._label_ids(layer,labels)
            if self.policy=='restrict' and len(ids)<moe.gate.topk:
                raise ValueError(f'Layer {layer}: reviewed labels expose fewer than top-k experts')
        self.labels=labels
        # Global prefetch is bounded; later native selections always override it.
        by_layer={layer:self._label_ids(layer,labels) for layer in self.moes}
        # Interleave layers so a capacity limit does not privilege shallow layers.
        candidates=[(layer,ids[i]) for i in range(max(map(len,by_layer.values()),default=0))
                    for layer,ids in by_layer.items() if i<len(ids)]
        selected=self._bounded_candidates(candidates)
        self.preferred=set(selected)
        self._ensure(selected,demand=False)

    def prefetch_experts(self, ranked_experts):
        """Set a bounded hint list without changing native routing; [] gives LRU."""
        if self.policy != 'prefetch':
            raise ValueError('Explicit cache hints require exact prefetch routing')
        keys = list(dict.fromkeys(tuple(key) for key in ranked_experts))
        if any(key not in self.experts for key in keys):
            raise ValueError('Unknown expert in prefetch hints')
        selected = self._bounded_candidates(keys)
        self.preferred = set(selected)
        self._ensure(selected, demand=False)

    def _bounded_candidates(self, keys):
        selected = []
        counts = Counter()
        for key in keys:
            if len(selected) == self.capacity:
                break
            if self.layer_capacity is not None and counts[key[0]] >= self.layer_capacity:
                continue
            selected.append(key)
            counts[key[0]] += 1
        return selected

    def begin_turn(self):
        """Opt in to demand observations for learning AFTER this turn executes."""
        self.turn_demands = set()

    def _label_ids(self,layer,labels):
        return list(dict.fromkeys(eid for label in labels
                                 for eid in self.mapping.get(str(layer),{}).get(label,[])))

    @torch.no_grad()
    def _route(self,layer,gate,args,output):
        weights,indices=output
        if self.policy=='restrict':
            allowed=torch.zeros(gate.weight.shape[0],dtype=torch.bool,device=args[0].device)
            allowed[self._label_ids(layer,self.labels)]=True
            # Same grouping, sigmoid, correction bias and scaling as the native gate.
            logits=F.linear(args[0],gate.weight)
            weights,indices=Gate.route_logits(gate,logits,allowed)
            weights=weights.type_as(args[0])
        demanded = [(layer,int(e)) for e in indices.unique().tolist()]
        if self.turn_demands is not None:
            self.turn_demands.update(demanded)
        self._ensure(demanded,demand=True)
        return weights,indices

    def snapshot(self):
        total=self.stats['hits']+self.stats['misses']
        return {**dict(self.stats),'hit_rate':self.stats['hits']/total if total else 0,
                'policy':self.policy,'device':str(self.device),'resident_experts':[list(k) for k in self.hot],
                'max_experts_per_layer': self.layer_capacity, 'peak_resident_by_layer': dict(self.peak_by_layer),
                'disk_store': self.backing_store.snapshot() if self.backing_store else None,
                'fixed_gate_calls': dict(self.fixed_router.calls) if self.fixed_router else None,
                'counter_unit':'unique layer-expert demanded per forward',
                'ram_backing_bytes': self.backing_store.bytes if self.backing_store else sum(t.numel()*t.element_size() for values in self.backing.values() for t in values.values())}

    def inventory(self):
        expert_parameters=set()
        rows=[]
        for (layer,eid),expert in self.experts.items():
            params=list(expert.parameters())
            expert_parameters.update(id(p) for p in params)
            rows.append({'layer':layer,'expert':eid,'device':str(params[0].device),
                         'weight_bytes':sum(tensor_bytes(p) for p in params),
                         'logical_weight_bytes': self.backing_store.entries[(layer,eid)]['weight_bytes'] if self.backing_store else sum(tensor_bytes(p) for p in params),
                         'ram_backing_bytes': (self.backing_store.entries[(layer,eid)]['weight_bytes'] if (layer,eid) in self.backing_store.hot else 0)
                            if self.backing_store else sum(tensor_bytes(t) for t in self.backing[(layer,eid)].values())})
        common=Counter(); buffers=Counter()
        for p in self.model.parameters():
            if id(p) not in expert_parameters: common[str(p.device)]+=tensor_bytes(p)
        for b in self.model.buffers(): buffers[str(b.device)]+=tensor_bytes(b)
        result={'experts':rows,'backbone_and_shared_weight_bytes':dict(common),'buffer_bytes':dict(buffers),
                'disk_store': self.backing_store.snapshot() if self.backing_store else None,
                'expert_weight_bytes_by_device':dict(Counter({device:sum(r['weight_bytes'] for r in rows if r['device']==device)
                    for device in {r['device'] for r in rows}})),
                'note': 'Disk-backed cold parameters are empty placeholders. RAM store is reported separately; no full expert RAM copy.'
                    if self.backing_store else 'Cold execution tensors alias RAM backing; do not sum twice. Tensor inventory is not process RSS.'}
        if self.device.type=='cuda':
            result['cuda']={'allocated_bytes':torch.cuda.memory_allocated(self.device),
                            'reserved_bytes':torch.cuda.memory_reserved(self.device),
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device)}
        return result

    def close(self):
        """Remove hooks; cold parameters become RAM aliases or empty disk placeholders."""
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        for h in self.handles: h.remove()
        self.handles=[]
        if self.fixed_router:
            self.fixed_router.close()
        for key in list(self.hot): self._place(key,torch.device('cpu'))
        self.hot.clear()
