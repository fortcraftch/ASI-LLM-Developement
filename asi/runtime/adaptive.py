"""Optional context-level mixtures and RAM / INT8 warm / dense hot residency."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from types import MethodType

import torch

from asi.runtime.cache import ExpertCacheManager, tensor_bytes, module_memory
from asi.models.original import native_moes


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pack(weight):
    value = weight.detach().float().cpu()
    scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return ((value / scale).round().clamp(-127, 127).to(torch.int8), scale)


class WarmExpertCache(ExpertCacheManager):
    """W8 storage, floating point execution. No transfers during model.forward.

    Every hot expert uses the same dequantized weights, including cold misses.
    This prevents cache history from changing the quantization policy.
    """
    def __init__(self, device, max_hot_pools=1, warm_bytes=0):
        super().__init__(device, max_hot_pools)
        if warm_bytes < 0:
            raise ValueError('Negative warm budget')
        self.warm_budget = warm_bytes
        self.packed = {}
        self.warm = {}

    def initialize(self, model):
        self.warm.clear()
        self.packed.clear()
        super().initialize(model)
        self.packed = {key: {name: pack(value) for name, value in host.items()}
                       for key, host in self.backing.items()}

    @staticmethod
    def packed_bytes(projections):
        return sum(tensor_bytes(t) for pair in projections.values() for t in pair)

    def _upload(self, key):
        if self.device.type == 'cuda':
            self.stats['host_to_device_bytes'] += self.packed_bytes(self.packed[key])
        return {name: tuple(t.to(self.device) for t in pair)
                for name, pair in self.packed[key].items()}

    def _place(self, key, device):
        if device.type == 'cpu' and (self.device.type != 'cpu' or not getattr(self, '_promoting', False)):
            return super()._place(key, device)
        data = self.warm.pop(key, None)
        if data is None:
            data = self._upload(key)
            self.stats['cold_promotions'] += 1
        else:
            self.stats['warm_promotions'] += 1
        _, expert = self.entries[key]
        for name, param in expert.named_parameters():
            q, scale = data[name]
            param.data = (q.float() * scale).to(self.backing[key][name].dtype)

    @torch.no_grad()
    def prepare(self, model, required_pool_ids, ranked_pool_ids=None):
        # Implement transitions explicitly: superclass counts full precision H2D.
        self._check(model)
        required = list(dict.fromkeys(required_pool_ids))
        ranking = list(dict.fromkeys(required + list(ranked_pool_ids or [])))
        if not required or len(required) > self.max_hot_pools or any(type(p) is not int or not 0 <= p < model.config.n_pools for p in ranking):
            raise ValueError('Requested pools do not fit or contain invalid IDs')
        hot = set(required)
        for key, (pool, _) in self.entries.items():
            if pool in self.resident_pools and pool not in hot:
                super()._place(key, torch.device('cpu'))
                self.stats['evictions'] += 1
        self._promoting = True
        try:
            for key, (pool, _) in self.entries.items():
                if pool in hot:
                    self.stats['requests'] += 1
                    self.stats['hits' if pool in self.resident_pools else 'misses'] += 1
                    if pool not in self.resident_pools:
                        self._place(key, self.device)
                        self.stats['loads'] += 1
        finally:
            self._promoting = False
        self.resident_pools = hot
        # Only complete pools enter warm storage; remove unwanted entries first.
        desired, remaining = set(), self.warm_budget
        for pool in ranking:
            if pool in hot:
                continue
            keys = [key for key, (p, _) in self.entries.items() if p == pool]
            size = sum(self.packed_bytes(self.packed[key]) for key in keys)
            if size <= remaining:
                desired.update(keys)
                remaining -= size
        self.warm = {key: value for key, value in self.warm.items() if key in desired}
        for key in sorted(desired - self.warm.keys()):
            self.warm[key] = self._upload(key)
            self.stats['warm_loads'] += 1
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return required

    def inventory(self):
        report = super().inventory()
        report.update(warm_compressed_bytes=sum(self.packed_bytes(v) for v in self.warm.values()),
                      packed_ram_bytes=sum(self.packed_bytes(v) for v in self.packed.values()),
                      warm_device=str(self.device), warm_budget=self.warm_budget,
                      quantization='symmetric INT8 per output row; dequantize at context change')
        return report

    def offload_all(self, model):
        self.warm.clear()
        super().offload_all(model)


class ContextMixture:
    """Inference-only union: all experts in all selected pools, constant weights."""
    def __init__(self, model, pool_ids):
        if model.training:
            raise ValueError('Context mixtures are inference-only')
        self.model = model
        self.pool_ids = list(dict.fromkeys(pool_ids))
        if not self.pool_ids or any(type(p) is not int or not 0 <= p < model.config.n_pools for p in self.pool_ids):
            raise ValueError('Select known pools')
        self.originals = []

    def __enter__(self):
        self.old_mask = self.model.active_expert_mask.clone()
        self.model.set_active_pools(self.pool_ids)
        for moe in native_moes(self.model).values():
            gate = moe.gate
            self.originals.append((gate, gate.__dict__.get('forward'), 'forward' in gate.__dict__))
            def fixed(gate, x, allowed_expert_mask=None):
                ids = allowed_expert_mask.nonzero(as_tuple=False).flatten()
                indices = ids.expand(x.shape[0], -1)
                weights = torch.full(indices.shape, gate.route_scale / len(ids), device=x.device, dtype=x.dtype)
                if gate.routing_observer is not None:
                    gate.routing_observer(gate, x, None, weights, indices, allowed_expert_mask)
                return weights, indices
            gate.forward = MethodType(fixed, gate)
        return self

    def __exit__(self, *exc):
        for gate, previous, existed in self.originals:
            if existed:
                gate.forward = previous
            else:
                del gate.forward
        self.model.active_expert_mask = self.old_mask


def memory_layout(model):
    pools = Counter()
    compressed = Counter()
    expert_params = set()
    largest_projection = 0
    for moe in native_moes(model).values():
        for index, expert in enumerate(moe.experts):
            pool = int(moe.expert_pool_ids[index])
            for param in expert.parameters():
                expert_params.add(id(param))
                pools[pool] += tensor_bytes(param)
                compressed[pool] += param.numel() + param.shape[0] * 4
                largest_projection = max(largest_projection, param.numel() * 4)
    base = sum(tensor_bytes(p) for p in model.parameters() if id(p) not in expert_params)
    buffers = sum(tensor_bytes(b) for b in model.buffers())
    return {'backbone_bytes': base, 'buffer_bytes': buffers, 'pool_bytes': dict(pools),
            'packed_pool_bytes': dict(compressed), 'largest_projection_fp32_bytes': largest_projection}


def choose_plan(layout, pools, available_bytes, reserve_bytes, warm_bytes=0,
                max_nll_delta=None, evidence=None, identity=None):
    """Never silently discard a requested label; unknown quality fails a quality constraint."""
    pools = sorted(set(pools))
    if not pools or any(p not in layout['pool_bytes'] for p in pools):
        raise ValueError('Unknown or empty pools')
    if min(available_bytes, reserve_bytes, warm_bytes) < 0 or (max_nll_delta is not None and (not math.isfinite(max_nll_delta) or max_nll_delta < 0)):
        raise ValueError('Invalid memory or quality budget')
    hot = sum(layout['pool_bytes'][p] for p in pools)
    # Reserve explicitly covers activations, kernel workspaces and allocator slack.
    # Extra hot-size covers temporary expert stacks used by the existing MoE.
    fixed = layout['backbone_bytes'] + layout['buffer_bytes'] + 2 * hot + reserve_bytes
    rows = []
    for mode in ['dense', 'int8']:
        temporary = max(layout['packed_pool_bytes'].values()) + 2 * layout['largest_projection_fp32_bytes'] if mode == 'int8' else 0
        warm = min(warm_bytes, max(0, available_bytes - fixed - temporary)) if mode == 'int8' else 0
        estimated = fixed + temporary + warm
        quality = None
        if evidence and evidence.get('identity') == identity:
            for profile in evidence.get('profiles', []):
                if profile.get('mode') == mode and profile.get('pools') == pools:
                    value = profile.get('nll_delta')
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        quality = value
        rows.append({'mode': mode, 'pools': pools, 'warm_bytes': warm,
                     'estimated_peak_bytes': estimated, 'available_bytes': available_bytes,
                     'reserve_bytes': reserve_bytes, 'observed_nll_delta': quality,
                     'fits_estimate': estimated <= available_bytes,
                     'quality_eligible': max_nll_delta is None or (quality is not None and quality <= max_nll_delta)})
    valid = [r for r in rows if r['fits_estimate'] and r['quality_eligible']]
    if not valid:
        return {'status': 'blocked', 'reason': 'Memory estimate or measured quality constraint cannot be satisfied without dropping labels', 'candidates': rows}
    # Prefer requested warm storage; otherwise avoid introducing quantization.
    valid.sort(key=lambda r: (r['mode'] != ('int8' if warm_bytes else 'dense'), r['estimated_peak_bytes']))
    return {'status': 'ready', **valid[0], 'candidates': rows,
            'quality_status': 'measured_on_calibration_windows' if valid[0]['observed_nll_delta'] is not None else 'unmeasured',
            'note': 'Memory estimate, not an OOM guarantee. Quality is empirical, not a bound for each prompt.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool-manifest', type=Path, required=True)
    parser.add_argument('--pools', help='Comma-separated pool names; otherwise use integrated classifier')
    parser.add_argument('--classifier', type=Path)
    parser.add_argument('--taxonomy', type=Path, help='Shared semantic taxonomy for classifier labels')
    parser.add_argument('--model-profile', type=Path, help='Checkpoint-bound category-to-pool contract')
    parser.add_argument('--classifier-threshold', type=float, default=.5)
    parser.add_argument('--category-depth', type=int)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-mib', type=float, help='Cap available free CUDA memory; required on CPU')
    parser.add_argument('--reserve-mib', type=float, default=1024)
    parser.add_argument('--warm-mib', type=float, default=0)
    parser.add_argument('--quality-report', type=Path)
    parser.add_argument('--max-nll-delta', type=float)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Otherwise plan only')
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        parser.error('Use a new output file and positive max-new-tokens')
    if bool(args.taxonomy) != bool(args.model_profile) or (args.taxonomy and (not args.classifier or args.pools)):
        parser.error('Semantic routing requires taxonomy, model-profile and classifier, without explicit pools')
    if not 0 <= args.classifier_threshold <= 1:
        parser.error('Classifier threshold must be in [0,1]')
    from asi.runtime.generation import load_model, validate_pool_identity, generate
    import tiktoken
    model, metadata = load_model(args.checkpoint)
    manifest = json.loads(args.pool_manifest.read_text(encoding='utf-8'))
    names = list(manifest['pools'])
    validate_pool_identity(model, metadata, names)
    enc = tiktoken.get_encoding('gpt2')
    classifier_memory = None
    classifier_seconds = None
    semantic_routing = None
    if args.pools:
        labels = args.pools.split(',')
    elif args.classifier:
        from asi.runtime.classifier import load_head, predict
        if args.taxonomy:
            from asi.taxonomy import Taxonomy, resolve, digest, validate_profile
            from asi.runtime.classifier import features
            taxonomy = Taxonomy.read(args.taxonomy)
            profile = json.loads(args.model_profile.read_text(encoding='utf-8-sig'))
            validate_profile(taxonomy, profile)
            if profile['pool_order'] != names or profile['pool_manifest_sha256'] != digest(manifest) or profile.get('checkpoint_sha256') != fingerprint(args.checkpoint):
                parser.error('Model profile must be bound to this checkpoint and pool manifest')
            head = load_head(args.classifier, model, taxonomy.labels, fingerprint(args.pool_manifest), taxonomy.identity)
            started = time.perf_counter()
            with torch.no_grad():
                values = head(features(model.embed, [enc.encode(args.prompt)])).sigmoid()[0].tolist()
            scores = {name: value for name, value in zip(taxonomy.labels, values) if name in head.trained_labels}
            semantic_routing = resolve(taxonomy, profile, scores, args.classifier_threshold, args.category_depth)
            labels = semantic_routing['pools']
        else:
            head = load_head(args.classifier, model, names, fingerprint(args.pool_manifest))
            started = time.perf_counter()
            labels = predict(model.embed, head, enc.encode(args.prompt), names, args.classifier_threshold)
        classifier_seconds = time.perf_counter() - started
        classifier_memory = module_memory(head)
    else:
        parser.error('Provide --pools or --classifier')
    if any(name not in names for name in labels):
        parser.error('Unknown pool name')
    ids = [names.index(name) for name in labels]
    device = torch.device(args.device)
    if device.type == 'cuda':
        available = torch.cuda.mem_get_info(device)[0]
        if args.memory_mib is not None:
            available = min(available, int(args.memory_mib * 1024**2))
    elif args.memory_mib is not None:
        available = int(args.memory_mib * 1024**2)
    else:
        parser.error('CPU requires an explicit logical memory budget')
    identity = {'checkpoint_sha256': fingerprint(args.checkpoint), 'pool_manifest_sha256': fingerprint(args.pool_manifest),
                'policy': 'uniform_union_v1'}
    evidence = json.loads(args.quality_report.read_text()) if args.quality_report else None
    layout = memory_layout(model)
    if semantic_routing and semantic_routing['status'] != 'ready':
        plan = {'status': 'blocked', 'reason': 'Semantic classification is uncertain, partially covered or needs refinement; no topics silently discarded'}
    else:
        plan = choose_plan(layout, ids, available, int(args.reserve_mib * 1024**2), int(args.warm_mib * 1024**2),
                           args.max_nll_delta, evidence, identity)
    result = {'identity': identity, 'labels': labels, 'plan': plan, 'layout': layout,
              'semantic_routing': semantic_routing,
              'classifier_memory': classifier_memory, 'classifier_seconds': classifier_seconds, 'executed': False}
    if args.execute and plan['status'] == 'ready':
        cache = (WarmExpertCache(args.device, len(ids), plan['warm_bytes']) if plan['mode'] == 'int8'
                 else ExpertCacheManager(args.device, len(ids)))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        cache.initialize(model)
        try:
            started = time.perf_counter()
            cache.prepare(model, ids, list(range(len(names))))
            result['prepare_seconds'] = time.perf_counter() - started
            before = cache.snapshot()
            started = time.perf_counter()
            with ContextMixture(model, ids):
                result['text'] = generate(model, enc, args.prompt, args.device, args.max_new_tokens, 0., 1)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            result['generation_seconds'] = time.perf_counter() - started
            if any(cache.snapshot().get(key, 0) != before.get(key, 0) for key in ['loads', 'host_to_device_bytes', 'evictions']):
                raise RuntimeError('Expert residency changed during generation')
            result.update(executed=True, before_generation=before, after_generation=cache.snapshot(), memory=cache.inventory())
        finally:
            cache.offload_all(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
