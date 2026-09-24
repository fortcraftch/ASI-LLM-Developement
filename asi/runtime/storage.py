"""Uncompressed expert shards on disk with a bounded application RAM cache."""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import re
import time

import torch

from asi.models.original import DEFAULT_ARCHITECTURE, file_sha256


class ExpertDiskStore:
    def __init__(self, directory, max_ram_bytes):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 1:
            raise ValueError('Unsupported expert store schema')
        self.entries = {tuple(entry['key']): entry for entry in self.manifest['experts']}
        if not self.entries or len(self.entries) != len(self.manifest['experts']):
            raise ValueError('Empty or duplicate expert manifest')
        if max_ram_bytes < max(e['weight_bytes'] for e in self.entries.values()):
            raise ValueError('RAM budget must fit at least the largest single expert')
        self.budget = max_ram_bytes
        self.hot = OrderedDict()
        self.bytes = 0
        self.stats = Counter()
        self.verified = set()
        for entry in self.entries.values():
            self.path(entry['file'])

    def path(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Shard path escapes store')
        return path

    def get(self, key):
        if key in self.hot:
            self.stats['ram_hits'] += 1
            self.hot.move_to_end(key)
            return self.hot[key]
        entry = self.entries[key]
        while self.bytes + entry['weight_bytes'] > self.budget:
            victim = next(iter(self.hot))
            del self.hot[victim]
            self.bytes -= self.entries[victim]['weight_bytes']
            self.stats['ram_evictions'] += 1
        path = self.path(entry['file'])
        start = time.perf_counter()
        if key not in self.verified:
            if file_sha256(path) != entry['sha256']:
                raise ValueError('Corrupt expert shard: ' + str(path))
            self.stats['verification_read_bytes'] += path.stat().st_size
            self.verified.add(key)
        weights = torch.load(path, map_location='cpu', weights_only=True)
        if set(weights) != set(entry['parameters']):
            raise ValueError('Expert parameter names do not match manifest')
        for name, value in weights.items():
            spec = entry['parameters'][name]
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
                raise ValueError('Expert shape/dtype mismatch: ' + name)
        actual_bytes = sum(t.numel()*t.element_size() for t in weights.values())
        if actual_bytes != entry['weight_bytes']:
            raise ValueError('Expert size mismatch')
        self.hot[key] = weights
        self.bytes += actual_bytes
        self.stats['ram_misses'] += 1
        self.stats['shard_reads'] += 1
        self.stats['file_read_bytes'] += path.stat().st_size
        self.stats['read_seconds'] += time.perf_counter()-start
        self.stats['peak_ram_weight_bytes'] = max(self.stats['peak_ram_weight_bytes'], self.bytes)
        return weights

    def snapshot(self):
        return {**dict(self.stats), 'ram_weight_bytes': self.bytes, 'ram_budget_bytes': self.budget,
                'resident_ram_experts': [list(key) for key in self.hot],
                'disk_file_bytes': sum(e['disk_bytes'] for e in self.entries.values()),
                'note': 'RAM budget covers cached expert tensors, not process RSS, deserialization overhead or OS page cache. '
                        'Read bytes are logical file reads, not physical SSD I/O. Shards are uncompressed.'}


def export_store(checkpoint, architecture, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new store directory')
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    groups, backbone = {}, {}
    for name, tensor in source['model'].items():
        match = re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)', name)
        if match:
            layer, expert, parameter = match.groups()
            groups.setdefault((int(layer), int(expert)), {})[parameter] = tensor
        else:
            backbone[name] = tensor
    if not groups:
        raise ValueError('No supported expert weights in checkpoint')
    output.mkdir(parents=True)
    torch.save(backbone, output/'backbone.pt')
    entries = []
    for key, weights in sorted(groups.items()):
        path = output/f'expert_{key[0]:03d}_{key[1]:04d}.pt'
        # Detach from potentially larger shared checkpoint storages when exporting views.
        payload = {name: value.detach().clone() for name, value in weights.items()}
        torch.save(payload, path)
        del payload
        entries.append({'key': list(key), 'file': path.name, 'sha256': file_sha256(path),
                        'disk_bytes': path.stat().st_size,
                        'weight_bytes': sum(t.numel()*t.element_size() for t in weights.values()),
                        'parameters': {n: {'shape': list(t.shape), 'dtype': str(t.dtype)} for n,t in weights.items()}})
    manifest = {'schema': 1, 'experts': entries, 'backbone_sha256': file_sha256(output/'backbone.pt'),
                'metadata': {'config': source['config'], 'step': source.get('step'), 'val_loss': source.get('val_loss'),
                             'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': file_sha256(checkpoint),
                             'architecture': str(Path(architecture).resolve()), 'architecture_sha256': file_sha256(architecture)}}
    # Manifest is written last: interrupted exports cannot be loaded as complete stores.
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture', type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_store(args.checkpoint, args.architecture, args.output)
    print(f'Exported {len(result["experts"])} experts and backbone to {args.output}')
