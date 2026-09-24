from collections import Counter
from dataclasses import asdict
import json
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from asi.experiments.decode_study import check_incremental, summarize_decode
from asi.models.original import IncrementalDecoder, load_streamed_model
from asi.runtime.cache import NativeExpertSessionCache, move_backbone
from asi.runtime.storage import ExpertDiskStore, export_store
from test_expert_runtime import tiny


class IncrementalTests(unittest.TestCase):
    def test_fp32_matches_full_forward_and_reset_overwrites_kv(self):
        model = tiny().eval()
        check_incremental(model, [1, 2, 3, 4, 5], 'cpu')
        decoder = IncrementalDecoder(model)
        with torch.inference_mode():
            first = decoder.step(2).clone()
            decoder.step(3)
            decoder.reset()
            torch.testing.assert_close(first, decoder.step(2))
            for _ in range(model.config.block_size-1):
                decoder.step(1)
        with self.assertRaises(ValueError):
            decoder.step(1)

    def test_layer_limit_eviction_and_invalid_union_is_atomic(self):
        model = tiny().eval()
        cache = NativeExpertSessionCache(model, {'layers': {}}, device='cpu', max_hot_experts=4, max_experts_per_layer=2)
        try:
            cache.prefetch_experts([(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)])
            self.assertEqual(set(cache.hot), {(1,0),(1,1),(2,0),(2,1)})
            before = cache.snapshot()
            with self.assertRaises(ValueError):
                cache._ensure([(1,0),(1,1),(1,2)], demand=True)
            self.assertEqual(cache.snapshot(), before)
            cache._ensure([(1,2),(1,3)], demand=True)
            self.assertEqual(set(cache.hot), {(1,2),(1,3),(2,0),(2,1)})
            self.assertEqual(dict(cache.peak_by_layer), {1: 2, 2: 2})
        finally:
            cache.close()

    def test_decode_metric_counts_no_load_steps_not_just_expert_hits(self):
        row = {'policy':'lru','tokens':[2,3,4],'output_equal':True,'routes_equal':True,
               'max_probe_difference':0.,'prefetch_delta':{'host_to_device_bytes':50},
               'events':[{'phase':'decode','seconds':.1,'cache_delta':{'hits':21,'misses':1,'host_to_device_bytes':10}},
                         {'phase':'decode','seconds':.1,'cache_delta':{'hits':22,'misses':0,'host_to_device_bytes':0}}]}
        result = summarize_decode([row])['lru']['decode']
        self.assertEqual(result['fraction_without_loads'], .5)
        self.assertEqual(result['h2d_bytes_per_step'], 5.)
        self.assertGreater(result['hit_rate'], .95)


class StorageTests(unittest.TestCase):
    def make_store(self, root):
        source = root/'architecture.py'
        source.write_text('from asi.models.domain import GPT as DeepSeekV3, GPTConfig as DeepSeekV3Config\n')
        model = tiny().eval()
        checkpoint = root/'checkpoint.pt'
        torch.save({'config':asdict(model.config),'model':model.state_dict()}, checkpoint)
        directory = root/'store'
        manifest = export_store(checkpoint, source, directory)
        return model, source, directory, manifest

    def test_store_budget_eviction_checksums_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, directory, manifest = self.make_store(Path(tmp))
            size = manifest['experts'][0]['weight_bytes']
            store = ExpertDiskStore(directory, size)
            keys = list(store.entries)
            store.get(keys[0]); store.get(keys[0]); store.get(keys[1])
            self.assertEqual(store.stats['ram_hits'], 1)
            self.assertEqual(list(store.hot), [keys[1]])
            self.assertLessEqual(store.bytes, size)
            self.assertLessEqual(store.stats['peak_ram_weight_bytes'], size)
            with self.assertRaises(ValueError):
                store.path('../outside.pt')
            with self.assertRaises(ValueError):
                ExpertDiskStore(directory, size-1)
            shard = directory/store.entries[keys[2]]['file']
            shard.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                store.get(keys[2])

    def test_meta_loader_never_loads_expert_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, source, directory, _ = self.make_store(Path(tmp))
            load = torch.load
            with patch('torch.load', wraps=load) as observed:
                model, _ = load_streamed_model(directory, source)
            self.assertEqual(len(observed.call_args_list), 1)
            self.assertEqual(Path(observed.call_args_list[0].args[0]).name, 'backbone.pt')
            self.assertTrue(all(p.numel() == 0 for name,p in model.named_parameters() if '.experts.' in name))
            self.assertFalse(any(p.is_meta for p in model.parameters()))
            # Windows cannot unlink an mmap-backed backbone while its model lives.
            del model
            gc.collect()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for disk-backed execution')
    def test_streamed_gpu_outputs_and_physical_residency(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, source, directory, manifest = self.make_store(Path(tmp))
            reference.cuda()
            decoder = IncrementalDecoder(reference)
            expected = [decoder.step(t).cpu() for t in [1,2,3,4]]
            reference.cpu()
            model, _ = load_streamed_model(directory, source)
            size = manifest['experts'][0]['weight_bytes']
            store = ExpertDiskStore(directory, 2*size)
            move_backbone(model, 'cuda')
            cache = NativeExpertSessionCache(model, {'layers':{}}, device='cuda', max_hot_experts=4,
                                            max_experts_per_layer=2, backing_store=store)
            decoder = IncrementalDecoder(model)
            try:
                for token, expected_logits in zip([1,2,3,4], expected):
                    torch.testing.assert_close(decoder.step(token).cpu(), expected_logits)
                    physical = Counter(layer for (layer,eid),expert in cache.experts.items()
                                       if next(expert.parameters()).device.type == 'cuda')
                    self.assertTrue(all(count <= 2 for count in physical.values()))
                    for key, expert in cache.experts.items():
                        if key not in cache.hot:
                            self.assertTrue(all(p.numel() == 0 for p in expert.parameters()))
                self.assertLessEqual(store.bytes, 2*size)
                self.assertGreater(store.stats['shard_reads'], 0)
            finally:
                cache.close()
