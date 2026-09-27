import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from asi.analysis.experts import RoutingTrace
from asi.runtime.adaptive import ContextMixture, WarmExpertCache, choose_plan, memory_layout, pack
from asi.runtime.classifier import IntegratedHead, features, load_head, read_examples
from test_expert_runtime import tiny


class AdaptiveTests(unittest.TestCase):
    def test_union_executes_all_selected_without_E_and_restores(self):
        model = tiny().eval()
        model.config.routing_mode = 'uniform_pool'
        for layer in model.layers[1:]:
            layer.ffn.gate.routing_mode = 'uniform_pool'
        model.set_active_pools([0])
        old = model.active_expert_mask.clone()
        with patch('asi.models.domain.Gate.route_logits', side_effect=AssertionError('E called')):
            with ContextMixture(model, [0, 1]), RoutingTrace(model, ['a', 'b']) as trace, torch.inference_mode():
                model(torch.tensor([[1, 2, 3]]))
            for row in trace.rows:
                self.assertEqual(row['experts'], [0, 1, 2, 3])
                self.assertAlmostEqual(sum(row['weights']), 1.)
        self.assertTrue(torch.equal(model.active_expert_mask, old))
        self.assertNotIn('forward', model.layers[1].ffn.gate.__dict__)

    def test_int8_error_bound_and_warm_history_independence(self):
        torch.manual_seed(3)
        value = torch.randn(5, 13)
        q, scale = pack(value)
        self.assertTrue(((value - q.float() * scale).abs() <= scale / 2 + 1e-6).all())
        model = tiny().eval()
        cache = WarmExpertCache('cpu', 1, 100000)
        cache.initialize(model)
        cache.prepare(model, [0], [0, 1])
        expected = model.layers[1].ffn.experts[0].w1.weight.clone()
        self.assertGreater(cache.inventory()['warm_compressed_bytes'], 0)
        self.assertLessEqual(cache.inventory()['warm_compressed_bytes'], cache.warm_budget)
        cache.prepare(model, [1])
        self.assertEqual(cache.stats['warm_promotions'], 4)
        cache.prepare(model, [0])
        self.assertTrue(torch.equal(expected, model.layers[1].ffn.experts[0].w1.weight))
        cache.offload_all(model)
        self.assertEqual(cache.inventory()['warm_compressed_bytes'], 0)

    def test_planner_requires_evidence_and_never_drops_labels(self):
        layout = memory_layout(tiny())
        blocked = choose_plan(layout, [0, 1], 1, 0)
        self.assertEqual(blocked['status'], 'blocked')
        blocked = choose_plan(layout, [0, 1], 10**8, 0, max_nll_delta=.1)
        self.assertEqual(blocked['status'], 'blocked')
        evidence = {'identity': {'id': 1}, 'profiles': [{'mode': 'dense', 'pools': [0, 1], 'nll_delta': .05}]}
        plan = choose_plan(layout, [0, 1], 10**8, 0, max_nll_delta=.1, evidence=evidence, identity={'id': 1})
        self.assertEqual(plan['pools'], [0, 1])
        self.assertEqual(plan['mode'], 'dense')
        wrong = choose_plan(layout, [0, 1], 10**8, 0, max_nll_delta=.1, evidence=evidence, identity={'id': 2})
        self.assertEqual(wrong['status'], 'blocked')

    def test_head_gradients_do_not_touch_embedding(self):
        model = tiny()
        before = model.embed.weight.detach().clone()
        head = IntegratedHead(16, 2, 8)
        x = features(model.embed, [[1, 2, 3], [4, 5]])
        torch.nn.functional.binary_cross_entropy_with_logits(head(x), torch.tensor([[1., 1.], [0., 1.]])).backward()
        self.assertIsNone(model.embed.weight.grad)
        self.assertTrue(torch.equal(before, model.embed.weight))
        self.assertIsNotNone(head.network[-1].weight.grad)

    def test_training_head_and_calibrating_policies_end_to_end(self):
        from asi.runtime.classifier import main as train_head
        from asi.experiments.adaptive import main as evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = tiny().eval()
            model.config.routing_mode = 'uniform_pool'
            checkpoint = root / 'model.pt'
            torch.save({'model': model.state_dict(), 'config': asdict(model.config), 'pool_names': ['a', 'b']}, checkpoint)
            manifest = root / 'pools.json'
            manifest.write_text(json.dumps({'pools': {name: {'categories': [name]} for name in ['a', 'b']}}))
            examples = root / 'examples.jsonl'
            examples.write_text('\n'.join(json.dumps(row) for row in [
                {'tokens': [1, 2], 'labels': ['a'], 'split': 'train'},
                {'tokens': [3, 4], 'labels': ['a', 'b'], 'split': 'train'},
                {'tokens': [2, 3], 'labels': ['b'], 'split': 'val'}]))
            args = ['head', '--checkpoint', str(checkpoint), '--pool-manifest', str(manifest),
                    '--examples', str(examples), '--output', str(root / 'head'), '--steps', '2', '--batch-size', '2']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                train_head()
            head = load_head(root / 'head/head.pt', model, ['a', 'b'])
            self.assertFalse(head.training)
            with self.assertRaises(ValueError):
                load_head(root / 'head/head.pt', model, ['b', 'a'])
            for name in ['a', 'b']:
                (root / name).mkdir()
                np.save(root / name / 'val_0.npy', np.arange(60, dtype=np.uint16))
            args = ['evaluate', '--checkpoint', str(checkpoint), '--pool-manifest', str(manifest),
                    '--data-root', str(root), '--contexts', 'a;a,b', '--seq-len', '4', '--windows-per-pool', '2',
                    '--device', 'cpu', '--output', str(root / 'quality.json')]
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                evaluate()
            report = json.loads((root / 'quality.json').read_text())
            self.assertEqual(len(report['profiles']), 4)
            self.assertAlmostEqual(report['profiles'][0]['nll_delta'], 0., places=6)
            for profile in report['profiles']:
                self.assertEqual(profile['before']['misses'], profile['after']['misses'])
            examples.write_text(examples.read_text() + '\n' + json.dumps({'tokens': [1, 2], 'labels': ['a'], 'split': 'val'}))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_examples(examples, ['a', 'b'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA needed to verify physical residency')
    def test_cuda_warm_promotion_and_no_transfers_during_decode(self):
        model = tiny().eval()
        cache = WarmExpertCache('cuda', 1, 100000)
        cache.initialize(model)
        try:
            cache.prepare(model, [0], [0, 1])
            self.assertTrue(all(q.device.type == 'cuda' for entry in cache.warm.values() for q, scale in entry.values()))
            before = cache.stats['host_to_device_bytes']
            cache.prepare(model, [1])
            self.assertEqual(cache.stats['host_to_device_bytes'], before)
            stats = dict(cache.stats)
            with ContextMixture(model, [1]), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                model(torch.tensor([[1, 2]], device='cuda'))
                model(torch.tensor([[3]], device='cuda'), start_pos=2)
            self.assertEqual(dict(cache.stats), stats)
            for pool, expert in cache.entries.values():
                self.assertEqual(next(expert.parameters()).device.type, 'cuda' if pool == 1 else 'cpu')
        finally:
            cache.offload_all(model)


if __name__ == '__main__':
    unittest.main()
