import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from asi.experiments.cache_study import forward_probe, load_sessions, summarize
from asi.runtime.cache import NativeExpertSessionCache, move_backbone
from asi.runtime.routing import ExpertUsagePredictor
from test_expert_runtime import tiny


class PredictorTests(unittest.TestCase):
    def test_transition_changes_ranking_without_observing_future(self):
        predictor = ExpertUsagePredictor([(1, 0), (1, 1)], smoothing=1)
        for _ in range(4):
            predictor.observe(['code'], ['neutral'], [(1, 0)])
            predictor.observe(['science'], ['neutral'], [(1, 1)])
        frozen = copy.deepcopy(predictor.to_dict())
        self.assertEqual(predictor.rank(['neutral'], ['code'])[0], (1, 0))
        self.assertEqual(predictor.rank(['neutral'], ['science'])[0], (1, 1))
        self.assertEqual(predictor.to_dict(), frozen)
        restored = ExpertUsagePredictor.from_dict(json.loads(json.dumps(frozen)))
        self.assertEqual(restored.rank(['neutral'], ['science']), predictor.rank(['neutral'], ['science']))

    def test_turn_presence_not_token_count_and_unknown_labels_backoff(self):
        predictor = ExpertUsagePredictor([(1, 0), (1, 1)])
        predictor.observe([], ['code', 'code'], [(1, 0)] * 100)
        self.assertEqual(predictor.tables['global'][1][(1, 0)], 1)
        self.assertEqual(predictor.tables['current:code'][0], 1)
        self.assertEqual(predictor.rank(['unseen']), predictor.rank(mode='popularity'))
        with self.assertRaises(ValueError):
            predictor.observe([], ['code'], [(99, 0)])
        invalid = predictor.to_dict()
        invalid['tables']['global']['usage'][0][2] = 100
        with self.assertRaises(ValueError):
            ExpertUsagePredictor.from_dict(invalid)

    def test_split_validation_rejects_leakage(self):
        rows = [{'id': 'a', 'split': 'train', 'prompts': ['Same input', 'Train only']},
                {'id': 'b', 'split': 'test', 'prompts': [' SAME   input ', 'Test only']}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sessions.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                load_sessions(path)
            rows[1]['prompts'][0] = 'Independent input'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(len(load_sessions(path)), 2)
            rows[1]['id'] = 'a'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                load_sessions(path)

    def test_summary_weights_nll_by_tokens_and_counts_transfers(self):
        base = {'policy': 'lru', 'pipeline_seconds': .1, 'tokens': 1, 'nll': 1.,
                'max_probe_abs_difference': 0., 'routes_equal': True, 'argmax_equal': True,
                'cache_delta': {'hits': 1, 'misses': 1, 'host_to_device_bytes': 20, 'prefetch_loads': 1}}
        other = dict(base, tokens=3, nll=3.)
        result = summarize([base, other])['lru']
        self.assertEqual(result['nll_token_weighted'], 2.5)
        self.assertEqual(result['hit_rate'], .5)
        self.assertEqual(result['cache_totals']['host_to_device_bytes'], 40)


class CacheHintTests(unittest.TestCase):
    def check_hints(self, device):
        model = tiny().eval()
        model.set_all_experts_active()
        model.to(device)
        tokens = [1, 2, 3, 4]
        reference = forward_probe(model, tokens, device)
        model.to('cpu')
        move_backbone(model, device)
        mapping = {'layers': {}}
        cache = NativeExpertSessionCache(model, mapping, device=device, max_hot_experts=4)
        try:
            before = cache.snapshot()
            with self.assertRaises(ValueError):
                cache.prefetch_experts([(999, 1)])
            self.assertEqual(cache.snapshot(), before)
            cache.prefetch_experts(list(cache.experts)[:2])
            cache.begin_turn()
            result = forward_probe(model, tokens, device)
            self.assertEqual(result['route_sha256'], reference['route_sha256'])
            torch.testing.assert_close(result['probe'], reference['probe'])
            self.assertEqual(cache.turn_demands, set(result['demands']))
            self.assertLessEqual(len(cache.hot), 4)
            cache.prefetch_experts([])
            self.assertFalse(cache.preferred)
        finally:
            cache.close()

    def test_hints_keep_native_outputs(self):
        self.check_hints('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_hints_keep_native_outputs(self):
        self.check_hints('cuda')
