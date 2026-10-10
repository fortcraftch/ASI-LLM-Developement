import unittest
from asi.experiments.mask_intervention import build_masks, select_calibration


class MaskInterventionTests(unittest.TestCase):
    def test_budget_and_shared_core(self):
        stats = {c: {str(l): {'tokens': 100, 'counts': [(e + offset) % 64 + 1 for e in range(64)]}
                     for l in range(16)} for offset, c in enumerate(['a', 'b', 'c'])}
        old = {c: {str(l): {'experts': list(range(32)), 'weights': [1/32]*32}
                   for l in range(16)} for c in stats}
        masks = build_masks(stats, old)
        for name, categories in masks.items():
            for layers in categories.values():
                expected = 576 if name in ['first2_free', 'last2_free', 'uniform36'] else 512
                self.assertEqual(sum(len(r['experts']) for r in layers.values()), expected)
                self.assertTrue(all(len(set(r['experts'])) == len(r['experts']) for r in layers.values()))
        for layer in range(16):
            cores = [masks['shared8_category24'][c][str(layer)]['experts'][:8] for c in stats]
            self.assertTrue(all(core == cores[0] for core in cores))
        self.assertEqual(masks['first2_free']['a']['0']['experts'], list(range(64)))
        self.assertEqual(masks['first2_free_budget512']['a']['1']['experts'], list(range(64)))

    def test_no_test_leakage_and_rejected_examples(self):
        from unittest.mock import patch
        import torch
        class Tokenizer:
            eos_token_id = 0
            def encode(self, text, **kw): return [2]
        rows = [{'id': str(i), 'split': split, 'category': 'a', 'prompt': 'q', 'answer': 'a'}
                for i, split in enumerate(['train', 'train', 'train', 'test'])]
        with patch('asi.experiments.mask_intervention.input_ids', return_value=torch.tensor([[1]])):
            selected = select_calibration({'examples': rows}, {'2'}, {'1'}, ['a'], Tokenizer(), 2, 10)
        self.assertEqual([r['id'] for r in selected], ['2', '0'])
