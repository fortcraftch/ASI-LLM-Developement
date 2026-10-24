import unittest
from asi.experiments.mask_intervention import build_masks, select_calibration, random_controls, reuse_calibration, global_control


class GlobalControlTests(unittest.TestCase):
    def test_equal_category_weight_not_raw_token_weight(self):
        stats = {c: {str(l): {'tokens': t, 'counts': [0]*64} for l in range(16)}
                 for c,t in [('a', 100), ('b', 1)]}
        for l in map(str, range(16)):
            stats['a'][l]['counts'][63] = 10
            stats['b'][l]['counts'][62] = 1
        masks = global_control(stats)['global32']
        self.assertEqual(masks['a'], masks['b'])
        self.assertEqual(masks['a']['0']['experts'], [62,63]+list(range(30)))
        self.assertEqual(sum(len(x['experts']) for x in masks['a'].values()),512)
        masks['a']['0']['experts'].clear()
        self.assertEqual(len(masks['b']['0']['experts']),32)

    def test_rejects_invalid_statistics(self):
        with self.assertRaises(ValueError): global_control({})
        stats = {'a': {str(l): {'tokens': 1, 'counts': [0]*64} for l in range(16)}}
        stats['a']['0']['counts'][0] = float('nan')
        with self.assertRaises(ValueError): global_control(stats)
        stats['a']['0']['counts'][0] = 0
        stats['a']['0']['tokens'] = 0
        with self.assertRaises(ValueError): global_control(stats)


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


class RandomControlTests(unittest.TestCase):
    def test_budget_category_blindness_and_reproducibility(self):
        old = {c: {str(l): {'experts': list(range(32))} for l in range(16)} for c in ['a', 'b']}
        masks = random_controls(old, [1337, 2027, 31415])
        self.assertEqual(masks, random_controls(dict(reversed(list(old.items()))), [1337, 2027, 31415]))
        self.assertNotEqual(masks['random32_seed1337'], masks['random32_seed2027'])
        for categories in masks.values():
            self.assertEqual(categories['a'], categories['b'])
            self.assertEqual(sum(len(v['experts']) for v in categories['a'].values()), 512)
            for v in categories['a'].values():
                self.assertEqual(len(set(v['experts'])), 32)
                self.assertTrue(all(0 <= x < 64 for x in v['experts']))
        self.assertEqual(old['a']['0']['experts'], list(range(32)))

    def test_rejects_incomparable_budget_and_duplicate_seeds(self):
        old = {'a': {'0': {'experts': list(range(32))}}}
        with self.assertRaises(ValueError): random_controls(old, [1, 1])
        old['a']['0']['experts'].pop()
        with self.assertRaises(ValueError): random_controls(old, [1])


class CalibrationReuseTests(unittest.TestCase):
    def donor(self):
        protocol = dict(per_category=1, batch_size=1, active=8, max_length=384, seed=1)
        return {'identity': {'source': 'model', 'files': {'corpus': 'sha'}, 'protocol': protocol},
                'calibration_examples': [{'id': 'train', 'category': 'a'}],
                'evaluation_examples': [{'id': 'dev'}], 'calibration_done': ['train'],
                'stats': {'a': {str(l): {'tokens': 1, 'counts': [0]*64} for l in range(16)}},
                'records': [{'text': 'do not copy'}], 'completed': True}

    def test_copies_calibration_without_outputs_or_aliasing(self):
        import copy
        donor = self.donor()
        target = {'identity': copy.deepcopy(donor['identity']), 'records': [], 'completed': False}
        reuse_calibration(target, donor)
        self.assertEqual(target['records'], [])
        self.assertFalse(target['completed'])
        target['stats']['a']['0']['counts'][0] = 3
        self.assertEqual(donor['stats']['a']['0']['counts'][0], 0)

    def test_rejects_identity_protocol_coverage_and_overlap(self):
        import copy
        for mode in ['source', 'files', 'protocol', 'incomplete', 'overlap', 'layers']:
            with self.subTest(mode=mode):
                donor = self.donor()
                target = {'identity': copy.deepcopy(donor['identity'])}
                if mode in ('source', 'files'): donor['identity'][mode] = 'changed'
                elif mode == 'protocol': donor['identity']['protocol']['active'] = 4
                elif mode == 'incomplete': donor['calibration_done'] = []
                elif mode == 'overlap': donor['evaluation_examples'][0]['id'] = 'train'
                else: del donor['stats']['a']['0']
                with self.assertRaises(ValueError): reuse_calibration(target, donor)


class SelectionComparisonTests(unittest.TestCase):
    def fixture(self, name):
        return {'completed': True, 'identity': {'source': 'model', 'files': {'data':'sha'},
                'protocol': dict(new_tokens=256, active=8, max_length=384, batch_size=4)},
                'evaluation_examples': [{'id':'q', 'tokens':[1,2], 'category':'a'}],
                'masks': {name: {'a': {'0': {'experts':list(range(32))}}}},
                'records': [{'variant':name, 'id':'q', 'ce':2., 'eos':True,
                             'ready': {'loads':32, 'host_to_device_bytes':128},
                             'after': {'loads':32, 'host_to_device_bytes':128}}]}

    def compare(self, a, b):
        import json, tempfile
        from pathlib import Path
        from asi.analysis.mask_intervention_report import compare_selection
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d)/'a.json',Path(d)/'b.json']
            for p,r in zip(paths,[a,b]): p.write_text(json.dumps(r),encoding='utf-8')
            return compare_selection(*paths)

    def test_does_not_pool_timings_or_claim_independent_test(self):
        r=self.compare(self.fixture('category'),self.fixture('random'))
        self.assertEqual(len(r['rows']),2)
        self.assertFalse(r['timing_comparable'])
        self.assertFalse(r['independent_test'])
        self.assertEqual(r['rows'][0]['mean_ce'],2.)

    def test_rejects_partial_mismatched_or_invalid_results(self):
        import copy
        for fault in ['pending','tokens','transfer','duplicate','protocol','nan','source']:
            with self.subTest(fault=fault):
                a,b=self.fixture('category'),self.fixture('random')
                if fault=='pending': b['completed']=False
                elif fault=='tokens': b['evaluation_examples'][0]['tokens']=[9]
                elif fault=='transfer': b['records'][0]['after']['host_to_device_bytes']+=1
                elif fault=='duplicate': b['records'].append(copy.deepcopy(b['records'][0]))
                elif fault=='protocol': b['identity']['protocol']['active']=16
                elif fault=='source': b['identity']['source']='different'
                else: b['records'][0]['ce']=float('nan')
                with self.assertRaises(ValueError): self.compare(a,b)
