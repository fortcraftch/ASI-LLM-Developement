import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from asi import ROOT
from asi.experiments.comparison import (bind_models, build_jobs, prepare, readiness, session_workloads,
                                        validate_config, verify_bundle)
from asi.experiments.comparison_report import paired, quality, summarize, timing
from asi.experiments.comparison_run import choose_mapping
from test_expert_runtime import tiny


def config():
    return json.loads((ROOT/'configs/comparison_suite.json').read_text(encoding='utf-8'))


class ComparisonTests(unittest.TestCase):
    def test_preparation_never_loads_weights_and_freezes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['expert_00_programming', 'expert_03_math', 'expert_02_ai']
            pools = {}
            for i, name in enumerate(names):
                category = f'category_{i}'; (root/category).mkdir()
                pools[name] = {'categories': [category]}
                if i < 2:
                    np.save(root/category/'val_000.npy', np.arange(100, dtype=np.uint16))
                np.save(root/category/'train_000.npy', np.arange(300, dtype=np.uint16))
            manifest = root/'pools.json'; manifest.write_text(json.dumps({'pools': pools}))
            c = config(); c['data_root'] = str(root); c['pool_manifest'] = str(manifest)
            c['quality'].update(lengths=[8], windows_per_class=4)
            with patch('torch.load', side_effect=AssertionError('Preparation loaded weights')):
                lock = prepare(c, root/'bundle')
                loaded, _ = verify_bundle(root/'bundle')
            self.assertEqual(lock['state'], 'prepared_not_executed')
            self.assertEqual(lock['coverage']['8'][names[2]]['windows'], 0)
            self.assertFalse(readiness(loaded)['ready_paths'])
            windows = json.loads((root/'bundle/windows.json').read_text())
            self.assertEqual(len(windows), 8)
            self.assertTrue(all(w['split'] == 'val' and w['tokens_sha256'] for w in windows))
            registered = copy.deepcopy(c)
            registered['models']['original']['checkpoint'] = 'future_checkpoint.pt'
            bind_models(root/'bundle', registered, root/'bound')
            self.assertEqual((root/'bundle/windows.json').read_bytes(), (root/'bound/windows.json').read_bytes())
            self.assertEqual(verify_bundle(root/'bound')[0]['models']['original']['checkpoint'], 'future_checkpoint.pt')
            registered['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'model slots only'):
                bind_models(root/'bundle', registered, root/'invalid')
            with self.assertRaises(FileExistsError): prepare(c, root/'bundle')
            (root/'bundle/jobs.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Frozen experiment changed'):
                verify_bundle(root/'bundle')

    def test_train_split_and_invalid_budgets_rejected(self):
        c = config(); c['quality']['split'] = 'train'
        with self.assertRaises(ValueError): validate_config(c)
        c = config(); c['capacities_per_layer'] = [0]
        with self.assertRaises(ValueError): validate_config(c)

    def test_matrix_deterministic_quality_not_repeated(self):
        c = config(); jobs = build_jobs(c)
        self.assertEqual(jobs, build_jobs(c))
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        self.assertEqual({j['role'] for j in jobs}, {'original','classified','domain'})
        self.assertTrue(all(j['capacity'] == 2 for j in jobs if j['policy'] == 'fixed'))
        self.assertTrue(all(j['repeat'] == 0 for j in jobs if j['phase'] == 'quality'))
        self.assertEqual({j['cache_start'] for j in jobs if j['phase'] == 'generation'}, {'cold','warm'})

    def test_workloads_cover_transitions_without_generated_history(self):
        sessions = session_workloads(['expert_00_programming','expert_03_math'])
        self.assertEqual(len(sessions), 10)
        self.assertEqual({s['scenario'] for s in sessions}, {'stable','switch','alternating','return','mixed'})
        self.assertTrue(all(len(s['turns']) == 6 for s in sessions))
        self.assertTrue(all(len(t['all_labels']) == 2 for s in sessions if s['scenario']=='mixed' for t in s['turns']))

    def test_domain_pool_mapping_does_not_invent_experts(self):
        model = tiny().eval(); names = ['code','math']
        mapping, reason = choose_mapping(model, {'adapter':'domain'}, 'domain',
            {'policy':'fixed','label_source':'oracle','candidates':2}, names, 'unused','unused')
        self.assertIsNone(reason)
        for layers in mapping['layers'].values():
            self.assertEqual(layers, {'code':[0,1], 'math':[2,3]})
        mapping, reason = choose_mapping(model, {'adapter':'domain'}, 'domain',
            {'policy':'restrict','label_source':'oracle','candidates':4}, names, 'unused','unused')
        self.assertIsNone(mapping)
        self.assertIn('fewer than 4', reason)

    def test_mapping_from_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mapping.json'; path.write_text(json.dumps({'provenance': {'checkpoint_sha256':'wrong'}}))
            with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                choose_mapping(tiny().eval(), {'adapter':'native','mapping':str(path)}, 'classified',
                               {'policy':'fixed','label_source':'oracle','candidates':2}, ['code','math'], 'right','manifest')

    def test_weighted_quality_and_paired_contrasts(self):
        rows = [{'window':'a','tokens':2,'nll':1.,'correct_tokens':1,'argmax':[1,2], 'true_class':'code','events':[]},
                {'window':'b','tokens':1,'nll':4.,'correct_tokens':0,'argmax':[3], 'true_class':'math','events':[]}]
        self.assertAlmostEqual(quality(rows)['nll'], 2.)
        self.assertAlmostEqual(quality(rows)['macro_domain_nll'], 2.5)
        other = copy.deepcopy(rows)
        for row in other: row['nll'] += .5
        result = paired(other, rows, 42)
        self.assertAlmostEqual(result['delta_nll'], .5)
        self.assertEqual(result['argmax_agreement'], 1.)
        with self.assertRaises(ValueError): paired(other[:-1], rows, 42)
        with self.assertRaises(ValueError): paired(other+other, rows, 42)

    def test_timing_no_decode_is_not_perfect_hit_rate(self):
        row = {'prepare_transfers':{'host_to_device_bytes':0},'classification_seconds':0,
               'prepare_seconds':0,'events':[],'memory':{'process_rss_bytes':100,'cuda_peak_allocated_bytes':200}}
        result = timing([row])
        self.assertNotIn('decode', result)
        self.assertEqual(result['generated_tokens'], 0)


if __name__ == '__main__':
    unittest.main()
