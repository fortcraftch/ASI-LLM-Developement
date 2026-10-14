import copy
import unittest

from asi.analysis.checkpoint_report import compare_full_suite


class FullSuiteComparisonTest(unittest.TestCase):
    def setUp(self):
        self.report = dict(completed=True, limit=None, evaluator_sha256='eval',
                           checkpoint_sha256='weights', datasets={'dataset': {'fingerprint': 'a', 'rows': 10}},
                           tasks={'task': {'accuracy': .5, 'total': 10}})

    def test_accepts_native_and_single_mode(self):
        mode = copy.deepcopy(self.report)
        mode['results'] = {'classified': {'tasks': mode.pop('tasks')}}
        self.assertEqual(set(compare_full_suite({'native': [self.report], 'asi': [mode]}, ['task'])), {'native', 'asi'})

    def test_rejects_invalid_evidence(self):
        for key, value in [('completed', False), ('limit', 100), ('evaluator_sha256', 'other'),
                           ('datasets', {}), ('tasks', {'task': {'accuracy': .5, 'total': 9}}),
                           ('tasks', {'task': {'accuracy': float('nan'), 'total': 10}})]:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.report)
                bad[key] = value
                with self.assertRaises(ValueError):
                    compare_full_suite({'a': [self.report], 'b': [bad]}, ['task'])

    def test_rejects_overlapping_blocks(self):
        with self.assertRaises(ValueError):
            compare_full_suite({'a': [self.report], 'b': [self.report, self.report]}, ['task'])

    def test_rejects_unselected_modes(self):
        bad = copy.deepcopy(self.report)
        bad['results'] = {'classified': {'tasks': bad['tasks']}, 'all_router': {'tasks': bad['tasks']}}
        with self.assertRaises(ValueError):
            compare_full_suite({'a': [self.report], 'b': [bad]}, ['task'])

    def test_rejects_raw_versus_normalized_accuracy(self):
        normalized = copy.deepcopy(self.report)
        normalized['tasks']['task']['accuracy_norm'] = .6
        with self.assertRaises(ValueError):
            compare_full_suite({'a': [self.report], 'b': [normalized]}, ['task'])
