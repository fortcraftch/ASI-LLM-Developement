import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from asi.experiments.public_moe import decode, select_decode_token


class DecodingPolicyTests(unittest.TestCase):
    def test_greedy_unchanged_and_eos_stops(self):
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[[1., 3., 2.]]]), past_key_values=None)
        result = decode(Model(), torch.tensor([[0]]), 5, 'cpu', stop_token_ids=[1])
        self.assertEqual(result['token_ids'], [1])
        self.assertTrue(torch.equal(select_decode_token(torch.tensor([[2., 2., 1.]])), torch.tensor([[0]])))

    def test_nucleus_keeps_crossing_token_and_excludes_tail(self):
        logits = torch.tensor([[.6, .3, .1]]).log()
        generator = torch.Generator().manual_seed(9)
        tokens = {select_decode_token(logits, 1., .7, generator).item() for _ in range(100)}
        self.assertEqual(tokens, {0, 1})
        self.assertEqual(select_decode_token(logits, 1., .1, generator).item(), 0)

    def test_response_seed_independent_of_global_rng(self):
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.zeros(1, 1, 20), past_key_values=None)
        def run(seed):
            return decode(Model(), torch.tensor([[0]]), 20, 'cpu', temperature=.7, top_p=.9, seed=seed)['token_ids']
        first = run(1337)
        torch.rand(200)
        self.assertEqual(first, run(1337))
        self.assertNotEqual(first, run(2027))

    def test_invalid_configuration_fails_before_forward(self):
        for kwargs in ({'temperature':-1}, {'top_p':0}, {'temperature':float('nan')}, {'temperature':.7}):
            with self.assertRaises(ValueError):
                decode(None, torch.tensor([[1]]), 2, 'cpu', **kwargs)


class DecodingReportTests(unittest.TestCase):
    def fixture(self):
        report = {'completed': True, 'examples': [{'id': 'q'}],
                  'decoding_plan': {'arms': ['original', 'restricted'],
                                    'protocol': {'new_tokens': 2},
                                    'policies': [{'name': 'sample', 'seeds': [1, 2]}],
                                    'limitations': []}, 'records': []}
        for arm in ['original', 'restricted']:
            for seed in [1, 2]:
                report['records'].append(dict(arm=arm, id='q', decoding_policy='sample',
                    sampling_seed=seed, input_ids=[1], ready={'loads': 0, 'host_to_device_bytes': 0},
                    after={'loads': 0, 'host_to_device_bytes': 0}, eos=True, text='<generated>', token_ids=[1],
                    metrics={'repeated_trigram_fraction': .2}))
        return report

    def test_seeds_are_not_independent_questions(self):
        from asi.analysis.mask_intervention_report import build_decoding
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'report.json'
            report = self.fixture()
            path.write_text(json.dumps(report), encoding='utf-8')
            result = build_decoding(path, report)
            self.assertEqual(result['responses'], 4)
            self.assertEqual(result['independent_questions'], 1)
            self.assertEqual(result['groups'][0]['mean_generated_tokens'], 1)
            self.assertEqual(result['groups'][0]['budget_exhaustion_fraction'], 0)
            self.assertIn('&lt;generated&gt;', (path.parent/'analysis/index.html').read_text(encoding='utf-8'))

    def test_rejects_incomplete_mismatched_and_swapping_runs(self):
        from asi.analysis.mask_intervention_report import build_decoding
        for corruption in ['partial', 'duplicate', 'prompt', 'transfer', 'nan', 'duplicate_question']:
            report = self.fixture()
            if corruption == 'partial': report['records'].pop()
            elif corruption == 'duplicate': report['records'].append(report['records'][0])
            elif corruption == 'prompt': report['records'][0]['input_ids'] = [2]
            elif corruption == 'nan': report['records'][0]['metrics']['repeated_trigram_fraction'] = float('nan')
            elif corruption == 'duplicate_question': report['examples'].append({'id': 'q'})
            else: report['records'][-1]['after']['host_to_device_bytes'] = 10
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                build_decoding(Path('unused.json'), report)
