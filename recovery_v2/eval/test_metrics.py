import unittest
import torch
from recovery_v2.eval.metrics import distribution_metrics, repetition, paired_bootstrap, mcnemar
from recovery_v2.eval.metrics import first_divergence, holm


class MetricsTests(unittest.TestCase):
    def test_divergence_censoring_and_eos(self):
        self.assertEqual(first_divergence([1,2,0], [1,3,0], 3)['position'], 1)
        self.assertTrue(first_divergence([1,0], [1,2], 2)['event_observed'])
        self.assertFalse(first_divergence([1,2,3], [1], 3)['event_observed'])
        self.assertEqual(first_divergence([1,2], [1,2], 1)['position'], 1)

    def test_holm_family(self):
        adjusted = holm({'a':.01, 'b':.03, 'c':.04})
        self.assertAlmostEqual(adjusted['a'], .03)
        self.assertAlmostEqual(adjusted['b'], .06)
        self.assertAlmostEqual(adjusted['c'], .06)
        with self.assertRaises(ValueError): holm({'x':float('nan')})

    def test_identity_and_mask(self):
        logits = torch.tensor([[[1., 2.], [8., -3.]]])
        result = distribution_metrics(logits, logits, torch.tensor([[1, 0]]), torch.tensor([[True, False]]))
        self.assertAlmostEqual(result['kl'], 0., places=6)
        self.assertEqual(result['tokens'], 1)
        self.assertEqual(result['top1_agreement'], 1.)

    def test_known_kl(self):
        t = torch.tensor([[.5, .5]]).log()
        s = torch.tensor([[.25, .75]]).log()
        result = distribution_metrics(t, s, torch.tensor([0]), torch.tensor([True]))
        self.assertAlmostEqual(result['kl'], .143841036, places=6)

    def test_loops(self):
        self.assertTrue(repetition([1, 2]*3)['loop'])
        self.assertFalse(repetition([1, 2]*2)['loop'])
        self.assertEqual(repetition([1]*5)['max_same_token_run'], 5)
        self.assertAlmostEqual(repetition([1]*5)['repeated_trigram_fraction'], 2/3)
        self.assertEqual(repetition([])['length'], 0)

    def test_pairing(self):
        result = paired_bootstrap({'a': 1., 'b': 2.}, {'b': 3., 'a': 2.})
        self.assertEqual(result['ci95'], [1., 1.])
        with self.assertRaises(ValueError):
            paired_bootstrap({'a': 1}, {'b': 1})

    def test_binary(self):
        self.assertEqual(mcnemar({'a': 1}, {'a': 1})['pvalue'], 1.)
        self.assertEqual(mcnemar(dict.fromkeys(range(6), 0), dict.fromkeys(range(6), 1))['pvalue'], .03125)


if __name__ == '__main__':
    unittest.main()
