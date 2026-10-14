import unittest
from unittest.mock import Mock

import torch

from asi.runtime.ensemble import BankResidency


class BankResidencyTests(unittest.TestCase):
    def models(self):
        # Mock movement only: verifies policy, not actual CUDA behavior.
        return {k: Mock(parameters=lambda: iter([torch.zeros(3, 4)])) for k in ('a', 'b')}

    def test_selected_hits_do_not_move_and_switch_evicts(self):
        models = self.models()
        bank = BankResidency(models, 'cuda', 'selected')
        self.assertIs(bank.activate('a'), models['a'])
        calls = models['a'].to.call_count
        bank.activate('a')
        self.assertEqual(models['a'].to.call_count, calls)
        bank.activate('b')
        models['a'].to.assert_called_with('cpu')
        models['b'].to.assert_called_with('cuda')
        self.assertEqual(bank.snapshot()['loads'], 2)
        self.assertEqual(bank.snapshot()['evictions'], 1)
        self.assertEqual(bank.weight_h2d_bytes, 96)
        self.assertEqual(bank.weight_d2h_bytes, 48)

    def test_all_resident_switches_without_transfer(self):
        models = self.models()
        bank = BankResidency(models, 'cuda', 'all')
        before = bank.snapshot()
        bank.activate('a')
        bank.activate('b')
        self.assertEqual(bank.loads, before['loads'])
        self.assertEqual(bank.evictions, 0)
        for model in models.values(): model.to.assert_called_once_with('cuda')

    def test_cpu_does_not_claim_gpu_transfers(self):
        bank = BankResidency(self.models(), 'cpu', 'selected')
        bank.activate('a')
        bank.activate('b')
        self.assertEqual(bank.loads, 0)
        self.assertEqual(bank.weight_h2d_bytes, 0)
        with self.assertRaises(ValueError): bank.activate('missing')
        self.assertEqual(bank.active, 'b')


if __name__ == '__main__':
    unittest.main()
