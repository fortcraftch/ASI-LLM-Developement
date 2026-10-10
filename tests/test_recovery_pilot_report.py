import unittest
from asi.analysis.recovery_pilot_report import mean_ce, repetition


class PilotReportTests(unittest.TestCase):
    def test_ce_does_not_use_mixed_objective(self):
        self.assertEqual(mean_ce({'mean_loss': 99, 'records': [
            {'ce': 1., 'loss': 50}, {'ce': 3., 'loss': 148}]}), 2.)

    def test_repetition_uses_token_sequences(self):
        self.assertEqual(repetition([1, 1, 1]), 0.)
        self.assertEqual(repetition([1, 1, 1, 1, 1]), .5)
        self.assertEqual(repetition([1, 2, 3, 4, 5]), 0.)


if __name__ == '__main__':
    unittest.main()
