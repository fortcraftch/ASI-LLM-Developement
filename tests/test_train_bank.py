import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from asi.experiments.train_bank import latest_checkpoint, validate_checkpoint, main


class TrainBankTests(unittest.TestCase):
    def test_checkpoint_numeric_order_ignores_uncommitted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['model_9.pt', 'model_10.pt', 'model_11.pt.tmp', 'model_bad.pt']:
                (root/name).touch()
            self.assertEqual(latest_checkpoint(root).name, 'model_10.pt')

    def test_checkpoint_pool_identity(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model_1.pt'
            torch.save({'step': 1, 'training_recipe': {'pool': 'a'}}, path)
            self.assertEqual(validate_checkpoint(path, {'recipe': {'pool': 'a'}}), 2)
            with self.assertRaises(ValueError):
                validate_checkpoint(path, {'recipe': {'pool': 'b'}})

    def test_completed_bank_registers_checkpoint_without_training(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = {'pool': 'a', 'source_manifest_sha256': 'test'}
            plan = {'pool_order': ['a'], 'jobs': [{'pool': 'a', 'recipe': recipe,
                     'recipe_file': 'router_a.json', 'parameters': 1, 'steps': 2, 'tokens': 4}],
                    'parameters': 1, 'tokens': 4, 'batch_size': 8,
                    'seed': 1337, 'checkpoint_policy': {'save_interval': 25, 'keep': 3}}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'a').mkdir()
            torch.save({'step': 1, 'training_recipe': recipe}, root/'a/model_1.pt')
            with patch('asi.experiments.train_bank.plan_bank', return_value=dict(plan)), \
                 patch('asi.experiments.train_bank.manifest_matches', return_value=True), \
                 patch('asi.experiments.train_bank.subprocess.run') as run, \
                 patch('sys.argv', ['train-bank', '--data-root', temp, '--output', temp, '--resume']):
                main()
            run.assert_not_called()
            self.assertEqual(json.loads((root/'router_bank.json').read_text()), {'a': 'a/model_1.pt'})
