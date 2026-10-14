import unittest
import json
from pathlib import Path
from dataclasses import asdict
import torch

from asi.models.domain import GPT, GPTConfig, Gate
from test_expert_runtime import tiny


class MatchedRoutingTests(unittest.TestCase):
    def test_real_recipes_differ_only_in_experimental_routing(self):
        from asi.models.design import validate_recipe, count_parameters
        root = Path(__file__).resolve().parents[1] / 'configs/matched_moe_124m_v2'
        rows = [json.loads((root / (name + '.json')).read_text(encoding='utf-8'))
                for name in ('free', 'pool_learned', 'pool_uniform')]
        self.assertEqual(rows[0]['model'], rows[1]['model'])
        for row in rows:
            validate_recipe(row, row['pool_order'])
            self.assertEqual(count_parameters(row['model']), row['parameter_counts'])
            self.assertLessEqual(row['parameter_counts']['total'], 124_000_000)
            stripped = {k: v for k, v in row.items() if k not in ('model', 'family')}
            self.assertEqual(stripped, {k: v for k, v in rows[0].items() if k not in ('model', 'family')})
            self.assertEqual({k:v for k,v in row['model'].items() if k != 'routing_mode'},
                             {k:v for k,v in rows[0]['model'].items() if k != 'routing_mode'})

    def test_opt_in_scale_preserves_selection_and_legacy_behavior(self):
        config = tiny().config
        config.route_scale = 1.7
        gate = Gate(config)
        logits = torch.tensor([[2., 1., 0., -1.]], requires_grad=True)
        native, indices = gate.route_logits(logits)
        expected = logits.softmax(-1).gather(1, indices) * 1.7
        torch.testing.assert_close(native, expected)
        gate.normalize_selected_weights = True
        controlled, selected = gate.route_logits(logits)
        self.assertTrue(torch.equal(selected, indices))
        torch.testing.assert_close(controlled.sum(-1), torch.tensor([1.7]))
        mask = torch.tensor([False, False, True, True])
        restricted, selected = gate.route_logits(logits, mask)
        self.assertEqual(set(selected.flatten().tolist()), {2, 3})
        torch.testing.assert_close(restricted.sum(-1), torch.tensor([1.7]))
        restricted[0, 0].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_serialized_config_preserves_option_and_identical_initial_weights(self):
        config = asdict(tiny().config)
        config.pop('normalize_selected_weights')
        self.assertFalse(GPTConfig(**config).normalize_selected_weights)
        config['normalize_selected_weights'] = True
        torch.manual_seed(24)
        learned = GPT(GPTConfig(**config))
        config['routing_mode'] = 'uniform_pool'
        torch.manual_seed(24)
        uniform = GPT(GPTConfig(**config))
        self.assertTrue(uniform.config.normalize_selected_weights)
        for name, value in learned.state_dict().items():
            self.assertTrue(torch.equal(value, uniform.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
