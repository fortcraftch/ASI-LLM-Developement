import os
import unittest
from unittest.mock import patch
import torch
from test_expert_runtime import tiny


class BackendTests(unittest.TestCase):
    def test_invalid_backend_rejected(self):
        with patch.dict(os.environ, {'ASI_MOE_BACKEND': 'invalid'}):
            with self.assertRaisesRegex(ValueError, 'ASI_MOE_BACKEND'):
                tiny()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_forced_bmm_trains_without_grouped(self):
        with patch.dict(os.environ, {'ASI_MOE_BACKEND': 'bmm'}):
            model = tiny().cuda().train()
        model.set_active_pools([0])
        with patch('asi.models.domain.MoE._forward_grouped', side_effect=AssertionError('grouped must not execute')):
            x = torch.tensor([[1, 2, 3]], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, x)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.layers[1].ffn.experts[0].w1.weight.grad)
        self.assertIsNone(model.layers[1].ffn.experts[2].w1.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_runtime_error_is_not_hidden_by_fallback(self):
        with patch.dict(os.environ, {'ASI_MOE_BACKEND': 'grouped'}):
            model = tiny().cuda().eval()
        for layer in model.layers[1:]:
            layer.ffn._grouped_mm_available = True
        with patch('asi.models.domain.MoE._forward_grouped', side_effect=RuntimeError('simulated kernel failure')):
            with patch('asi.models.domain.MoE._forward_bmm', side_effect=AssertionError('unsafe fallback')):
                with self.assertRaisesRegex(RuntimeError, 'simulated kernel failure'):
                    model(torch.tensor([[1, 2]], device='cuda'))
