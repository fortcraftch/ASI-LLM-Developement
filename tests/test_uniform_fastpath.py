import unittest
import torch
from asi.models.domain import GPTConfig,MoE


class UniformFastpathTest(unittest.TestCase):
    def test_outputs_masks_and_observer(self):
        torch.manual_seed(7)
        cfg=GPTConfig(n_embd=32,moe_inter_dim=12,routing_mode='uniform_pool')
        moe=MoE(cfg).eval();moe.moe_backend='bmm'
        with torch.no_grad():
            for parameter in moe.parameters():parameter.normal_(0,.02)
        x=torch.randn(2,9,32)
        with torch.no_grad():
            for pool in (0,3,0):
                mask=moe.allowed_mask_for_pools([pool])
                moe.uniform_fastpath=False;expected=moe(x,mask)
                moe.uniform_fastpath=True;actual=moe(x,mask)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            # In-place changes must invalidate cached expert ids.
            mask.copy_(moe.allowed_mask_for_pools([1]))
            actual=moe(x,mask);moe.uniform_fastpath=False
            torch.testing.assert_close(actual,moe(x,mask),rtol=1e-5,atol=1e-6)
            seen=[];moe.uniform_fastpath=True
            moe.gate.routing_observer=lambda *args:seen.append(args)
            moe(x,mask);self.assertEqual(len(seen),1)
            moe.gate.routing_observer=None
            with self.assertRaises(ValueError):moe(x,torch.ones_like(mask))

    def test_training_keeps_reference_path(self):
        moe=MoE(GPTConfig(n_embd=16,moe_inter_dim=8,routing_mode='uniform_pool'))
        with torch.no_grad():
            for parameter in moe.parameters():parameter.normal_(0,.02)
        moe.moe_backend='bmm'
        x=torch.randn(1,4,16,requires_grad=True)
        moe(x,moe.allowed_mask_for_pools([0])).sum().backward()
        self.assertIsNotNone(moe.experts[0].w1.weight.grad)
        self.assertIsNone(moe.experts[2].w1.weight.grad)


if __name__=='__main__':unittest.main()
