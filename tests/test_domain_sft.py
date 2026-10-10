import unittest
import torch
from asi.experiments.domain_sft import assistant_loss,coverage_selection
from asi.models.domain import GPT, GPTConfig, Gate


class DomainSFTTest(unittest.TestCase):
    def test_coverage_reserves_rare_pools_without_duplicates(self):
        selected=coverage_selection([0,0,0,0,1,1,2],3,5,1)
        self.assertEqual(len(selected),5)
        self.assertEqual(len(set(selected)),5)
        self.assertIn(6,selected)
        with self.assertRaises(ValueError):coverage_selection([0,0,1],3,3,1)
    def test_prompt_and_padding_have_no_direct_loss_gradient(self):
        logits=torch.randn(1,4,7,requires_grad=True)
        targets=torch.tensor([[1,2,3,4]])
        mask=torch.tensor([[0,1,1,0]])
        loss=assistant_loss(logits,targets,mask)
        expected=torch.nn.functional.cross_entropy(logits[:,1:3].reshape(-1,7),targets[:,1:3].reshape(-1))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertEqual(float(logits.grad[:,[0,3]].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,1:3].abs().sum()),0.)
        with self.assertRaises(ValueError): assistant_loss(logits,targets,torch.zeros_like(mask))

    def test_fixed_pool_sft_updates_selected_experts_only(self):
        torch.manual_seed(3)
        cfg=GPTConfig(n_layer=2,n_embd=32,n_head=2,inter_dim=64,moe_inter_dim=8,
            kv_lora_rank=16,qk_nope_head_dim=8,qk_rope_head_dim=4,v_head_dim=8,
            block_size=16,max_seq_len=16,original_seq_len=16,max_batch_size=1,
            vocab_size=32,routing_mode='uniform_pool')
        model=GPT(cfg);model.set_active_pools([0])
        for module in model.modules():
            if isinstance(module,Gate):
                for p in module.parameters(): p.requires_grad_(False)
        for layer in model.layers:
            if hasattr(layer.ffn,'moe_backend'):layer.ffn.moe_backend='bmm'
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
        logits,_=model(torch.tensor([[1,2,3,4]]))
        assistant_loss(logits,torch.tensor([[2,3,4,5]]),torch.ones(1,4)).backward()
        opt.step()
        changed={n for n,p in model.named_parameters() if not torch.equal(before[n],p)}
        self.assertTrue(any('.experts.0.' in n for n in changed))
        self.assertTrue(any('.experts.1.' in n for n in changed))
        self.assertFalse(any('.gate.' in n for n in changed))
        for expert in range(2,cfg.n_routed_experts):
            self.assertFalse(any(f'.experts.{expert}.' in n for n in changed))
        self.assertTrue(any('shared_experts' in n for n in changed))
        self.assertTrue(any('.attn.' in n for n in changed))


if __name__=='__main__':unittest.main()
