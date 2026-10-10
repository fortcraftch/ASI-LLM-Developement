import unittest
import torch
from asi.models.domain import GPT, GPTConfig, Gate
from asi.experiments.checkpoint_eval import configure_mode


class RoutingAblationsTest(unittest.TestCase):
    def test_uniform_all_preserves_scale_and_parameters(self):
        config=GPTConfig(n_layer=2,n_embd=32,n_head=2,inter_dim=64,moe_inter_dim=8,
                         kv_lora_rank=16,qk_nope_head_dim=8,qk_rope_head_dim=4,v_head_dim=8,
                         block_size=16,max_seq_len=16,original_seq_len=16,max_batch_size=1,
                         routing_mode='uniform_pool')
        model=GPT(config)
        gate=next(m for m in model.modules() if isinstance(m,Gate))
        original=gate.weight.detach().clone()
        x=torch.randn(3,32)
        configure_mode(model,'all_uniform','uniform_pool')
        weights,indices=gate(x,torch.ones(config.n_routed_experts,dtype=torch.bool))
        self.assertEqual(indices.shape,(3,config.n_routed_experts))
        torch.testing.assert_close(weights.sum(-1),torch.full((3,),gate.route_scale))
        configure_mode(model,'all_router','uniform_pool')
        self.assertEqual(gate(x)[1].shape,(3,config.n_activated_experts))
        configure_mode(model,'classified','uniform_pool')
        mask=torch.arange(config.n_routed_experts)<config.n_activated_experts
        self.assertTrue((gate(x,mask)[1]<config.n_activated_experts).all())
        torch.testing.assert_close(gate.weight,original)
        with self.assertRaises(ValueError):
            gate(x,torch.ones_like(mask))


if __name__=='__main__':
    unittest.main()
