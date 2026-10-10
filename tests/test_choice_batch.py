import unittest
from types import SimpleNamespace
import torch
from asi.models.domain import GPT,GPTConfig
from asi.experiments.checkpoint_eval import choice_losses


class ChoiceBatchTest(unittest.TestCase):
    def test_padding_and_truncation_preserve_scores(self):
        torch.manual_seed(42)
        cfg=GPTConfig(n_layer=2,n_embd=32,n_head=2,inter_dim=64,moe_inter_dim=8,
            kv_lora_rank=16,qk_nope_head_dim=8,qk_rope_head_dim=4,v_head_dim=8,
            block_size=16,max_seq_len=16,original_seq_len=16,max_batch_size=4,
            vocab_size=32,routing_mode='uniform_pool')
        model=GPT(cfg).eval();model.set_active_pools([0])
        adapter=SimpleNamespace(enc=SimpleNamespace(encode=lambda s:[ord(c)%31 for c in s],eot_token=31),
            device='cpu',block_size=16,select=lambda ids:None,logits=lambda ids:model(ids)[0])
        for context in ('short','very long context exceeding window'):
            serial=choice_losses(adapter,context,['a','abcd','long completion','xy'],1)
            batch=choice_losses(adapter,context,['a','abcd','long completion','xy'],4)
            torch.testing.assert_close(torch.tensor(serial),torch.tensor(batch),rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
