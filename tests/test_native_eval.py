from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from asi.models.original import import_architecture,DEFAULT_ARCHITECTURE
from asi.experiments.native_eval import NativeView,audit_routing
from asi.experiments.benchmark_batch import generate_requests,pack


@unittest.skipUnless(DEFAULT_ARCHITECTURE.exists(),'Original architecture required')
class NativeEvalTest(unittest.TestCase):
    def setUp(self):
        _,Config,Model=import_architecture(DEFAULT_ARCHITECTURE)
        cfg=Config(n_layer=2,n_embd=16,n_head=2,inter_dim=32,moe_inter_dim=8,
                   kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,
                   block_size=16,max_seq_len=16,original_seq_len=16,max_batch_size=2,vocab_size=32)
        torch.manual_seed(3);self.native=Model(cfg).eval();self.view=NativeView(self.native)
        self.enc=SimpleNamespace(encode=lambda s:[1,2,3],decode=lambda ids:','.join(map(str,ids)),eot_token=31)

    @torch.no_grad()
    def test_batched_native_generation_matches_full_context(self):
        adapter=SimpleNamespace(model=self.view,device='cpu',enc=self.enc,block_size=16,
                                route_key=lambda ids:(),activate=lambda ids:None,supports_kv_cache=False)
        requests={(pack([1,2,3]),4,True):None,(pack([2,3,4]),4,False):None}
        generate_requests(adapter,requests,2,32)
        for (raw,steps,stop),actual in requests.items():
            from asi.experiments.benchmark_batch import unpack
            ids=unpack(raw);expected=[]
            for _ in range(steps):
                token=int(self.native(torch.tensor([ids]))[0][0,-1].argmax());ids.append(token);expected.append(token)
                if stop and token==31:break
            self.assertEqual(actual,expected)

    @torch.no_grad()
    def test_audit_counts_only_new_last_positions_and_preserves_logits(self):
        x=torch.tensor([[1,2,3]]);before=self.native(x)[0].clone()
        report=audit_routing(self.native,self.enc,['prompt'],'cpu',4)
        probe=report['probes'][0];layer=probe['layers']['1']
        self.assertEqual(sum(layer['prefill_assignments']),6)
        self.assertEqual(sum(layer['last_position_assignments']),2*probe['generated_tokens'])
        self.assertEqual(layer['activated_per_position'],2)
        self.assertFalse(self.native.layers[1].ffn.gate._forward_hooks)
        torch.testing.assert_close(before,self.native(x)[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
