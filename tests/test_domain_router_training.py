import unittest
import torch
from test_expert_runtime import tiny
from asi.models.domain import Gate
from asi.experiments.domain_sft import configure_trainable

class RouterOnlyTests(unittest.TestCase):
    def test_update_changes_only_gate(self):
        model=tiny(); configure_trainable(model, True)
        model.set_active_pools(list(range(model.config.n_pools)))
        before={k:v.clone() for k,v in model.state_dict().items()}
        gate_ids={id(p) for m in model.modules() if isinstance(m,Gate) for p in m.parameters()}
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},gate_ids)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        x=torch.randint(0,model.config.vocab_size,(1,8))
        logits,_=model(x)
        torch.nn.functional.cross_entropy(logits.flatten(0,1),x.flatten()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad))
        opt.step()
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(changed)
        named=dict(model.named_parameters())
        self.assertTrue(all(id(named[k]) in gate_ids for k in changed))
        self.assertEqual(model.config.routing_mode,'learned')
