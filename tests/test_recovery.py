import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from asi.experiments.recovery import (enable_routers, router_state, restore_routers, objective, read_examples,
    adaptation_state, restore_adaptation, robust_score, supervised_rows, compare_reports)
from asi.models.recovery_lora import ExpertLoRA
from asi.models.hf_olmoe import OlmoeExpertCache
from test_public_moe import tiny


class RecoveryTests(unittest.TestCase):
    def test_comparison_checks_exposure_and_identity(self):
        settings={key:1 for key in ('supervised','steps','seed','lr','alpha','active','experts','mask_sizes',
                                    'max_length','new_tokens','generation_per_category')}
        settings['adaptation']='router'
        records=[{'id':'x','mask':'2','ce':3.}]
        test={'records':records,'masks':{'2':{'records':records}}}
        generation={'mask':'2','exact_match':True,'repeated_trigram_fraction':0.,'cache_before':{'loads':2},'cache_after':{'loads':2}}
        report={'completed':True,'action':'train','identity':{'source':'same'},'settings':settings,
                'steps':[{'id':'x','mask':'2'}],'test_before':test,'test_after':test,
                'generations':[dict(generation,stage=s) for s in ('before','after')],
                'selected_step':1,'trainable_parameters':10,'peak_cuda_bytes':1024,'training_with_validation_seconds':2.}
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ('a.json','b.json')]
            for path in paths: path.write_text(json.dumps(report))
            self.assertEqual(len(compare_reports(paths)['rows']),4)
            bad=copy.deepcopy(report); bad['steps'][0]['id']='another'
            paths[1].write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'exposure'): compare_reports(paths)

    def test_expert_updates_survive_eviction_and_restore_without_base_changes(self):
        model=tiny()
        cache=OlmoeExpertCache(model,'cpu',4)
        routers=enable_routers(model)
        for p in routers: p.requires_grad_(False)
        first={str(i):{'experts':[0,1],'weights':[.25,.25]} for i in range(2)}
        second={str(i):{'experts':[2,3],'weights':[.25,.25]} for i in range(2)}
        bank={'2':{'a':first,'b':second}}
        adapter=ExpertLoRA(model,bank,rank=2)
        cache.expert_adapter=adapter
        frozen={n:p.detach().clone() for n,p in model.named_parameters()}
        initial=adaptation_state(model,adapter)
        row={'tokens':[3,4,5,6],'prompt_length':2}
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.01,weight_decay=0)
        cache.set_context(first,mixture='restricted',active=2)
        with torch.no_grad(): before=model(torch.tensor([row['tokens']])).logits.clone()
        loss,_,_=objective(model,row,torch.device('cpu'),1.)
        loss.backward(); optimizer.step()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.parameters()))
        learned=adaptation_state(model,adapter)
        with torch.no_grad(): after=model(torch.tensor([row['tokens']])).logits.clone()
        self.assertFalse(torch.equal(before,after))
        cache.set_context(second,mixture='restricted',active=2)
        cache.set_context(first,mixture='restricted',active=2)
        loads=cache.stats['loads']
        with torch.no_grad(): returned=model(torch.tensor([row['tokens']])).logits
        torch.testing.assert_close(returned,after,rtol=0,atol=0)
        self.assertEqual(cache.stats['loads'],loads)
        for n,p in model.named_parameters(): torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        restore_adaptation(model,adapter,initial)
        with torch.no_grad(): torch.testing.assert_close(model(torch.tensor([row['tokens']])).logits,before,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lora.pt'; torch.save(learned,path)
            restore_adaptation(model,adapter,torch.load(path,weights_only=True))
        with torch.no_grad(): torch.testing.assert_close(model(torch.tensor([row['tokens']])).logits,after,rtol=0,atol=0)
        cache.close()

    def test_worst_group_rejects_hidden_regression(self):
        baseline={'records':[{'mask':'16','category':'a','loss':2.},{'mask':'32','category':'b','loss':2.}]}
        candidate={'records':[{'mask':'16','category':'a','loss':.1},{'mask':'32','category':'b','loss':2.1}]}
        self.assertGreater(robust_score(candidate,baseline),1.)
        self.assertEqual(robust_score(baseline,baseline),1.)

    def test_supervised_answer_includes_eos_and_masks_prompt(self):
        from unittest.mock import patch
        class Tokenizer:
            eos_token_id=9
            def encode(self,text,add_special_tokens=False): return [5,6]
        with patch('asi.experiments.recovery.input_ids',return_value=torch.tensor([[1,2,3]])):
            rows=supervised_rows({'examples':[{'id':'x','prompt':'q','answer':'a'}]},Tokenizer(),6)
        self.assertEqual(rows[0]['tokens'],[1,2,3,5,6,9])
        self.assertEqual(rows[0]['prompt_length'],3)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_bf16_forward_fp32_router_backward(self):
        model=tiny().to(device='cuda',dtype=torch.bfloat16)
        cache=OlmoeExpertCache(model,'cuda',4)
        parameters=enable_routers(model)
        mapping={str(i):{'experts':[0,1],'weights':[.25,.25]} for i in range(2)}
        adapter=ExpertLoRA(model,{'2':{'a':mapping}},rank=2)
        cache.expert_adapter=adapter
        parameters+=list(adapter.parameters())
        cache.set_context(mapping,mixture='restricted',active=2)
        row={'tokens':[3,4,5,6],'prompt_length':2,'teacher_logp':torch.zeros(2,64).log_softmax(-1)}
        loss,_,_=objective(model,row,torch.device('cuda'))
        loss.backward()
        self.assertTrue(all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        cache.close()

    def test_generation_stops_at_eos(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from asi.experiments.public_moe import decode
        class Model:
            def __call__(self,**kwargs):
                return SimpleNamespace(logits=torch.tensor([[[0.,10.,0.]]]),past_key_values=None)
        with patch('torch.cuda.synchronize'):
            result=decode(Model(),torch.tensor([[2]]),10,'cpu',stop_token_ids=[1])
        self.assertEqual(result['token_ids'],[1])
        self.assertFalse(result['fixed_length'])

    def test_step_updates_only_persistent_routers_and_roundtrips(self):
        model=tiny()
        teacher=copy.deepcopy(model)
        cache=OlmoeExpertCache(model,'cpu',4)
        parameters=enable_routers(model)
        original=router_state(model)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        mapping={str(i):{'experts':[0,1],'weights':[.25,.25]} for i in range(2)}
        cache.set_context(mapping,mixture='restricted',active=2)
        tokens=[3,4,5,6,7]
        with torch.no_grad():
            logp=teacher(torch.tensor([tokens])).logits[0,1:-1].log_softmax(-1)
        row={'tokens':tokens,'prompt_length':2,'teacher_logp':logp}
        optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=0)
        loads=cache.stats['loads']
        loss,ce,kl=objective(model,row,torch.device('cpu'))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        optimizer.step()
        changed=router_state(model)
        self.assertTrue(any(not torch.equal(original[k],changed[k]) for k in original))
        for name,param in model.named_parameters():
            if name in frozen: torch.testing.assert_close(param,frozen[name],rtol=0,atol=0)
        self.assertEqual(cache.stats['loads'],loads)
        restore_routers(model,original)
        for k,v in router_state(model).items(): torch.testing.assert_close(v,original[k],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'router.pt'; torch.save({'routers':changed},path)
            restore_routers(model,torch.load(path,weights_only=True)['routers'])
            for k,v in router_state(model).items(): torch.testing.assert_close(v,changed[k],rtol=0,atol=0)
        cache.close()

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'examples.json'
            path.write_text(json.dumps({'examples':[{'id':s,'split':s,'prompt':'Same question','answer':'x'} for s in ['train','dev','test']]}))
            with self.assertRaises(ValueError): read_examples(path)


if __name__=='__main__': unittest.main()
