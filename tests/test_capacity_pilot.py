from dataclasses import asdict
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import numpy as np
import torch

from asi.experiments.capacity_pilot import initialize_paired,pool_schedule,run_variant
from asi.models.domain import GPT,GPTConfig
from asi.analysis.capacity_compare import routing_metrics,window_identity,compare


class CapacityPilotTest(unittest.TestCase):
    def test_normalization_preserves_selection_and_router_gradient(self):
        from asi.experiments.capacity_pilot import normalize_learned_gates
        model=GPT(self.config('learned'));gate=model.layers[1].ffn.gate
        logits=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
        before,indices=gate.route_logits(logits,None)
        normalize_learned_gates(model)
        after,new_indices=gate.route_logits(logits,None)
        self.assertTrue(torch.equal(indices,new_indices))
        torch.testing.assert_close(after.sum(-1),torch.tensor([gate.route_scale]))
        self.assertLess(float(before.sum()),gate.route_scale)
        after[0,0].backward()
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_interval_metrics_distinguish_concentration_from_category_dependence(self):
        def layer(counts,mass=1):
            total=sum(map(sum,counts))
            return {'assignments_by_input_pool':counts,'mixture_weight_sum':total*mass,'observed_tokens':total}
        restricted={'1':layer([[10,0],[0,10]])}
        m=routing_metrics(restricted)
        self.assertAlmostEqual(m['effective_experts'],2)
        self.assertAlmostEqual(m['normalized_pool_information'],1)
        concentrated={'1':layer([[19,1],[19,1]],.2)}
        m=routing_metrics(concentrated)
        self.assertLess(m['effective_experts'],1.3)
        self.assertAlmostEqual(m['normalized_pool_information'],0)
        self.assertAlmostEqual(m['selected_weight_mass'],.2)
        cumulative={'1':layer([[29,1],[19,11]])}
        m=routing_metrics(cumulative,concentrated)
        self.assertAlmostEqual(m['effective_experts'],2)
        self.assertAlmostEqual(m['normalized_pool_information'],1)

    def test_window_identity_detects_changed_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.npz',Path(tmp)/'b.npz'
            np.savez(a,x=np.arange(4));np.savez(b,x=np.arange(4))
            self.assertEqual(window_identity(a),window_identity(b))
            np.savez(b,x=np.arange(4)+1)
            self.assertNotEqual(window_identity(a),window_identity(b))

    def test_comparison_rejects_unmatched_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[]
            for seed in (1,2):
                run=root/str(seed);run.mkdir();runs.append(run)
                np.savez(run/'windows.npz',x=np.arange(4))
                plan={'status':'completed','settings':{'seed':seed,'steps':2},
                      'variants':['fixed','free'],'source_recipe_sha256':'a',
                      'pool_manifest_sha256':'b','local_manifest_sha256':'c'}
                (run/'plan.json').write_text(json.dumps(plan))
                for name,policy in [('fixed','uniform_pool'),('free','learned')]:
                    layer={'assignments_by_input_pool':[[2,0],[0,2]],'mixture_weight_sum':4,'observed_tokens':4}
                    r={'status':'completed','variant':{'model':{'routing_mode':policy}},
                       'common_initial_weights_sha256':str(seed),'tokens':4,
                       'validation':[{'updates':0,'mean':4.},{'updates':2,'mean':3.}],
                       'usage_history':[{'updates':2,'layers':{'1':layer}}]}
                    (run/(name+'.json')).write_text(json.dumps(r))
            result=compare(runs,root/'report')
            self.assertEqual(result['mean_nll_difference'],0.)
            from asi.analysis.capacity_compare import compare_normalization
            import shutil
            normalized=[]
            for i,run in enumerate(runs):
                target=root/f'normalized_{i}';shutil.copytree(run,target);normalized.append(target)
                plan=json.loads((target/'plan.json').read_text());plan['settings']['normalize_selected']=True
                (target/'plan.json').write_text(json.dumps(plan))
                r=json.loads((target/'free.json').read_text());r['validation'][-1]['mean']=2.5
                (target/'free.json').write_text(json.dumps(r))
            effect=compare_normalization(runs,normalized,root/'effect')
            self.assertEqual(effect['mean_learned_nll_delta'],-.5)
            with self.assertRaises(ValueError): compare_normalization(runs,runs,root/'invalid_effect')
            np.savez(runs[1]/'windows.npz',x=np.arange(4)+1)
            with self.assertRaisesRegex(ValueError,'windows differ'): compare_normalization(runs,normalized,root/'bad_effect')
            with self.assertRaisesRegex(ValueError,'windows differ'):compare(runs,root/'bad')
    def config(self, routing='uniform_pool'):
        return GPTConfig(n_layer=2,n_embd=16,n_head=2,inter_dim=32,moe_inter_dim=8,
            kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,
            block_size=8,max_seq_len=8,original_seq_len=8,max_batch_size=1,
            vocab_size=32,n_pools=2,n_routed_experts=4,experts_per_pool=2,
            n_activated_experts=2,routing_mode=routing)

    def test_balanced_schedule_and_paired_initialization(self):
        schedule=pool_schedule(8,32,1337)
        self.assertEqual(schedule,pool_schedule(8,32,1337))
        self.assertEqual([schedule.count(i) for i in range(8)],[4]*8)
        with self.assertRaises(ValueError):pool_schedule(8,31,1337)
        a=GPT(self.config());b=GPT(self.config('learned'))
        self.assertEqual(initialize_paired(a,9),initialize_paired(b,9))
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb);torch.testing.assert_close(pa,pb,rtol=0,atol=0)

    def test_pilot_uses_all_pools_and_tracks_assignments(self):
        args=SimpleNamespace(seed=3,batch_size=1,seq_len=8,device='cpu',steps=2,
                             accumulation=1,warmup_steps=1,lr=1e-4)
        x=np.arange(16,dtype=np.uint16).reshape(2,1,8)
        arrays={s+'_'+k:v for s in ('train','val') for k,v in [('x',x),('y',x+1),('pools',np.array([0,1]))]}
        with tempfile.TemporaryDirectory() as tmp:
            r=run_variant(args,{'name':'tiny','model':asdict(self.config())},arrays,['a','b'],Path(tmp)/'result.json')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['tokens'],16)
        self.assertEqual(r['expert_usage']['1']['token_assignments'],[8,8,8,8])
        self.assertEqual(r['expert_usage']['1']['steps_with_nonzero_gradient'],[1,1,1,1])
        self.assertEqual(r['expert_usage']['1']['assignments_by_input_pool'],[[8,8,0,0],[0,0,8,8]])
        self.assertAlmostEqual(r['expert_usage']['1']['mean_selected_weight_mass'],1.)
        self.assertEqual(set(r['validation'][-1]['by_pool']),{'a','b'})


if __name__=='__main__':unittest.main()
