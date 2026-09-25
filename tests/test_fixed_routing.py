import copy
import unittest
from unittest.mock import Mock

import torch

from asi.experiments.fixed_study import fixed_mappings, paired_contrast
from asi.runtime.cache import NativeExpertSessionCache
from asi.runtime.routing import FixedExpertRouting
from asi.models.original import native_moes
from test_expert_runtime import tiny


class FixedRoutingTests(unittest.TestCase):
    def mapping(self):
        return {'layers':{str(layer):{'code':[0,1],'math':[2,3]} for layer in (1,2)},
                'fixed_mass':{str(layer):{'code':.4,'math':.6} for layer in (1,2)}}

    def test_bypasses_E_and_restores_overridden_forward(self):
        model=tiny().eval()
        originals={}
        for layer,moe in native_moes(model).items():
            originals[layer]=Mock(side_effect=RuntimeError('E must not run'))
            moe.gate.forward=originals[layer]
        cache=NativeExpertSessionCache(model,self.mapping(),device='cpu',max_hot_experts=4,
                                       max_experts_per_layer=2,policy='fixed')
        try:
            cache.set_context_labels(['code'])
            before=cache.snapshot()
            with torch.no_grad():
                result=model(torch.tensor([[1,2,3,4]]))[0]
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(cache.snapshot().get('misses',0),before.get('misses',0))
            for original in originals.values(): original.assert_not_called()
            self.assertEqual(set(cache.hot),{(1,0),(1,1),(2,0),(2,1)})
            cache.set_context_labels(['math'])
            self.assertEqual(set(cache.hot),{(1,2),(1,3),(2,2),(2,3)})
        finally:
            cache.close()
        for layer,moe in native_moes(model).items():
            self.assertIs(moe.gate.forward,originals[layer])

    def test_weights_constant_and_no_multilabel_union(self):
        model=tiny().eval(); moes=native_moes(model)
        router=FixedExpertRouting(moes,self.mapping(),'calibrated')
        router.select(['code']); router.attach()
        try:
            x=torch.randn(5,16)
            for moe in moes.values():
                weights,ids=moe.gate(x)
                torch.testing.assert_close(weights,torch.full((5,2),.2))
                torch.testing.assert_close(ids,torch.tensor([[0,1]]*5))
                torch.testing.assert_close(moe.gate(x*100)[0],weights)
            with self.assertRaises(ValueError): router.select(['code','math'])
            self.assertEqual(router.label,'code')
        finally: router.close()
        self.assertTrue(all('forward' not in moe.gate.__dict__ for moe in moes.values()))

    def test_rejects_unequal_class_sizes_and_insufficient_residency(self):
        model=tiny().eval(); mapping=self.mapping()
        mapping['layers']['1']['math']=[2]
        with self.assertRaises(ValueError): FixedExpertRouting(native_moes(model),mapping)
        with self.assertRaises(ValueError):
            NativeExpertSessionCache(model,self.mapping(),device='cpu',max_hot_experts=2,policy='fixed')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fixed_gpu_same_as_fixed_resident_and_no_forward_transfers(self):
        model=tiny().cuda().eval(); reference=copy.deepcopy(model)
        router=FixedExpertRouting(native_moes(reference),self.mapping()); router.select(['code']); router.attach()
        cache=NativeExpertSessionCache(model,self.mapping(),device='cuda',max_hot_experts=4,
                                       max_experts_per_layer=2,policy='fixed')
        try:
            cache.set_context_labels(['code']); before=cache.snapshot()
            x=torch.tensor([[1,2,3]],device='cuda')
            with torch.no_grad(): torch.testing.assert_close(model(x)[0],reference(x)[0])
            self.assertEqual(cache.snapshot()['host_to_device_bytes'],before['host_to_device_bytes'])
            self.assertEqual(cache.snapshot().get('misses',0),0)
        finally: cache.close(); router.close()

    def test_build_mapping_uses_train_mass_and_retains_N(self):
        mapping={'layers':{'1':{'a':[2,1,0],'b':[1,0,2]}},'provenance':{'source':'train'}}
        evidence=[{'layer':1,'expert':i,'label':label,'selection_rate':rate,'mean_routing_weight':.1}
                  for label in ('a','b') for i,rate in enumerate([.1,.9,1.])]
        fixed,global_fixed=fixed_mappings(mapping,evidence,2)
        self.assertEqual(fixed['layers']['1']['a'],[2,1])
        self.assertAlmostEqual(fixed['fixed_mass']['1']['a'],.3)
        self.assertEqual(global_fixed['layers']['1']['global'],[2,1])

    def test_paired_quality_comparison_uses_same_windows(self):
        rows=[{'mode':mode,'window':i,'tokens':4,'nll':nll+i}
              for mode,nll in [('a',1.),('b',1.5)] for i in range(3)]
        result=paired_contrast(rows,'a','b',123)
        self.assertEqual(result['delta_nll'],-.5)
        self.assertEqual(result['window_bootstrap_95'],[-.5,-.5])
        with self.assertRaises(ValueError): paired_contrast(rows[:-1],'a','b',123)


if __name__=='__main__': unittest.main()
