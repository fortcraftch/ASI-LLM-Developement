import sys
from pathlib import Path
import copy
import tempfile
import unittest
import numpy as np
import torch
from asi.models.domain import GPT, GPTConfig, Gate, MoE
from asi.runtime.cache import ExpertCacheManager
from asi.analysis.experts import RoutingTrace
from asi.experiments.train import TokenShardStream
from asi.runtime.generation import validate_pool_identity


def tiny():
    return GPT(GPTConfig(block_size=32, max_seq_len=32, original_seq_len=32,
                         max_batch_size=2, vocab_size=64, n_layer=3, n_head=2, n_embd=16,
                         inter_dim=32, moe_inter_dim=8, kv_lora_rank=8,
                         qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
                         n_routed_experts=4, n_pools=2, experts_per_pool=2, n_activated_experts=2))


class StreamTests(unittest.TestCase):
    def test_small_shards_wrap_overlap_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, values in enumerate(([0,1,2], [], [3,4], [5,6,7])):
                path = Path(tmp)/f'{i}.npy'
                np.save(path, np.array(values, dtype=np.uint16))
                paths.append(path)
            stream = TokenShardStream(paths, 2, 3)
            x,y = stream.next_batch()
            self.assertEqual(x.flatten().tolist(), [0,1,2,3,4,5])
            self.assertEqual(y.flatten().tolist(), [1,2,3,4,5,6])
            state = stream.state()
            second = stream.next_batch()
            self.assertEqual(second[0].flatten().tolist(), [6,7,0,1,2,3])
            stream.load_state(state)
            self.assertTrue(torch.equal(second[1],stream.next_batch()[1]))
            stream.tokens._mmap.close()


class RoutingTests(unittest.TestCase):
    def test_inactive_experts_unchanged_after_adamw_and_pool_switch(self):
        torch.manual_seed(4)
        model = tiny().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        tokens = torch.randint(0,64,(1,8))
        for pool in (0,1,0):
            optimizer.zero_grad(set_to_none=True)
            model.set_active_pools([pool])
            inactive = [p for layer in model.layers if isinstance(layer.ffn,MoE)
                        for eid,expert in enumerate(layer.ffn.experts) if eid//2 != pool
                        for p in expert.parameters()]
            before = [p.detach().clone() for p in inactive]
            _,loss=model(tokens,tokens)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in inactive))
            optimizer.step()
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,inactive)))

    def test_active_backend_matches_explicit_expert_sum(self):
        model=tiny().eval()
        moe=model.layers[1].ffn
        x=torch.randn(2,4,16)
        mask=model.build_expert_mask([1])
        with torch.no_grad():
            weights,indices=moe.gate(x.reshape(-1,16),mask)
            expected=moe.shared_experts(x.reshape(-1,16))
            for token in range(8):
                for j,eid in enumerate(indices[token].tolist()):
                    expected[token] += weights[token,j]*moe.experts[eid](x.reshape(-1,16)[token])
            actual=moe(x,mask)
        torch.testing.assert_close(actual,expected.reshape_as(actual))

    def test_sigmoid_bias_cannot_select_forbidden_experts(self):
        gate=Gate(GPTConfig(n_embd=16,n_routed_experts=4,n_activated_experts=2,score_func='sigmoid'))
        gate.bias=torch.nn.Parameter(torch.tensor([100.,100.,0.,0.]))
        _,ids=gate(torch.randn(8,16),torch.tensor([False,False,True,True]))
        self.assertTrue((ids>=2).all())

    def test_traces_identify_depth_and_no_leakage(self):
        model=tiny().eval()
        model.set_active_pools([1])
        with RoutingTrace(model,['a','b'],max_token_rows=2) as trace, torch.no_grad():
            model(torch.randint(0,64,(1,5)))
        report=trace.report()
        self.assertEqual(sum(s['mask_violations'] for s in report['layers'].values()),0)
        self.assertEqual(len(report['token_routes']),2)
        self.assertEqual(report['dropped_token_rows'],8)
        self.assertTrue(report['cross_layer_cooccurrence'])
        self.assertIsNone(model.layers[1].ffn.gate.routing_observer)

    def test_incremental_chunk_cache_matches_full_forward(self):
        model=tiny().eval()
        x=torch.randint(0,64,(1,8))
        with torch.no_grad():
            full=model(x)[0]
            model(x[:,:3])
            chunk=model(x[:,3:],start_pos=3)[0]
        torch.testing.assert_close(chunk,full[:,3:],rtol=1e-4,atol=1e-5)

    def test_manifest_order_is_identity(self):
        with self.assertRaises(ValueError):
            validate_pool_identity(tiny(),{'pool_names':['a','b']},['b','a'])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class CudaCacheTests(unittest.TestCase):
    def test_cuda_alias_pinning_eviction_and_logits(self):
        torch.manual_seed(7)
        model=tiny().eval()
        reference=copy.deepcopy(model).cuda().eval()
        cache=ExpertCacheManager('cuda',max_hot_pools=1,pin_memory=True)
        cache.initialize(model)
        x=torch.randint(0,64,(1,8),device='cuda')
        for pool in [0,0,1,1,0]:
            cache.prepare(model,[pool])
            model.set_active_pools([pool]); reference.set_active_pools([pool])
            with torch.no_grad():
                torch.testing.assert_close(model(x)[0],reference(x)[0])
            for row in cache.inventory()['experts']:
                self.assertEqual(row['device'],'cuda:0' if row['pool']==pool else 'cpu')
        self.assertEqual(cache.snapshot()['hits'],8)
        self.assertEqual(cache.snapshot()['misses'],12)
        self.assertEqual(cache.snapshot()['evictions'],8)
        self.assertGreater(cache.snapshot()['pinned_bytes'],0)
        cache.offload_all(model)
        self.assertTrue(all(r['device']=='cpu' for r in cache.inventory()['experts']))

from asi.analysis.experts import ExpertCalibrator

from asi.runtime.cache import NativeExpertSessionCache

class PretrainedAdapterTests(unittest.TestCase):
    def test_calibration_rates_and_multilabel_mapping(self):
        model=tiny().eval()
        with ExpertCalibrator(model) as calibration,torch.no_grad():
            calibration.set_labels(['math','code'])
            model(torch.randint(0,64,(1,8)))
        rows=calibration.report()['evidence']
        self.assertEqual(len(rows),16)
        for layer in (1,2):
            rate=sum(r['selection_rate'] for r in rows if r['layer']==layer and r['label']=='math')
            self.assertEqual(rate,2.0)
        self.assertFalse(calibration.propose_mapping(2)['reviewed'])
        self.assertTrue(calibration.propose_mapping(2)['layers']['1']['math'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_native_cache_preserves_routes_and_restricts_explicitly(self):
        model=tiny().cuda().eval()
        ref=copy.deepcopy(model)
        x=torch.randint(0,64,(1,6),device='cuda')
        mapping={'layers':{str(i):{'math':[2,3]} for i in (1,2)}}
        cache=NativeExpertSessionCache(model,mapping,max_hot_experts=4,policy='prefetch')
        cache.set_context_labels(['math'])
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0],ref(x)[0])
        self.assertLessEqual(len(cache.hot),4)
        self.assertGreater(cache.snapshot()['host_to_device_bytes'],0)
        cache.close()
        cache=NativeExpertSessionCache(model,mapping,max_hot_experts=4,policy='restrict')
        cache.set_context_labels(['math'])
        ref.set_active_pools([1])
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0],ref(x)[0])
        cache.close()

if __name__=='__main__':
    unittest.main()
