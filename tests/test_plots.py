import copy
import unittest

from asi.analysis.plots import event_metrics, memory_metrics, route_changes


class PlotMetricsTests(unittest.TestCase):
    def event(self, pair, phase='decode', misses=0, seconds=.1):
        return {'phase':phase,'seconds':seconds,'routes':{'1':[pair]},
                'cache_delta':{'misses':misses,'evictions':misses,'host_to_device_bytes':misses*1024**2}}

    def test_pair_order_is_not_an_expert_change(self):
        rows=[{'events':[self.event([1,2],'prefill'),self.event([2,1]),self.event([1,3])]}]
        self.assertEqual(route_changes(rows), (1,2))
        self.assertEqual(route_changes(rows+rows), (2,4))

    def test_transfer_and_route_change_are_separate(self):
        row={'events':[self.event([1,2],'prefill'),self.event([1,3]),self.event([2,3])],
             'prefetch_delta':{'host_to_device_bytes':3*1024**2}}
        metrics=event_metrics([row])
        self.assertEqual(metrics['pair_changes'],2)
        self.assertEqual(metrics['loads'],0)
        self.assertEqual(metrics['no_load_percent'],100)
        self.assertEqual(metrics['prepare_mib'],3)
        self.assertAlmostEqual(metrics['steps_per_second'],10)

    def test_missing_decode_is_not_zero_cost_evidence(self):
        result=event_metrics([{'events':[self.event([1,2],'prefill')],
                               'prefetch_delta':{'host_to_device_bytes':0}}])
        self.assertIsNone(result['no_load_percent'])
        self.assertIsNone(result['steps_per_second'])
        self.assertIsNone(result['h2d_mib_per_step'])

    def test_ram_backing_not_doubled_and_missing_not_zero(self):
        experts=[{'device':'cuda:0','weight_bytes':1024**2,'ram_backing_bytes':1024**2},
                 {'device':'cpu','weight_bytes':1024**2,'ram_backing_bytes':1024**2}]
        result=memory_metrics({'inventory':{'experts':experts}})
        self.assertEqual(result['gpu_experts'],1)
        self.assertEqual(result['gpu_expert_mib'],1)
        self.assertEqual(result['ram_expert_mib'],2)
        self.assertIsNone(result['cuda_allocated_mib'])
        self.assertIsNone(memory_metrics({})['gpu_experts'])
        missing=copy.deepcopy(experts)
        for expert in missing: del expert['ram_backing_bytes']
        self.assertIsNone(memory_metrics({'experts':missing})['ram_expert_mib'])


if __name__=='__main__': unittest.main()
