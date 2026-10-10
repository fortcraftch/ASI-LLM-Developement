import unittest
import torch
from asi.experiments.routing_paired import summarize


class PairedRoutingTests(unittest.TestCase):
    def test_exclusion_is_unrestricted_not_executed(self):
        p=torch.tensor([[.6,.3,.1],[.1,.3,.6]])
        ids=torch.tensor([[1],[2]])
        stats=summarize(ids,torch.tensor([[.3],[.6]]),p,[1,2],1)
        self.assertEqual(stats['outside_topk_fraction'],.5)
        self.assertAlmostEqual(stats['outside_topk_mass'],.3,places=6)
        self.assertEqual(stats['counts'],[0,1,1])
        self.assertAlmostEqual(stats['mean_executed_mass'],.45,places=6)


if __name__=='__main__':unittest.main()
