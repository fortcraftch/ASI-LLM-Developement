from types import SimpleNamespace
import copy
import unittest
import torch
from asi.experiments.local_generation import timed_generate
from asi.analysis.local_generation_report import summarize


class FakeModel(torch.nn.Module):
    config=SimpleNamespace(block_size=8)
    def forward(self,x):
        out=torch.zeros(1,x.shape[1],6)
        out[:,:,-1]=100  # padded vocabulary must never be generated
        out[:,:,3]=10   # real EOS
        return out,None


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.enc=SimpleNamespace(encode=lambda *a,**k:[1,2], decode=lambda ids:str(ids),n_vocab=4,eot_token=3)

    def test_eos_stops_quality_but_not_equal_work_timing(self):
        a=timed_generate(FakeModel(),self.enc,'prompt','cpu',4)
        b=timed_generate(FakeModel(),self.enc,'prompt','cpu',4,stop_at_eos=False)
        self.assertEqual(a['output_ids'],[3]);self.assertEqual(a['stop_reason'],'eos')
        self.assertEqual(b['output_ids'],[3]*4);self.assertEqual(b['stop_reason'],'token_limit')

    def test_refuses_unmatched_context_truncation(self):
        with self.assertRaises(ValueError):timed_generate(FakeModel(),self.enc,'prompt','cpu',7)

    def test_incomplete_report_is_not_evidence(self):
        with self.assertRaises(ValueError):summarize({'completed':False})

    def test_rejects_variable_work(self):
        data={'completed':True,'kind':'local_124m_generation','probes':[{'prompt':'x'}],
              'settings':{'repeats':1,'new_tokens':4},'models':[{'runs':[{'prompt':'x','repeat':0,'input_ids':[1],
              'stop_at_eos':True,'generated_tokens':1}]}]}
        with self.assertRaisesRegex(ValueError,'identical fixed'):summarize(data)

    def test_checks_tokens_and_separates_disk_from_gpu(self):
        run={'prompt':'x','repeat':0,'input_ids':[1],'stop_at_eos':False,'generated_tokens':4,
             'tokens_per_second':2.,'first_token_seconds':.25,'peak_cuda_allocated_bytes':8*2**20,
             'memory':{'weight_bytes_by_device':{'cpu':3*2**20,'cuda:0':2*2**20}},
             'loads_during_generation':0,'h2d_bytes_during_generation':0}
        model={'variant':'deepseek','parameters':100,'parameter_bytes':5*2**20,
               'checkpoint_file_bytes':2**30,'runs':[run]}
        data={'completed':True,'kind':'local_124m_generation','probes':[{'prompt':'x'}],
              'settings':{'repeats':1,'new_tokens':4},'models':[model,copy.deepcopy(model)]}
        r=summarize(data)[0]
        self.assertEqual(r['gpu_weight_mib'],2)
        self.assertEqual(r['weight_mib'],5)
        self.assertEqual(r['checkpoint_gib'],1)
        self.assertEqual(r['median_first_token_ms'],250)
        data['models'][1]['runs'][0]['input_ids']=[2]
        with self.assertRaisesRegex(ValueError,'tokenized inputs'):summarize(data)


if __name__=='__main__':unittest.main()
