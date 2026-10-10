import copy
import unittest
from asi.analysis.recovery_hardware import build_profiles,choose_profile


class RecoveryHardwareTests(unittest.TestCase):
    def fixtures(self):
        train={'action':'train','completed':True,'selected_step':0,'identity':{'model':'x'},
               'dev_before':{'records':[{'mask':'16','category':'a','kl':.3},{'mask':'16','category':'b','kl':.5}]}}
        evaluation={'action':'evaluate','completed':True,'selected_step':0,'identity':train['identity'],
            'checkpoint_sha256':'hash','settings':{'expert_ram_gib':.25,'new_tokens':32},
            'hardware':{'gpu':{'name':'GPU'}},'rss_bytes':123,
            'generations':[{'id':'test','mask':'16','stage':'after','token_ids':[1,2],
                'memory_scope':'standalone evaluation process','peak_cuda_allocated_bytes':4*2**30,
                'prepare_seconds':1.,'generation_seconds':2.,
                'cache_before':{'loads':4,'host_to_device_bytes':100},
                'cache_after':{'loads':4,'host_to_device_bytes':100}}]}
        return train,evaluation

    def test_selection_uses_dev_and_rejects_memory_or_quality_shortfall(self):
        train,evaluation=self.fixtures();report=build_profiles(train,[evaluation]);profiles=report['profiles']
        self.assertEqual(profiles[0]['worst_category_dev_kl'],.5)
        self.assertEqual(choose_profile(profiles,6*2**30,2**30,.6,2**29)['status'],'estimated_candidate')
        for gpu,ram,quality in [(4*2**30,2**30,.6),(6*2**30,1,.6),(6*2**30,2**30,.2)]:
            self.assertEqual(choose_profile(profiles,gpu,ram,quality,2**29)['status'],'blocked')
        with self.assertRaises(ValueError):choose_profile(profiles,6*2**30,2**30,float('nan'),0)

    def test_reject_training_memory_changed_checkpoint_and_changed_outputs(self):
        train,evaluation=self.fixtures()
        for key,value in [('action','train'),('identity',{}),('selected_step',1)]:
            changed=copy.deepcopy(evaluation);changed[key]=value
            with self.assertRaises(ValueError):build_profiles(train,[changed])
        changed=copy.deepcopy(evaluation);changed['checkpoint_sha256']='other'
        with self.assertRaises(ValueError):build_profiles(train,[evaluation,changed])
        changed=copy.deepcopy(evaluation);changed['generations'][0]['token_ids']=[3]
        with self.assertRaises(ValueError):build_profiles(train,[evaluation,changed])


if __name__=='__main__':unittest.main()
