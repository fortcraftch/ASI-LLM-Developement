import unittest

from asi.experiments.recovery_generation import checkpoint_identity, counter_delta
from asi.taxonomy import digest


class RecoveryGenerationIdentityTest(unittest.TestCase):
    def test_full_bank_preserved_for_16_inference(self):
        corpus={'examples': []}
        study={'mappings': {'32': {'category': 'wide'}, '16': {'category': 'narrow'}}}
        state={'recovery_settings': {'mask_sizes': [32,16]}, 'identity': {
            'source': 'source', 'corpus': digest(corpus), 'mapping': digest(study['mappings']['32']),
            'experts': 32, 'active': 8, 'policy': 'restricted',
            'objective': 'assistant_ce_native_kl_eos_v3', 'mask_bank': digest(study['mappings']),
            'evaluation_subset': digest(['a','b'])}}
        self.assertEqual(checkpoint_identity('source',corpus,study,state,['b','a']),study['mappings'])
        study['mappings']['16']={'category': 'changed'}
        with self.assertRaises(ValueError):
            checkpoint_identity('source',corpus,study,state,['a','b'])

    def test_counter_delta_separates_phases(self):
        before={'loads': 5, 'host_to_device_bytes': 100, 'disk_ram': {'logical_read_bytes': 200}}
        after={'loads': 5, 'host_to_device_bytes': 100, 'disk_ram': {'logical_read_bytes': 200}}
        self.assertEqual(counter_delta(before,after)['loads'],0)
        after['loads']=8
        after['disk_ram']['logical_read_bytes']=250
        difference=counter_delta(before,after)
        self.assertEqual(difference['loads'],3)
        self.assertEqual(difference['logical_read_bytes'],50)


if __name__=='__main__':
    unittest.main()
