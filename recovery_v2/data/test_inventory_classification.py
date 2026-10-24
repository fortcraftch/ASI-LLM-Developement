import unittest
from recovery_v2.data.consolidate_inventory import classify, verify_scan
from recovery_v2.data.evaluation_sources import dataset_keys


class InventoryTests(unittest.TestCase):
    def test_download_not_evaluation(self):
        self.assertEqual(classify(['train.arrow'], {}), 'cached_overlap_unresolved')
        self.assertEqual(classify([], {}), 'no_cached_overlap_detected')

    def test_evaluation_dominates(self):
        self.assertEqual(classify(['train.arrow', 'test.arrow'], {'test.arrow':'recorded_evaluation_split'}),
                         'recorded_evaluation_overlap')

    def test_dataset_identity_parser(self):
        value = {'nested':[{'datasets':{"('openai/gsm8k', 'main', 'test')":{}, 'invalid':{}}}]}
        self.assertEqual(list(dataset_keys(value)), [('openai/gsm8k','main','test')])

    def test_incomplete_scan_rejected(self):
        with self.assertRaises(ValueError):
            verify_scan({'scan_finished':False}, None, None)
        with self.assertRaises(ValueError):
            verify_scan({'scan_finished':True,'files':[{'status':'failed'}]}, None, None)


if __name__ == '__main__': unittest.main()
