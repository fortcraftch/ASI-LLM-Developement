import tempfile
import unittest
from recovery_v2.data.scan_checkpoint import ScanCheckpoint


class CheckpointTests(unittest.TestCase):
    def test_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScanCheckpoint(tmp, {'candidates':'abc'})
            result = {'status':'complete','excluded':['x']}
            store.write('one.arrow', 'sha', result)
            self.assertEqual(ScanCheckpoint(tmp, {'candidates':'abc'}).read('one.arrow','sha'), result)
            with self.assertRaises(ValueError): store.read('one.arrow','changed')
            with self.assertRaises(ValueError): ScanCheckpoint(tmp, {'candidates':'different'})
            with self.assertRaises(ValueError): store.write('two.arrow','sha',{'status':'failed'})
            self.assertIsNone(store.read('three.arrow','sha'))


if __name__ == '__main__': unittest.main()
