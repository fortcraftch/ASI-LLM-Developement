import unittest
from recovery_v2.eval.sandbox_probe import command


class SandboxCommandTests(unittest.TestCase):
    def test_confinement_and_no_host_mount(self):
        args = command('python@sha256:'+'a'*64, 'probe', 'pass')
        self.assertEqual(args[args.index('--network')+1], 'none')
        self.assertEqual(args[args.index('--cap-drop')+1], 'ALL')
        self.assertEqual(args[args.index('--user')+1], '65534:65534')
        self.assertIn('--read-only', args)
        self.assertIn('no-new-privileges', args)
        self.assertNotIn('-v', args)
        self.assertNotIn('--volume', args)
        self.assertNotIn('--privileged', args)
        self.assertEqual(args[-3:], ['-I','-c','pass'])


if __name__ == '__main__': unittest.main()
