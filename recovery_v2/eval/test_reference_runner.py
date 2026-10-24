"""Trusted fixtures only; dataset code is never run in these host tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from recovery_v2.eval.reference_runner import LOADER


class LoaderTests(unittest.TestCase):
    def test_importable_and_spawn_picklable(self):
        row={'code':'def add_one(x): return x+1', 'reference':'''import unittest,importlib,multiprocessing
class TestCases(unittest.TestCase):
 def test_import(self):
  self.assertIs(importlib.import_module(__name__).add_one,add_one)
 def test_spawn(self):
  with multiprocessing.get_context('spawn').Pool(1) as pool:
   self.assertEqual(pool.map(add_one,[1]),[2])
'''}
        with tempfile.TemporaryDirectory() as tmp:
            run=subprocess.run([sys.executable,'-I','-c',LOADER],input=json.dumps(row),
                               text=True,capture_output=True,cwd=tmp,timeout=25)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout.splitlines()[-1])['tests_run'],2)


if __name__=='__main__':unittest.main()
