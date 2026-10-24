import copy,unittest
from asi.analysis.checkpoint_report import recover_native_suite
class RecoverSuiteTests(unittest.TestCase):
 def test_recovery_preserves_sources_and_rejects_incomplete_evidence(self):
  base=dict(kind='native_deepseek_benchmark',checkpoint_sha256='c',evaluator_sha256='e',precision='p',routing='r',runtime={'batch':4},parameters=1,limit=None,metadata={'architecture_sha256':'a'},datasets={'d':{'rows':2,'fingerprint':'f'}})
  a=dict(base,completed=False,tasks={'a':{'total':2,'accuracy':.5}})
  b=dict(base,completed=True,tasks={'b':{'total':2,'accuracy':.5}})
  ref=dict(base,completed=True,tasks={**a['tasks'],**b['tasks']})
  self.assertTrue(recover_native_suite([a,b],ref,['a','b'])['completed']);self.assertFalse(a['completed'])
  for field,value in [('runtime',{'batch':8}),('completed',False),('tasks',{'b':{'total':1,'accuracy':.5}}),('precision','other')]:
   bad=copy.deepcopy(b);bad[field]=value
   with self.assertRaises(ValueError):recover_native_suite([a,bad],ref,['a','b'])
if __name__=='__main__':unittest.main()
