import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from asi.analysis.reference_code_check import command, check_one, digest, prepare


class ReferenceChecks(unittest.TestCase):
    def test_no_mount_network_or_privileged_execution(self):
        argv = command('docker', 'sha256:'+'a'*64, 'asi-reference-test')
        for flag, value in [('--network','none'), ('--user','65534:65534'),
                            ('--memory','256m'), ('--memory-swap','256m'),
                            ('--cap-drop','ALL'), ('--pull','never')]:
            self.assertEqual(argv[argv.index(flag)+1], value)
        self.assertIn('--read-only', argv)
        self.assertNotIn('--privileged', argv)
        self.assertNotIn('--volume', argv)
        self.assertNotIn('--mount', argv)

    def test_assertion_failure_is_not_infrastructure_failure(self):
        with patch('subprocess.run', side_effect=[subprocess.CompletedProcess([],10),
                                                  subprocess.CompletedProcess([],0)]) as run:
            result = check_one('docker','image',dict(id='x',code='',tests=['assert False'],answer_sha256='a'),1)
        self.assertEqual(result['status'], 'assertion_failed')
        self.assertEqual(run.call_args_list[1].args[0][1:3], ['rm','-f'])
        self.assertEqual(run.call_args_list[0].kwargs['stdout'], subprocess.DEVNULL)

    def test_timeout_removes_container_and_checks_absence(self):
        with patch('subprocess.run', side_effect=[subprocess.TimeoutExpired('docker',1),
                subprocess.CompletedProcess([],1), subprocess.CompletedProcess([],0,stdout=b'')]) as run:
            result=check_one('docker','image',dict(id='x',answer_sha256='a'),1)
        self.assertEqual(result['status'],'timeout')
        self.assertEqual(run.call_count,3)

    def test_unverified_cleanup_stops(self):
        with patch('subprocess.run', side_effect=[subprocess.CompletedProcess([],0),
                subprocess.CompletedProcess([],1),subprocess.CompletedProcess([],0,stdout=b'live-id')]):
            with self.assertRaisesRegex(RuntimeError,'cleanup'):
                check_one('docker','image',dict(id='x',answer_sha256='a'),1)

    def test_changed_sources_and_duplicate_cases_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'corpus.json'; t=Path(tmp)/'cases.json'
            answer='```python\ndef f(): return 1\n```'
            c.write_text(json.dumps({'examples':[dict(id='x',answer=answer)]}))
            case=dict(id='x',answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                      tests=['assert f()==1'],purpose='fixture')
            plan=dict(corpus_sha256=digest(c),cases=[case])
            t.write_text(json.dumps(plan))
            self.assertEqual(len(prepare(c,t)),1)
            plan['cases']=[case,case]; t.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'Duplicate case'): prepare(c,t)
            c.write_text(c.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Corpus differs'): prepare(c,t)


import copy
from asi.analysis.reference_code_check import validate_execution_report

class ExecutionEvidenceChecks(unittest.TestCase):
    def test_reject_stale_incomplete_duplicate_and_failed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus=root/'corpus.json';cases=root/'cases.json';runner=root/'runner.py'
            import hashlib
            answer='```python\ndef f(): return 1\n```';ah=hashlib.sha256(answer.encode()).hexdigest()
            corpus.write_text(json.dumps({'examples':[{'id':'x','answer':answer}]}));runner.write_text('# pinned runner')
            cases.write_text(json.dumps({'corpus_sha256':digest(corpus),'cases':[{'id':'x','answer_sha256':ah,'tests':['assert f()==1'],'purpose':'fixture'}]}))
            report=dict(kind='curated_reference_assertions',completed=True,executed=True,corpus_sha256=digest(corpus),cases_sha256=digest(cases),runner_sha256=digest(runner),image_id='sha256:'+'a'*64,results=[dict(id='x',answer_sha256=ah,status='passed',exit_code=0)])
            self.assertEqual(validate_execution_report(report,corpus,cases,runner)['verified_cases'],1)
            bad=[]
            for key,value in [('completed',False),('executed',False),('runner_sha256','old'),('image_id','python:latest')]:
                r=copy.deepcopy(report);r[key]=value;bad.append(r)
            r=copy.deepcopy(report);r['results']*=2;bad.append(r)
            r=copy.deepcopy(report);r['results'][0]['exit_code']=10;bad.append(r)
            for r in bad:
                with self.assertRaises(ValueError):validate_execution_report(r,corpus,cases,runner)

if __name__ == '__main__': unittest.main()
