"""Sequential isolated reference validation; no GPU, no sealed test reader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
from recovery_v2.eval.sandbox_probe import command
from recovery_v2.preflight import save

LOADER = """import json,sys,unittest,importlib.util,multiprocessing
from pathlib import Path
row=json.load(sys.stdin)
module_path=Path.cwd()/'reference_under_test.py'
module_path.write_text(row['code']+'\\n'+row['reference'],encoding='utf-8')
sys.path.insert(0,str(Path.cwd()))
multiprocessing.set_start_method('spawn',force=True)
spec=importlib.util.spec_from_file_location('reference_under_test',module_path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.TestCases)
result=unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}))
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 1)
"""


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--sandbox-report',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--timeout',type=float,default=30.)
    args=p.parse_args();data=Path(args.data);out=Path(args.output)
    if args.timeout <= 0: raise SystemExit('Positive timeout required')
    out.mkdir(parents=True,exist_ok=True)
    sandbox=json.loads(Path(args.sandbox_report).read_text())
    if sandbox.get('passed') is not True: raise SystemExit('Sandbox probe not passed')
    image=sandbox['image'];payload=data.read_bytes()
    identity={'source_sha256':hashlib.sha256(payload).hexdigest(),'image':image,'timeout_seconds':args.timeout,
              'sandbox_report_sha256':hashlib.sha256(Path(args.sandbox_report).read_bytes()).hexdigest(),
              'sandbox_command_sha256':hashlib.sha256(json.dumps(command(image,'IDENTITY','CODE')).encode()).hexdigest(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=out/'report.json'
    report={'status':'partial','identity':identity,'results':[]}
    if path.exists():
        if not args.resume: raise SystemExit('Use --resume; previous results preserved')
        report=json.loads(path.read_text())
        if report['identity']!=identity: raise SystemExit('Resume identity mismatch')
    done={r['id'] for r in report['results']}
    rows=json.loads(payload)['examples']
    if len({r['id'] for r in rows})!=len(rows):raise SystemExit('Duplicate task IDs')
    for row in rows:
        if row['id'] in done:continue
        save(out/'STATUS.json',{'status':'partial','completed':len(done),'total':len(rows),'task':row['id']})
        name='asi-v2-reference-'+uuid.uuid4().hex
        cmd=command(image,name,LOADER);cmd.insert(2,'-i')
        try:
            run=subprocess.run(cmd,input=json.dumps(row['row']),text=True,capture_output=True,timeout=args.timeout)
            item={'id':row['id'],'returncode':run.returncode,'passed':run.returncode==0,
                  'stdout':run.stdout,'stderr':run.stderr}
        except subprocess.TimeoutExpired:
            item={'id':row['id'],'passed':False,'timeout':True}
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
        report['results'].append(item);done.add(row['id']);save(path,report)
    report['status']='complete';report['passed']=sum(r['passed'] for r in report['results'])
    report['scope']='Reference validation only; task failures are retained, not model errors.'
    save(path,report);save(out/'STATUS.json',{'status':'complete','completed':len(done),'total':len(rows)})


if __name__=='__main__':main()
