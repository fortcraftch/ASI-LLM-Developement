"""Probe a local Docker sandbox before executing any dataset code."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import re
from recovery_v2.preflight import save


def command(image, name, code):
    return ['docker','run','--rm','--name',name,'--network','none','--read-only',
            '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','32',
            '--memory','1g','--cpus','1','--user','65534:65534',
            '--tmpfs','/work:rw,noexec,nosuid,size=128m,mode=1777','--workdir','/work',
            '--env','HOME=/work','--env','TMPDIR=/work','--env','MPLCONFIGDIR=/work/matplotlib',
            '--env','MPLBACKEND=Agg','--env','OPENBLAS_NUM_THREADS=1',
            '--entrypoint','python',image,'-I','-c',code]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Existing local image repository@sha256:digest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', args.image):
        raise SystemExit('Pin image by SHA-256 digest before probing.')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists(): raise SystemExit('Existing report preserved; choose a new attempt directory.')
    if not shutil.which('docker'): raise SystemExit('Docker unavailable; sandbox gate remains pending.')
    inspection = subprocess.run(['docker','image','inspect',args.image],capture_output=True,text=True,timeout=20)
    if inspection.returncode: raise SystemExit('Pinned image is not available locally; no automatic pull.')
    probes = {
        'filesystem': "from pathlib import Path\nPath('/work/allowed').write_text('ok')\ntry:\n Path('/etc/v2_probe').write_text('bad')\nexcept OSError:\n print('BLOCKED')\nelse:\n raise SystemExit(9)",
        'network': "import socket\ns=socket.socket();s.settimeout(2)\ntry:\n s.connect(('1.1.1.1',443))\nexcept OSError:\n print('BLOCKED')\nelse:\n raise SystemExit(9)",
        'timeout': "print('STARTED',flush=True)\nwhile True: pass",
    }
    results = {}
    for key, code in probes.items():
        name = 'asi-v2-probe-'+uuid.uuid4().hex
        save(out/'STATUS.json',{'status':'partial','stage':key})
        try:
            run = subprocess.run(command(args.image,name,code),capture_output=True,text=True,timeout=10)
            results[key] = {'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,
                            'passed':key != 'timeout' and run.returncode == 0 and run.stdout.strip() == 'BLOCKED'}
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b''
            if isinstance(output, bytes): output = output.decode('utf-8',errors='replace')
            results[key] = {'timed_out':True,'passed':key == 'timeout' and 'STARTED' in output}
        finally:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
        # Confirm the workload did not survive the timeout/cleanup.
        check = subprocess.run(['docker','ps','-a','--filter','name='+name,'--format','{{.Names}}'],
                               capture_output=True,text=True,timeout=20)
        if check.returncode != 0 or name in check.stdout: results[key]['passed'] = False
    passed = all(r['passed'] for r in results.values())
    save(out/'report.json',{'status':'complete' if passed else 'failed','passed':passed,
                            'image':args.image,'image_inspection':json.loads(inspection.stdout),
                            'probes':results,'scope':'Sandbox smoke checks only, not reference validation.'})
    save(out/'STATUS.json',{'status':'complete' if passed else 'failed','stage':'finished'})


if __name__ == '__main__': main()
