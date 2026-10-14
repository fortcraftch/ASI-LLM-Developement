"""Run curated reference-code assertions in a local, restricted Linux container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid


WORKER = '''import json,sys
p=json.load(sys.stdin)
try:
    for test in p['tests']:
        ns={}
        exec(compile(p['code'],'reference','exec'),ns)
        exec(compile(test,'assertion','exec'),ns)
except AssertionError:
    sys.exit(10)
except BaseException:
    sys.exit(11)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(corpus_path, cases_path):
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    if cases['corpus_sha256'] != digest(corpus_path):
        raise ValueError('Corpus differs from the pinned test cases')
    by_id = {r['id']: r for r in corpus['examples']}
    if len(by_id) != len(corpus['examples']):
        raise ValueError('Duplicate corpus IDs')
    jobs, seen = [], set()
    for case in cases['cases']:
        if case['id'] in seen:
            raise ValueError('Duplicate case ID')
        seen.add(case['id'])
        row = by_id[case['id']]
        if hashlib.sha256(row['answer'].encode()).hexdigest() != case['answer_sha256']:
            raise ValueError('Reference answer changed: '+case['id'])
        match = re.fullmatch(r'\s*```python\s*\n(.*?)\n```\s*', row['answer'], re.S)
        if not match or not case['tests'] or not all(isinstance(t, str) and t.strip() for t in case['tests']):
            raise ValueError('Expected fenced code and nonempty tests')
        jobs.append(dict(id=case['id'], code=match.group(1), tests=case['tests'],
                         purpose=case['purpose'], answer_sha256=case['answer_sha256']))
    if not jobs:
        raise ValueError('Empty test cases')
    return jobs


def command(docker, image_id, name):
    return [docker, 'run', '--rm', '--pull', 'never', '--name', name,
            '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--user', '65534:65534',
            '--memory', '256m', '--memory-swap', '256m', '--cpus', '1',
            '--pids-limit', '32', '--log-driver', 'none',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
            '--ulimit', 'fsize=65536:65536', '-i', '--entrypoint', 'python',
            image_id, '-I', '-S', '-c', WORKER]


def check_one(docker, image_id, job, timeout):
    name = 'asi-reference-'+uuid.uuid4().hex
    started = time.monotonic()
    try:
        # Discard all candidate output: a loop printing text cannot fill host RAM/disk.
        result = subprocess.run(command(docker, image_id, name),
                                input=json.dumps(job).encode(), stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        status = {0: 'passed', 10: 'assertion_failed', 11: 'reference_or_test_error',
                  137: 'resource_or_external_kill'}.get(result.returncode, 'container_error')
        code = result.returncode
    except subprocess.TimeoutExpired:
        status, code = 'timeout', None
    finally:
        # Killing the CLI alone does not prove the container stopped. Remove only our UUID name.
        cleanup = subprocess.run([docker, 'rm', '-f', name], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=15, check=False)
        if cleanup.returncode:
            probe = subprocess.run([docker, 'container', 'ls', '-aq', '--filter', 'name=^/'+name+'$'],
                                   capture_output=True, timeout=15, check=False)
            if probe.returncode or probe.stdout.strip():
                raise RuntimeError('Could not verify cleanup of '+name)
    return dict(id=job['id'], status=status, exit_code=code,
                seconds=time.monotonic()-started, answer_sha256=job['answer_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('results/recovery_corpus_v3/corpus.json'))
    parser.add_argument('--cases', type=Path, default=Path('configs/reference_code_cases_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', default='python:3.11-slim')
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.timeout <= 0:
        parser.error('Use a new output file and a positive timeout')
    jobs = prepare(args.corpus, args.cases)
    report = dict(schema=1, kind='curated_reference_assertions', completed=False,
                  executed=False, corpus_sha256=digest(args.corpus), cases_sha256=digest(args.cases),
                  runner_sha256=digest(Path(__file__)), timeout=args.timeout, results=[],
                  limitations=['Curated diagnostic cases, not a benchmark score or exhaustive correctness proof.',
                               'Container isolation is not an adversarial evaluator; curated references must not forge exit status.',
                               'Timeout includes container startup; it is not model inference latency.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix(args.output.suffix+'.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        tmp.replace(args.output)
    if args.dry_run:
        report['prepared_ids'] = [j['id'] for j in jobs]
        save()
        print(f'Prepared {len(jobs)} references; none executed')
        return
    docker = shutil.which('docker')
    if not docker:
        report['error'] = 'Docker executable unavailable; no host execution fallback'
        save()
        raise SystemExit(report['error'])
    try:
        info = subprocess.run([docker, 'image', 'inspect', args.image], capture_output=True,
                              timeout=15, check=True)
        image = json.loads(info.stdout)[0]
        image_id = image['Id']
        if image.get('Os') != 'linux' or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise ValueError('Requires an existing Linux image with immutable image ID')
        report['image_id'] = image_id
        save()
        for job in jobs:
            result = check_one(docker, image_id, job, args.timeout)
            report['executed'] = True
            report['results'].append(result)
            save()
            if result['status'] == 'container_error':
                raise RuntimeError('Container failed; not counted as a reference failure')
        report['completed'] = True
    except Exception as error:
        report['error'] = str(error)
        save()
        raise
    save()
    print(json.dumps(report['results'], indent=2))
    if any(r['status'] != 'passed' for r in report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
