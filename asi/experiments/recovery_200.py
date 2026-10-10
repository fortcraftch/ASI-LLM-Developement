"""Paired 200-update CE versus CE+KL pilot, queued behind the memory experiment."""
import argparse
import ast
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from asi.experiments.recovery_validation import read, fingerprint, CATEGORIES


def screen(row):
    """Structural reference screening, not a claim of semantic verification."""
    text=row['answer']
    exclusions={'gsm8k_train_4105':'reference uses Tuesday instead of Monday; total contradicts prompt',
                'mbpp_test_101':'undefined cylinder perimeter; reference computes rectangle perimeter',
                'sciq_test_578':'malformed respiration cofactor name in prompt and reference',
                'sciq_train_5022':'astronomy example mislabeled as biology'}
    if row['id'] in exclusions:return exclusions[row['id']]
    if '\ufffd' in text or '\ufffd' in row['prompt']:return 'replacement character'
    if row.get('answer_tokens',999)>160:return 'long reference'
    if row.get('evaluation')=='python_syntax_and_manual_tests':
        match=re.fullmatch(r'\s*```python\s*\n(.*?)\n```\s*',text,re.S)
        if not match:return 'reference code format'
        try:ast.parse(match.group(1))
        except (SyntaxError,ValueError):return 'reference syntax'
        if not row.get('reference_tests'):return 'missing reference tests'
    if 'expected_number' in row:
        endings=re.findall(r'Final answer:\s*([^\n]+)',text)
        if endings!=[row['expected_number']]:return 'numeric reference mismatch'
    if 'expected_phrase' in row and row['expected_phrase'].casefold() not in text.casefold():return 'missing answer phrase'
    return None


def prepare(root):
    from transformers import AutoTokenizer
    from asi.experiments.recovery import read_examples, supervised_rows, select_examples
    corpus_path=Path('results/recovery_corpus_v3/corpus.json')
    corpus=read_examples(corpus_path)
    tokenizer=AutoTokenizer.from_pretrained('public_models/olmoe',local_files_only=True,trust_remote_code=False)
    excluded=set(read('results/recovery_quick_20261006/ids.json'))
    excluded.update(read('results/recovery_validation_20261008/prompt_ids.json'))
    rng=random.Random(20261009);ids=[];rejected={}
    for split,count in [('train',80),('dev',6),('test',6)]:
        for category in CATEGORIES:
            candidates=sorted([r for r in corpus['examples'] if r['split']==split and r['category']==category and r['id'] not in excluded],key=lambda r:r['id'])
            rng.shuffle(candidates);chosen=[]
            for row in candidates:
                reason=screen(row)
                if reason is None:
                    try:supervised_rows({'examples':[row]},tokenizer,384)
                    except ValueError:reason='tokenized length exceeds 384'
                if reason:rejected[row['id']]=reason;continue
                chosen.append(row['id'])
                if len(chosen)==count:break
            if len(chosen)!=count:raise ValueError('Not enough screened references: '+split+' '+category)
            ids+=chosen
    selected=select_examples(corpus,ids)
    root.mkdir(parents=True,exist_ok=False)
    (root/'ids.json').write_text(json.dumps(ids,indent=2),encoding='utf8')
    common=['--examples',str(corpus_path),'--example-ids',str(root/'ids.json'),
            '--study','results/recovery_calibration_batched_v3/report.json','--model-dir','public_models/olmoe',
            '--teacher-dir',str(root/'teacher'),'--teacher-eos','--max-length','384',
            '--new-tokens','256','--generation-per-category','6','--ram-gib','12','--expert-ram-gib','2',
            '--mask-sizes','32','--experts','32','--active','8','--seed','1337','--reference-batch-size','1']
    train=['--adaptation','router-expert-lora','--rank','8','--steps','200','--eval-interval','50',
           '--paired-example-order','--train-mask','32','--generation-selection','--dev-generation-per-category','2']
    jobs=[dict(name='teacher',argv=['prepare',*common,'--output',str(root/'prepare')]),
          dict(name='ce',argv=['train',*common,*train,'--supervised','--output',str(root/'ce')]),
          dict(name='ce_kl',argv=['train',*common,*train,'--alpha','0.5','--output',str(root/'ce_kl')]),
          dict(name='native',argv=['native',*common,'--output',str(root/'native')])]
    sources=[corpus_path,Path('results/recovery_calibration_batched_v3/report.json'),root/'ids.json']
    plan=dict(version=1,steps=200,seed=1337,counts={'train':240,'dev':18,'test':18},jobs=jobs,
              source_hashes={str(p):fingerprint(p) for p in sources},
              dependency='results/recovery_memory_comparison_20261008.json',
              screening='Structural checks and exact token lengths only; not full semantic reference verification.',
              excluded_ids=sorted(excluded),rejected_candidates=rejected,
              selection='Dev-only numeric error, worst-category repetition, format error, relative CE (lexicographic). Human semantic review still required.',
              controls='Same original initialization, rank, resident mask, active count, seed, order, 200 steps and data; CE weight 1 vs 0.5 CE + 0.5 KL.',
              test_policy='Test used only after selecting each checkpoint. No picking between objectives using test.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    sample=[]
    for category in CATEGORIES:
        sample.extend([r for r in selected['examples'] if r['split']=='train' and r['category']==category][:6])
    (root/'reference_review_sample.md').write_text('\n\n'.join('## '+r['id']+'\n'+r['prompt']+'\n\n'+r['answer'] for r in sample),encoding='utf8')
    print('Prepared 240 train / 18 dev / 18 test; structural screening, semantic review not certified',flush=True)


def resume_teacher_command(root, argv):
    """Preserve interrupted reports/logs; native preparation validates saved shards."""
    if not (root/'teacher/manifest.json').is_file():
        raise ValueError('Cannot resume without a teacher manifest')
    attempt=1
    while (root/f'prepare_resume_{attempt:03d}').exists() or (root/f'teacher_resume_{attempt:03d}.log').exists():
        attempt+=1
    command=list(argv)
    command[command.index('--output')+1]=str(root/f'prepare_resume_{attempt:03d}')
    command.append('--resume-teacher')
    return command,root/f'teacher_resume_{attempt:03d}.log'


def run(root,wait,resume_teacher=False):
    if (root/'SUPERSEDED.txt').exists():
        raise ValueError('This draft plan was superseded; read SUPERSEDED.txt')
    plan=read(root/'plan.json')
    def status(state,stage=None):
        pending=root/'status.tmp'
        pending.write_text(json.dumps({'state':state,'stage':stage,'plan_sha256':fingerprint(root/'plan.json')},indent=2),encoding='utf8')
        pending.replace(root/'status.json')
    for path,expected in plan['source_hashes'].items():
        if fingerprint(path)!=expected:raise ValueError('Frozen source changed: '+path)
    deadline=time.monotonic()+wait
    dependency=Path(plan['dependency'])
    status('waiting_for_memory_comparison')
    print('Waiting for completed memory comparison; no concurrent GPU jobs',flush=True)
    while True:
        try:
            if dependency.exists() and read(dependency).get('completed'):break
        except json.JSONDecodeError:
            pass  # The other process may still be publishing its JSON.
        if time.monotonic()>=deadline:raise RuntimeError('Memory comparison not complete; pilot has not started')
        time.sleep(min(30,max(0,deadline-time.monotonic())))
    for job in plan['jobs']:
        argv=list(job['argv'])
        log_path=root/(job['name']+'.log')
        out=Path(job['argv'][job['argv'].index('--output')+1])
        if out.exists():
            r=read(out/'report.json') if (out/'report.json').exists() else {}
            done=r.get('subset_completed') if job['name']=='teacher' else r.get('completed')
            if done:continue
            if job['name']=='teacher' and resume_teacher:
                argv,log_path=resume_teacher_command(root,argv)
            else:
                raise ValueError('Partial stage preserved; inspect '+str(out))
        print('Running '+job['name'],flush=True)
        status('running',job['name'])
        with log_path.open('x',encoding='utf8') as log:
            subprocess.run([sys.executable,'-u','-m','asi','recovery',*argv],stdout=log,stderr=subprocess.STDOUT,check=True)
    a,b=read(root/'ce/report.json'),read(root/'ce_kl/report.json')
    if [(r['id'],r['mask']) for r in a['steps']]!=[(r['id'],r['mask']) for r in b['steps']]:
        raise ValueError('Unpaired training exposure')
    import torch
    initial=[torch.load(root/arm/'adaptation_step_0000.pt',map_location='cpu',weights_only=True) for arm in ('ce','ce_kl')]
    for field in ('routers','adapters'):
        if initial[0][field].keys()!=initial[1][field].keys() or any(not torch.equal(initial[0][field][k],initial[1][field][k]) for k in initial[0][field]):
            raise ValueError('Training initializations differ')
    result={'completed':True,'arms':{},'limitations':['Single seed; automatic selection proxies do not establish semantic recovery.']}
    for name,r in [('ce',a),('ce_kl',b)]:
        result['arms'][name]={'selected_step':r['selected_step'],'trainable_scope':r['trainable_scope'],
                             'dev_selection_key':r['dev_best_generation_key'],
                             'test_ce':sum(x['ce'] for x in r['test_after']['records'])/len(r['test_after']['records']),
                             'report_sha256':fingerprint(root/name/'report.json')}
    (root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    status('completed')
    print('Paired pilot completed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=0)
    p.add_argument('--resume-teacher-stage',action='store_true',help='Verify/reuse saved teacher shards; preserve interrupted reports')
    a=p.parse_args()
    if a.wait_seconds<0:p.error('Negative wait')
    if a.action=='prepare':prepare(a.output)
    else:run(a.output,a.wait_seconds,a.resume_teacher_stage)


if __name__=='__main__':main()
