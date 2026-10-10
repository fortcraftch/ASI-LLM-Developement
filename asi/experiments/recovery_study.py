"""Frozen multi-seed recovery protocol and blinded human review export."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import statistics
import copy
import time

from asi.taxonomy import digest


def run_pilot(root, wait_seconds=0):
    """Run an explicit local pilot queue after its subset references are ready."""
    if wait_seconds<0: raise ValueError('Negative wait budget')
    jobs=json.loads((root/'jobs.json').read_text(encoding='utf8'))
    if len(jobs)!=4 or [j['argv'][:2] for j in jobs]!=[['recovery','native']]+[['recovery','train']]*3:
        raise ValueError('Pilot requires native then three training jobs')
    reports=[]
    for job in jobs:
        argv=job['argv']
        if '--example-ids' not in argv: raise ValueError('Explicit pilot IDs required')
        output=Path(argv[argv.index('--output')+1])
        if output.exists(): raise ValueError('Pilot output already exists: '+str(output))
        reports.append(output/'report.json')
    deadline=time.monotonic()+wait_seconds
    while True:
        path=root/'prepare'/'report.json'
        prep=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        if prep.get('subset_completed'): break
        if time.monotonic()>=deadline: raise ValueError('Pilot preparation not complete; inspect prepare.log')
        time.sleep(min(10,max(0,deadline-time.monotonic())))
    expected=json.loads((root/'ids.json').read_text(encoding='utf8'))
    if prep.get('identity',{}).get('evaluation_subset')!=digest(sorted(expected)):
        raise ValueError('Prepared subset identity differs')
    for job in jobs:
        argv=job['argv']
        if json.loads(Path(argv[argv.index('--example-ids')+1]).read_text(encoding='utf8'))!=expected:
            raise ValueError('Job subset differs from prepared IDs')
        print('Pilot stage:',job['name'],flush=True)
        with (root/(job['name']+'.log')).open('x',encoding='utf8') as log:
            subprocess.run([sys.executable,'-u','-m','asi',*argv],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=summarize(reports)
    threshold=json.loads((root/'protocol.json').read_text(encoding='utf8')).get('dev_screen',{}).get('threshold',.95) if (root/'protocol.json').exists() else .95
    result['pilot_screen']={'threshold':threshold,'based_on':'worst relative dev objective, including step zero',
                            'manual_review_required':True,'models':[]}
    for path in reports[1:]:
        r=json.loads(path.read_text(encoding='utf8'))
        ratio=r.get('dev_best_worst_group_ratio')
        result['pilot_screen']['models'].append({'adaptation':r['settings']['adaptation'],
            'dev_ratio':ratio,'passes_screen':ratio is not None and ratio<=threshold})
    (root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    corpus=json.loads(Path(jobs[0]['argv'][jobs[0]['argv'].index('--examples')+1]).read_text(encoding='utf8'))
    export_review(corpus,reports,root/'review')
    print('Pilot complete:',root/'comparison.json',flush=True)


def curriculum_plan(source, output):
    """Extend a frozen plan without moving its teacher or previous run outputs."""
    plan=copy.deepcopy(source)
    if 'curriculum_groups' in plan:
        raise ValueError('Plan already contains curriculum jobs')
    prerequisites=[j['name'] for j in plan['jobs'] if j['argv'][1]=='train']
    if not prerequisites:
        raise ValueError('Complete 32-resident protocol required')
    groups=[]
    for seed in plan['seeds']:
        template=next(j for j in source['jobs'] if j['name']==f'router-expert-lora_{seed}')
        base=template['argv']
        epoch_steps=int(base[base.index('--steps')+1])//plan['epochs']
        group=[]
        for policy in ('direct16','mixed','gradual'):
            name=f'curriculum_{policy}_{seed}'
            argv=list(base)
            argv[argv.index('--steps')+1]=str(epoch_steps*3)
            pos=argv.index('--mask-sizes')
            argv[pos+1:pos+2]=['32','16']
            argv[-1]=(output/name).as_posix()
            argv+=['--paired-example-order']
            if policy=='direct16': argv+=['--train-mask','16']
            if policy=='gradual': argv+=['--curriculum-32-steps',str(epoch_steps)]
            plan['jobs'].append({'name':name,'dependencies':['teacher','native',*prerequisites], 'argv':argv})
            group.append(name)
        groups.append(group)
    plan['curriculum_groups']=groups
    plan['parent_plan_identity']=digest(source)
    plan['curriculum_contract']={
        'initialization':'Each run starts from the original OLMoE, not the selected step-11 adapter.',
        'controls':'Same examples/order/steps/seed, both masks evaluated, same union of LoRA parameters.',
        'treatment':'Direct16 versus alternating32/16 versus 32 warmup then alternating32/16.',
        'limitations':'Different mask exposure is intentional. Gradual does not end with exclusively16. No test-based selection of policy.'}
    return plan


def plan_status(plan):
    rows=[]
    for job in plan['jobs']:
        argv=job['argv']; output=Path(argv[argv.index('--output')+1]); path=output/'report.json'
        row={'name':job['name'],'output':str(output),'status':'pending'}
        if path.exists():
            try:
                report=json.loads(path.read_text(encoding='utf8'))
                row['status']='completed' if report.get('completed') else 'incomplete'
                row['recorded_steps']=len(report.get('steps',[]))
            except (OSError,ValueError): row['status']='invalid_report'
        rows.append(row)
    states={r['name']:r['status'] for r in rows}
    for job,row in zip(plan['jobs'],rows):
        row['waiting_for']=[d for d in job['dependencies'] if states.get(d)!='completed']
    teacher=next((j for j in plan['jobs'] if j['name']=='teacher'),None)
    progress=None
    if teacher:
        argv=teacher['argv']; manifest=Path(argv[argv.index('--teacher-dir')+1])/'manifest.json'
        if manifest.exists():
            try:
                data=json.loads(manifest.read_text(encoding='utf8'))
                progress={'entries':len(data.get('entries',[])),'completed':data.get('completed',False),
                          'note':'Manifest progress only; use teacher-audit to verify shards.'}
            except (OSError,ValueError): progress={'error':'Invalid manifest'}
    return {'jobs':rows,'teacher':progress}


def summarize_curriculum(paths):
    reports=[json.loads(p.read_text(encoding='utf8')) for p in paths]
    if not reports or any(not r.get('completed') or r.get('action')!='train' for r in reports):
        raise ValueError('Completed curriculum training reports required')
    groups={}
    for path,r in zip(paths,reports):
        s=r['settings']
        if not s.get('paired_example_order') or set(s.get('mask_sizes',[]))!={16,32}:
            raise ValueError('Paired examples and both evaluation masks required')
        policy='direct16' if s.get('train_mask')=='16' else 'gradual' if s.get('curriculum_32_steps',0)>0 else 'mixed'
        if s.get('train_mask') not in (None,'16'): raise ValueError('Unsupported fixed mask')
        group=groups.setdefault(s['seed'],{})
        if policy in group: raise ValueError('Duplicate policy/seed')
        group[policy]=(path,r)
    rows=[]
    invariant=('steps','lr','alpha','active','experts','mask_sizes','max_length','new_tokens',
               'generation_per_category','rank','adaptation','teacher_eos','supervised')
    anchor=reports[0]
    for r in reports:
        if r['identity']!=anchor['identity'] or any(r['settings'].get(k)!=anchor['settings'].get(k) for k in invariant):
            raise ValueError('Cross-seed source or protocol differs')
    for seed,group in groups.items():
        if set(group)!={'direct16','mixed','gradual'}: raise ValueError('Each seed needs all three policies')
        first=group['mixed'][1]
        for policy,(path,r) in group.items():
            if r['identity']!=first['identity'] or not r.get('teacher_manifest_sha256') or r['teacher_manifest_sha256']!=first.get('teacher_manifest_sha256'):
                raise ValueError('Source, corpus, masks or teacher differ')
            if any(r['settings'].get(k)!=first['settings'].get(k) for k in invariant):
                raise ValueError('Training/evaluation protocol differs')
            if r['trainable_parameters']!=first['trainable_parameters']:
                raise ValueError('Trainable parameter budgets differ')
            if len(r['steps'])!=r['settings']['steps'] or [s['id'] for s in r['steps']]!=[s['id'] for s in first['steps']]:
                raise ValueError('Training examples/order differ')
            for stage in ('test_before','test_after'):
                if [(v['id'],v['mask']) for v in r[stage]['records']]!=[(v['id'],v['mask']) for v in first[stage]['records']]:
                    raise ValueError('Evaluation examples differ')
            if any(abs(a['ce']-b['ce'])>1e-5 for a,b in zip(r['test_before']['records'],first['test_before']['records'])):
                raise ValueError('Initial baselines differ')
            if [(v['id'],v['stage'],v['mask']) for v in r.get('generations',[])]!=[(v['id'],v['stage'],v['mask']) for v in first.get('generations',[])]:
                raise ValueError('Generated evaluation examples differ')
            for mask in ('32','16'):
                values=[v['ce'] for v in r['test_after']['records'] if v['mask']==mask]
                if not values: raise ValueError('Missing evaluation mask')
                rows.append({'seed':seed,'policy':policy,'mask':mask,'ce':statistics.mean(values),
                             'selected_step':r['selected_step'],'report':str(path)})
    return {'kind':'paired_residency_curriculum','completed':True,'rows':rows,
            'limitations':'CE is not conversation quality. Review complete generations blindly. Exposure to masks differs intentionally; results do not establish hardware feasibility on another device.'}


def make_plan(corpus, examples, study, output, seeds, epochs, model_dir):
    train=sum(r['split']=='train' for r in corpus['examples'])
    common=['--examples',examples.as_posix(),'--study',study.as_posix(),'--model-dir',model_dir.as_posix(),
            '--max-length','384','--new-tokens','256','--generation-per-category','60','--reference-batch-size','4',
            '--ram-gib','32','--expert-ram-gib','16',
            '--teacher-eos','--teacher-dir',(output/'teacher').as_posix()]
    jobs=[]
    def add(name, action, extra, dependencies):
        jobs.append({'name':name,'dependencies':dependencies,
                     'argv':['recovery',action,*common,*extra,'--output',(output/name).as_posix()]})
    add('teacher','prepare',[],[])
    # Teacher references live in a separate directory from the preparation report.
    jobs[-1]['argv'][-1]=(output/'teacher_report').as_posix()
    add('native','native',[],[])
    training=[]
    for seed in seeds:
        for adaptation in ('router','expert-lora','router-expert-lora'):
            name=f'{adaptation}_{seed}'
            add(name,'train',['--adaptation',adaptation,'--rank','8','--mask-sizes','32',
                             '--seed',str(seed),'--steps',str(train*epochs),'--eval-interval','500'],['teacher'])
            training.append(name)
    return {'schema':3,'corpus_identity':digest(corpus),'examples':examples.as_posix(),
            'study':study.as_posix(),'jobs':jobs,'seeds':seeds,'epochs':epochs,
            'teacher_dense_fp32_bytes':corpus.get('teacher_dense_fp32_bytes'),
            'comparison_groups':[[f'{a}_{seed}' for a in ('router','expert-lora','router-expert-lora')] for seed in seeds],
            'deferred_curriculum':{'reason':'Separate preregistered follow-up after 32-resident study; do not pick settings using test.',
                'flags':['--mask-sizes','32','16','--curriculum-32-steps',str(train), '--steps',str(train*3)]},
            'quality_contract':'Numeric final-answer scoring, code syntax/format only; correctness and coherence need blinded human review. No host execution of generated code.'}


def summarize(paths):
    from asi.experiments.recovery import compare_reports
    loaded=[(path,json.loads(path.read_text(encoding='utf-8'))) for path in paths]
    native=[(p,r) for p,r in loaded if r.get('action')=='native']
    trained=[(p,r) for p,r in loaded if r.get('action')=='train']
    if len(native)!=1 or not trained or any(not r.get('completed') for _,r in loaded):
        raise ValueError('Provide one completed native report and completed training reports')
    reference=native[0][1]
    groups={}
    for path,report in trained:
        if any(report['identity'][k]!=reference['identity'][k] for k in ('source','corpus')):
            raise ValueError('Native source/corpus differs')
        if {r['id'] for r in report['test_after']['records']}!={r['id'] for r in reference['records']}:
            raise ValueError('Native test examples differ')
        if report['settings']['new_tokens']!=reference['settings']['new_tokens']:
            raise ValueError('Native generation budget differs')
        native_ids={r['id'] for r in reference['generations']}
        for mask in report['test_after']['masks']:
            ids={r['id'] for r in report['generations'] if r['stage']=='after' and r['mask']==mask}
            if ids!=native_ids: raise ValueError('Native generated examples differ')
        groups.setdefault(report['settings']['seed'],[]).append(path)
    comparisons=[]
    for seed,group in groups.items():
        modes=[r['settings']['adaptation'] for p,r in trained if p in group]
        if sorted(modes)!=['expert-lora','router','router-expert-lora']:
            raise ValueError('Each seed needs exactly the three adaptation modes')
        comparisons.extend(dict(row,seed=seed) for row in compare_reports(group)['rows'])
    metrics=[]
    for mode in ('router','expert-lora','router-expert-lora'):
        for mask in sorted({r['mask'] for r in comparisons}):
            rows=[r for r in comparisons if r['adaptation']==mode and r['mask']==mask and r['stage']=='after']
            values=[r['ce'] for r in rows]
            metrics.append({'adaptation':mode,'mask':mask,'seeds':len(rows),
                'ce_mean':statistics.mean(values),'ce_std':statistics.stdev(values) if len(values)>1 else None})
    generations=reference['generations']
    numeric=[r['task_correct'] for r in generations if r.get('task_correct') is not None]
    return {'kind':'recovery_expanded','completed':True,'identity':reference['identity'],
            'native_ce':statistics.mean(r['ce'] for r in reference['records']),
            'native_numeric_passed':sum(numeric),'native_numeric_tasks':len(numeric),
            'native_repeated_trigram_fraction':statistics.mean(r['repeated_trigram_fraction'] for r in generations),
            'native_expert_loads':sum(r['cache_after']['loads']-r['cache_before']['loads'] for r in generations),
            'per_seed':comparisons,'summary':metrics,
            'manual_quality':'Not scored automatically. Use review export; word presence and syntax do not establish correctness.'}


def run_job(plan, name):
    job=next((j for j in plan['jobs'] if j['name']==name),None)
    if job is None: raise ValueError('Unknown stage: '+name)
    corpus=json.loads(Path(plan['examples']).read_text(encoding='utf-8'))
    if digest(corpus)!=plan['corpus_identity']: raise ValueError('Frozen corpus changed')
    calibration=json.loads(Path(plan['study']).read_text(encoding='utf-8'))
    if not calibration.get('completed') or digest(calibration)!=plan.get('study_identity'):
        raise ValueError('Frozen calibration changed or is incomplete; regenerate the plan before starting runs')
    for dependency in job['dependencies']:
        previous=next(j for j in plan['jobs'] if j['name']==dependency)
        previous_args=previous['argv']
        report=json.loads((Path(previous_args[previous_args.index('--output')+1])/'report.json').read_text())
        if not report.get('completed'): raise ValueError('Incomplete dependency: '+dependency)
    argv=list(job['argv'])
    if name=='teacher':
        directory=Path(argv[argv.index('--teacher-dir')+1])
        if (directory/'manifest.json').exists():
            # Reject corruption before changing any prior run directory.
            manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            if manifest['identity']['corpus']!=plan['corpus_identity']:
                raise ValueError('Teacher corpus differs')
            output=Path(argv[argv.index('--output')+1])
            archive_partial_teacher_report(output)
            argv.append('--resume-teacher')
    # Never skip or overwrite a previous run silently. CLI requires fresh output.
    subprocess.run([sys.executable,'-u','-m','asi',*argv],check=True)


def archive_partial_teacher_report(output):
    """Preserve interrupted reports while retaining the plan's canonical output."""
    if not output.exists(): return
    report=json.loads((output/'report.json').read_text(encoding='utf-8'))
    if report.get('action')!='prepare' or report.get('completed'):
        raise ValueError('Refusing to move a completed or unrelated report')
    parent=output.resolve().parent
    for attempt in range(1,10000):
        destination=output.with_name(output.name+f'.attempt{attempt:04d}')
        if destination.exists(): continue
        if output.resolve().parent!=parent or destination.resolve().parent!=parent:
            raise ValueError('Report archive escaped its parent')
        output.rename(destination)
        print('Preserved interrupted teacher report:',destination,flush=True)
        return
    raise ValueError('Too many archived teacher attempts')


def export_review(corpus, reports, output):
    expected={r['id']:r for r in corpus['examples'] if r['split']=='test'}
    rows=[]; key=[]; checks=[]
    for path in reports:
        report=json.loads(path.read_text(encoding='utf-8'))
        if not report.get('completed') or report['identity']['corpus']!=digest(corpus):
            raise ValueError('Incomplete report or corpus mismatch')
        for generation in report.get('generations',[]):
            row=expected[generation['id']]
            review_id=hashlib.sha256((str(path)+generation['id']+generation['stage']+generation['mask']).encode()).hexdigest()[:20]
            rows.append({'review_id':review_id,'prompt':row['prompt'],'reference':row['answer'],
                         'response':generation['text'],'reference_tests':row.get('reference_tests'),
                         'correctness_0_2':None,'coherence_0_2':None,'instruction_following_0_2':None,'notes':''})
            key.append({'review_id':review_id,'source':str(path),'id':generation['id'],
                        'stage':generation['stage'],'mask':generation['mask']})
            checks.append({'review_id':review_id,**{k:generation.get(k) for k in
                ('task_correct','format_pass','python_syntax_pass','answer_phrase_present','repeated_trigram_fraction')}})
    output.mkdir(parents=True,exist_ok=False)
    (output/'review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['review_id'])),encoding='utf-8')
    (output/'key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    (output/'automatic_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (output/'rubric.md').write_text('Score independently: 0 = fails, 1 = partially meets, 2 = meets.\n'
        'Correctness: factual/math/code validity. Coherence: comprehensible, consistent, no disruptive repetition.\n'
        'Instruction following: requested format and all requested subtasks. Leave unknown scores null.\n'
        'Keep key.json and automatic_checks.json hidden from the reviewer until scoring is complete.\n'
        'Python syntax is not functional correctness; reference tests are provided for an isolated review environment.\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run','review','summarize','status','curriculum','curriculum-summary','pilot-run'])
    p.add_argument('--examples',type=Path,default=Path('results/recovery_corpus_v3/corpus.json'))
    p.add_argument('--study',type=Path,default=Path('results/recovery_calibration_batched_v3/report.json'))
    p.add_argument('--model-dir',type=Path,default=Path('public_models/olmoe'))
    p.add_argument('--output',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[1337,2027,31415])
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--plan',type=Path)
    p.add_argument('--stage')
    p.add_argument('--reports',type=Path,nargs='+')
    p.add_argument('--wait-seconds',type=int,default=0,help='Maximum wait for an already running pilot preparation')
    args=p.parse_args()
    if args.action=='pilot-run':
        if not args.output: p.error('Provide pilot root as --output')
        run_pilot(args.output,args.wait_seconds)
        return
    if args.action=='status':
        if not args.plan: p.error('Provide --plan')
        print(json.dumps(plan_status(json.loads(args.plan.read_text(encoding='utf8'))),indent=2))
        return
    if args.action=='curriculum':
        if not args.plan or not args.output or args.output.exists(): p.error('Provide --plan and fresh --output')
        result=curriculum_plan(json.loads(args.plan.read_text(encoding='utf8')),args.output)
        args.output.mkdir(parents=True)
        (args.output/'plan.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        return
    if args.action=='run':
        if not args.plan or not args.stage: p.error('Provide --plan and --stage')
        run_job(json.loads(args.plan.read_text(encoding='utf-8')),args.stage)
        return
    if not args.output or args.output.exists(): p.error('Provide fresh --output')
    if args.action in ('summarize','curriculum-summary'):
        if not args.reports: p.error('Provide --reports')
        result=(summarize_curriculum if args.action=='curriculum-summary' else summarize)(args.reports)
        args.output.mkdir(parents=True)
        (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return
    corpus=json.loads(args.examples.read_text(encoding='utf-8'))
    if args.action=='review':
        if not args.reports: p.error('Provide --reports')
        export_review(corpus,args.reports,args.output)
    else:
        if args.epochs<1 or len(set(args.seeds))!=len(args.seeds): p.error('Positive epochs and unique seeds required')
        calibration=json.loads(args.study.read_text(encoding='utf-8'))
        if not calibration.get('completed') or calibration.get('corpus_identity')!=digest(corpus):
            p.error('Complete calibration for this corpus before freezing the plan')
        plan=make_plan(corpus,args.examples,args.study,args.output,args.seeds,args.epochs,args.model_dir)
        plan['study_identity']=digest(calibration)
        args.output.mkdir(parents=True)
        (args.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print(json.dumps({'jobs':[j['name'] for j in plan['jobs']],
                          'teacher_gib':(plan['teacher_dense_fp32_bytes'] or 0)/2**30},indent=2))
