"""Frozen multi-seed recovery protocol and blinded human review export."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import statistics

from asi.taxonomy import digest


def make_plan(corpus, examples, study, output, seeds, epochs, model_dir):
    train=sum(r['split']=='train' for r in corpus['examples'])
    common=['--examples',str(examples),'--study',str(study),'--model-dir',str(model_dir),
            '--max-length','384','--new-tokens','256','--generation-per-category','60','--reference-batch-size','4',
            '--ram-gib','32','--expert-ram-gib','16',
            '--teacher-eos','--teacher-dir',str(output/'teacher')]
    jobs=[]
    def add(name, action, extra, dependencies):
        jobs.append({'name':name,'dependencies':dependencies,
                     'argv':['recovery',action,*common,*extra,'--output',str(output/name)]})
    add('teacher','prepare',[],[])
    # Teacher references live in a separate directory from the preparation report.
    jobs[-1]['argv'][-1]=str(output/'teacher_report')
    add('native','native',[],[])
    training=[]
    for seed in seeds:
        for adaptation in ('router','expert-lora','router-expert-lora'):
            name=f'{adaptation}_{seed}'
            add(name,'train',['--adaptation',adaptation,'--rank','8','--mask-sizes','32',
                             '--seed',str(seed),'--steps',str(train*epochs),'--eval-interval','500'],['teacher'])
            training.append(name)
    return {'schema':3,'corpus_identity':digest(corpus),'examples':str(examples),
            'study':str(study),'jobs':jobs,'seeds':seeds,'epochs':epochs,
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
        report=json.loads((Path(previous['argv'][-1])/'report.json').read_text())
        if not report.get('completed'): raise ValueError('Incomplete dependency: '+dependency)
    argv=list(job['argv'])
    if name=='teacher':
        directory=Path(argv[argv.index('--teacher-dir')+1])
        if (directory/'manifest.json').exists(): argv.append('--resume-teacher')
    # Never skip or overwrite a previous run silently. CLI requires fresh output.
    subprocess.run([sys.executable,'-u','-m','asi',*argv],check=True)


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
    p.add_argument('action',choices=['plan','run','review','summarize'])
    p.add_argument('--examples',type=Path,default=Path('results/recovery_corpus_v3/corpus.json'))
    p.add_argument('--study',type=Path,default=Path('results/recovery_calibration_batched_v3/report.json'))
    p.add_argument('--model-dir',type=Path,default=Path('public_models/olmoe'))
    p.add_argument('--output',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[1337,2027,31415])
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--plan',type=Path)
    p.add_argument('--stage')
    p.add_argument('--reports',type=Path,nargs='+')
    args=p.parse_args()
    if args.action=='run':
        if not args.plan or not args.stage: p.error('Provide --plan and --stage')
        run_job(json.loads(args.plan.read_text(encoding='utf-8')),args.stage)
        return
    if not args.output or args.output.exists(): p.error('Provide fresh --output')
    if args.action=='summarize':
        if not args.reports: p.error('Provide --reports')
        result=summarize(args.reports)
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
