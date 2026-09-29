"""Masked recovery: routers and persistent expert LoRA, with held-out splits."""
import argparse
import json
from pathlib import Path
import random
import re
import time
import statistics
import torch
from torch.nn import functional as F
from asi import ROOT
from asi.taxonomy import digest
from asi.models.olmoe_storage import load_disk_backed
from asi.models.hf_olmoe import OlmoeExpertCache, CategoryCalibration
from asi.models.recovery_lora import ExpertLoRA
from asi.experiments.public_moe import read_lock, input_ids, choose_capacity, hardware, check_rss, decode, sha256


def read_examples(path):
    content = json.loads(path.read_text(encoding='utf-8'))
    rows = content['examples']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty example IDs required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']) for r in rows):
        raise ValueError('Example IDs must be safe file names')
    if set(r['split'] for r in rows) != {'train', 'dev', 'test'}:
        raise ValueError('Separate train/dev/test splits required')
    texts = [r['prompt'].strip().casefold() for r in rows]
    if len(set(texts)) != len(texts) or any(not r['answer'].strip() or not r['prompt'].strip() for r in rows):
        raise ValueError('Duplicate prompts or empty answers')
    return content


def supervised_rows(corpus, tokenizer, max_length):
    rows = []
    if tokenizer.eos_token_id is None:
        raise ValueError('Supervised recovery requires an EOS token')
    for row in corpus['examples']:
        prompt = input_ids(tokenizer, row['prompt'])[0].tolist()
        answer = tokenizer.encode(row['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt)+len(answer) > max_length:
            raise ValueError('Example exceeds length budget: '+row['id'])
        rows.append({**row, 'tokens': prompt+answer, 'prompt_length': len(prompt)})
    return rows


def adaptation_state(model, adapter):
    return {'routers': router_state(model), 'adapters': adapter.cpu_state() if adapter is not None else {}}


def restore_adaptation(model, adapter, state):
    restore_routers(model, state['routers'])
    if adapter is not None:
        adapter.load_state_dict(state['adapters'], strict=True)
    elif state.get('adapters'):
        raise ValueError('Checkpoint requires expert adapters')


def evaluate_masks(model, cache, rows, mappings, active, device, alpha):
    results = {name: evaluate(model, cache, rows, mapping, active, device, alpha)
               for name, mapping in mappings.items()}
    records = [{**r, 'mask': name} for name, result in results.items() for r in result['records']]
    return {'mean_loss': sum(r['loss'] for r in records)/len(records),
            'records': records, 'masks': results}


def group_losses(result):
    groups = {}
    for row in result['records']:
        groups.setdefault((row['mask'], row['category']), []).append(row['loss'])
    return {key: sum(values)/len(values) for key, values in groups.items()}


def robust_score(result, baseline):
    """Worst relative loss across categories AND masks; step zero scores one."""
    current, original = group_losses(result), group_losses(baseline)
    if current.keys() != original.keys():
        raise ValueError('Validation groups differ')
    return max(current[key]/max(original[key], 1e-8) for key in current)


def generation_metrics(tokens, text, row):
    triples = [tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return {'repeated_trigram_fraction': 1-len(set(triples))/len(triples) if triples else 0.,
            'exact_match': text.strip()==row['expected_exact'] if 'expected_exact' in row else None,
            'note': 'Repetition is a diagnostic, not semantic correctness; exact match only for explicit exact-answer tasks.'}


def compare_reports(paths):
    reports=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if len(reports)<2 or any(not r.get('completed') or r.get('action')!='train' for r in reports):
        raise ValueError('Compare at least two completed training reports')
    first=reports[0]
    settings=('supervised','steps','seed','lr','alpha','active','experts','mask_sizes','max_length','new_tokens','generation_per_category')
    for report in reports[1:]:
        if report['identity']!=first['identity'] or any(report['settings'][k]!=first['settings'][k] for k in settings):
            raise ValueError('Different data, masks, source or training/generation protocol')
        if [(r['id'],r['mask']) for r in report['steps']]!=[(r['id'],r['mask']) for r in first['steps']]:
            raise ValueError('Training exposure differs')
        if len(report['test_before']['records'])!=len(first['test_before']['records']):
            raise ValueError('Baseline evaluation size differs')
        if any(abs(a['ce']-b['ce'])>1e-5 or (a['id'],a['mask'])!=(b['id'],b['mask'])
               for a,b in zip(report['test_before']['records'],first['test_before']['records'])):
            raise ValueError('Unadapted baselines differ')
    rows=[]
    for path,report in zip(paths,reports):
        for mask in report['test_before']['masks']:
            for stage in ('before','after'):
                records=report['test_'+stage]['masks'][mask]['records']
                generations=[r for r in report['generations'] if r['stage']==stage and r['mask']==mask]
                exact=[r['exact_match'] for r in generations if r['exact_match'] is not None]
                rows.append({'source':str(path),'adaptation':report['settings']['adaptation'],'mask':mask,'stage':stage,
                    'test_examples':len(records),'ce':statistics.mean(r['ce'] for r in records),
                    'generated_examples':len(generations),'exact_passed':sum(exact),'exact_tasks':len(exact),
                    'repeated_trigram_fraction':statistics.mean(r['repeated_trigram_fraction'] for r in generations),
                    'expert_loads_during_generation':sum(r['cache_after']['loads']-r['cache_before']['loads'] for r in generations),
                    'selected_step':report['selected_step'],'trainable_parameters':report['trainable_parameters'],
                    'peak_training_gib':report['peak_cuda_bytes']/2**30,
                    'training_with_validation_seconds':report['training_with_validation_seconds']})
    return {'kind':'recovery_comparison','completed':True,'identity':first['identity'],'rows':rows,
            'limitations':['Synthetic single-seed pilot; no semantic judge or external benchmark.',
                           'Exact matches cover only explicit literal-answer tasks, not all generations.',
                           'Repetition can improve by premature EOS; inspect lengths and generated text.',
                           'Same training exposure, but different trainable parameter and compute budgets.']}


def transition_check(model, cache, rows, mappings, active, device, alpha):
    first=rows[0]
    other=next((r for r in rows if r['category']!=first['category']),None)
    if other is None: return []
    checks=[]
    for name,mapping in mappings.items():
        before=evaluate(model,cache,[first],mapping,active,device,alpha)['records'][0]
        evaluate(model,cache,[other],mapping,active,device,alpha)
        returned=evaluate(model,cache,[first],mapping,active,device,alpha)['records'][0]
        delta=abs(returned['ce']-before['ce'])
        if delta>1e-5: raise RuntimeError('A-B-A context transition changed prediction loss')
        checks.append({'mask':name,'first':first['id'],'other':other['id'],'ce_difference_after_return':delta})
    return checks


def enable_routers(model):
    model.requires_grad_(False)
    parameters = []
    for layer in model.model.layers:
        weight = layer.mlp.gate.weight
        weight.data = weight.data.float()
        weight.requires_grad_(True)
        parameters.append(weight)
    return parameters


def router_state(model):
    return {str(i): layer.mlp.gate.weight.detach().cpu().clone() for i, layer in enumerate(model.model.layers)}


def restore_routers(model, state):
    if set(state) != {str(i) for i in range(len(model.model.layers))}:
        raise ValueError('Router layer set differs')
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            if state[str(i)].shape != layer.mlp.gate.weight.shape:
                raise ValueError('Router shape differs')
            layer.mlp.gate.weight.copy_(state[str(i)])


def objective(model, example, device, alpha=.5):
    sequence = torch.tensor([example['tokens']], device=device)
    # Autocast retains FP32 master router parameters with BF16 model computation.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
        logits = model(sequence, use_cache=False).logits[0, example['prompt_length']-1:-1].float()
    targets = sequence[0, example['prompt_length']:]
    logp = logits.log_softmax(-1)
    ce = F.nll_loss(logp, targets)
    if 'teacher_logp' in example:
        teacher = example['teacher_logp'].to(device).float()
        kl = (teacher.exp() * (teacher-logp)).sum(-1).mean()
    else:
        if alpha != 1:
            raise ValueError('Distillation requires teacher distributions')
        kl = ce.new_zeros(())
    return alpha*ce+(1-alpha)*kl, ce, kl


def evaluate(model, cache, rows, mapping, active, device, alpha):
    records = []
    for row in rows:
        cache.set_context(mapping[row['category']], mixture='restricted', active=active)
        before = cache.stats['loads']
        with torch.no_grad():
            loss, ce, kl = objective(model, row, device, alpha)
        if cache.stats['loads'] != before:
            raise RuntimeError('Evaluation changed expert residency')
        records.append({'id': row['id'], 'category': row['category'], 'ce': ce.item(),
                        'kl': kl.item() if 'teacher_logp' in row else None, 'loss': loss.item()})
    return {'mean_loss': sum(r['loss'] for r in records)/len(records), 'records': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['compare','calibrate', 'prepare', 'train', 'evaluate'])
    p.add_argument('--reports',type=Path,nargs='+',help='Completed reports for offline comparison')
    p.add_argument('--model-dir', type=Path, default=ROOT/'public_models/olmoe')
    p.add_argument('--study', type=Path, default=ROOT/'results/olmoe_routing_study_v1.json')
    p.add_argument('--examples', type=Path, default=ROOT/'configs/recovery_pilot_v1.json')
    p.add_argument('--teacher-dir', type=Path, default=ROOT/'results/recovery_teacher_v1')
    p.add_argument('--resume-teacher', action='store_true', help='Reuse verified examples from a partial teacher manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--supervised', action='store_true', help='Assistant-token CE plus EOS, no teacher preparation')
    p.add_argument('--adaptation', choices=['router','expert-lora','router-expert-lora'], default='router')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--mask-sizes', type=int, nargs='+', help='Train/evaluate each calibrated resident selection, e.g. 32 16')
    p.add_argument('--generation-per-category', type=int, default=1)
    p.add_argument('--skip-generation', action='store_true', help='Evaluate checkpoint losses and transitions without repeating free generation')
    p.add_argument('--calibration-per-category', type=int, default=4)
    p.add_argument('--experts', type=int, default=32)
    p.add_argument('--active', type=int, default=8)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--eval-interval', type=int, default=6)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--max-length', type=int, default=96)
    p.add_argument('--new-tokens', type=int, default=32)
    p.add_argument('--ram-gib', type=float, default=32)
    args = p.parse_args()
    if args.action=='compare':
        if args.output.exists() or not args.reports: p.error('Provide --reports and a fresh output directory')
        comparison=compare_reports(args.reports)
        args.output.mkdir(parents=True)
        (args.output/'report.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
        print(json.dumps(comparison,indent=2))
        return
    if args.skip_generation and args.action!='evaluate': p.error('--skip-generation is only for evaluate')
    loaded_checkpoint = None
    if args.action == 'evaluate' and args.checkpoint is not None:
        loaded_checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        settings = loaded_checkpoint.get('recovery_settings', {})
        for key in ('supervised','adaptation','rank','mask_sizes'):
            if key in settings: setattr(args, key, settings[key])
    if args.supervised:
        args.alpha = 1.
        if args.action == 'prepare': p.error('Supervised mode does not need a teacher; use train')
    if args.rank < 1 or args.generation_per_category < 1:
        p.error('Rank and generation count must be positive')
    if args.mask_sizes and (args.experts not in args.mask_sizes or min(args.mask_sizes)<args.active or len(set(args.mask_sizes))!=len(args.mask_sizes)):
        p.error('Unique mask sizes must include --experts and be at least --active')
    if args.output.exists() or min(args.steps,args.eval_interval,args.experts,args.active,args.max_length,args.new_tokens,args.lr,args.ram_gib)<=0 or not 0<=args.alpha<=1:
        p.error('Choose a fresh output and valid positive settings')
    if args.action=='prepare' and args.teacher_dir.exists() and not args.resume_teacher:
        p.error('Choose a fresh teacher directory')
    if args.action=='evaluate' and args.checkpoint is None:
        p.error('evaluate requires --checkpoint')
    torch.manual_seed(args.seed)
    source = digest(read_lock(args.model_dir))
    corpus = read_examples(args.examples)
    if args.action == 'calibrate':
        hw = hardware(args.model_dir,args.ram_gib)
        if not hw['gpu'] or args.calibration_per_category < 1: p.error('CUDA and positive calibration count required')
        model,tokenizer,store=load_disk_backed(args.model_dir,256*1024**2,int(args.ram_gib*2**30))
        sizes=args.mask_sizes or [args.experts]
        choose_capacity(model,max(sizes),hw['gpu']['free_bytes'],2*2**30)
        cache=OlmoeExpertCache(model,capacity=max(sizes)*model.config.num_hidden_layers,store=store)
        collector=CategoryCalibration(model.config.num_hidden_layers,model.config.num_experts)
        cache.callbacks.append(collector.observe)
        train=[r for r in supervised_rows(corpus,tokenizer,args.max_length) if r['split']=='train']
        categories=sorted({r['category'] for r in train})
        args.output.mkdir(parents=True)
        report={'completed':False,'source_identity':source,'corpus_identity':digest(corpus),'train_ids':[],
                'protocol':'native routing on train prompts plus assistant answers and EOS','hardware':hw}
        try:
            for category in categories:
                rows=[r for r in train if r['category']==category][:args.calibration_per_category]
                if len(rows)!=args.calibration_per_category: raise ValueError('Not enough calibration train examples')
                collector.labels=[category]
                for row in rows:
                    print('Calibration '+row['id'],flush=True)
                    with torch.inference_mode(): model(torch.tensor([row['tokens']],device='cuda'),use_cache=False)
                    report['train_ids'].append(row['id'])
                    (args.output/'report.json').write_text(json.dumps(report,indent=2))
                    check_rss(int(args.ram_gib*2**30))
            report.update(completed=True,mappings={str(k):collector.mapping(categories,k) for k in sizes})
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
        finally: cache.close()
        return
    study = json.loads(args.study.read_text())
    if not study.get('completed') or study['source_identity'] != source:
        p.error('Study must be completed on this checkpoint')
    mapping = study['mappings'][str(args.experts)]
    mappings = {str(k): study['mappings'][str(k)] for k in (args.mask_sizes or [args.experts])}
    if any(r['category'] not in mapping for r in corpus['examples']):
        p.error('Corpus category absent from calibrated mapping')
    if args.active>args.experts:
        p.error('Active count exceeds resident count')
    identity = {'source': source, 'mapping': digest(mapping), 'corpus': digest(corpus),
                'experts': args.experts, 'active': args.active, 'policy': 'restricted'}
    if args.supervised: identity['objective'] = 'assistant_ce_with_eos'
    if args.mask_sizes: identity['mask_bank'] = digest(mappings)
    hw = hardware(args.model_dir,args.ram_gib)
    if not hw['gpu']:
        p.error('CUDA required')
    model, tokenizer, store = load_disk_backed(args.model_dir, 256*1024**2, int(args.ram_gib*2**30))
    capacity = max(map(int, mappings))
    choose_capacity(model,capacity,hw['gpu']['free_bytes'],2*2**30)
    cache = OlmoeExpertCache(model,capacity=capacity*model.config.num_hidden_layers,store=store)
    device = torch.device('cuda')
    args.output.mkdir(parents=True)
    report = {'identity':identity,'action':args.action,'completed':False,'hardware':hw,
              'torch_version':str(torch.__version__),
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    def save():
        pending=args.output/'report.tmp'
        pending.write_text(json.dumps(report,indent=2),encoding='utf-8')
        pending.replace(args.output/'report.json')
    save()
    try:
        if args.action=='prepare':
            args.teacher_dir.mkdir(parents=True,exist_ok=args.resume_teacher)
            entries=[]
            manifest_path=args.teacher_dir/'manifest.json'
            if args.resume_teacher and manifest_path.exists():
                previous=json.loads(manifest_path.read_text())
                if previous['identity']!=identity: raise ValueError('Teacher resume identity differs')
                entries=previous['entries']
                if len({e['id'] for e in entries})!=len(entries) or not {e['id'] for e in entries}<={r['id'] for r in corpus['examples']}:
                    raise ValueError('Invalid teacher resume entries')
                for entry in entries:
                    path=(args.teacher_dir/entry['file']).resolve()
                    if not path.is_relative_to(args.teacher_dir.resolve()) or sha256(path)!=entry['sha256']:
                        raise ValueError('Teacher resume shard changed')
            elif args.resume_teacher:
                raise ValueError('Resuming requires a verified manifest')
            def save_manifest(completed=False):
                pending=manifest_path.with_suffix('.tmp')
                pending.write_text(json.dumps({'identity':identity,'entries':entries,'completed':completed},indent=2))
                pending.replace(manifest_path)
            save_manifest()
            for row in corpus['examples']:
                if any(e['id']==row['id'] for e in entries): continue
                print('Teacher '+row['id'],flush=True)
                prompt = input_ids(tokenizer,row['prompt'])[0].tolist()
                answer = tokenizer.encode(row['answer'],add_special_tokens=False)
                if not answer or len(prompt)+len(answer)>args.max_length:
                    raise ValueError('Example exceeds length budget: '+row['id'])
                tokens=prompt+answer
                with torch.inference_mode():
                    logits=model(torch.tensor([tokens],device=device),use_cache=False).logits[0,len(prompt)-1:-1].float()
                    logp=logits.log_softmax(-1).cpu()
                filename=row['id']+'.pt'
                torch.save({**row,'tokens':tokens,'prompt_length':len(prompt),'teacher_logp':logp},args.teacher_dir/filename)
                entries.append({'id':row['id'],'split':row['split'],'file':filename,'sha256':sha256(args.teacher_dir/filename)})
                save_manifest()
                check_rss(int(args.ram_gib*2**30))
            save_manifest(completed=True)
            report['teacher_examples']=len(entries)
        else:
            if args.supervised:
                rows=supervised_rows(corpus,tokenizer,args.max_length)
            else:
                manifest=json.loads((args.teacher_dir/'manifest.json').read_text())
                if manifest['identity']!=identity:
                    raise ValueError('Teacher identity differs from source, corpus or mapping')
                rows=[]
                expected={r['id']:r for r in corpus['examples']}
                if {e['id'] for e in manifest['entries']}!=set(expected) or len(manifest['entries'])!=len(expected):
                    raise ValueError('Teacher examples differ from corpus')
                for entry in manifest['entries']:
                    path=(args.teacher_dir/entry['file']).resolve()
                    if not path.is_relative_to(args.teacher_dir.resolve()) or sha256(path)!=entry['sha256']:
                        raise ValueError('Teacher shard changed or escaped directory')
                    row=torch.load(path,map_location='cpu',weights_only=True)
                    if any(row.get(k)!=v for k,v in expected[entry['id']].items()):
                        raise ValueError('Teacher text, category or split changed')
                    if row['teacher_logp'].shape != (len(row['tokens'])-row['prompt_length'],model.config.vocab_size):
                        raise ValueError('Teacher distribution shape differs')
                    rows.append(row)
            parameters=enable_routers(model)
            if args.adaptation == 'expert-lora':
                for parameter in parameters: parameter.requires_grad_(False)
                parameters=[]
            adapter = ExpertLoRA(model,mappings,args.rank) if args.adaptation != 'router' else None
            cache.expert_adapter=adapter
            if adapter is not None: parameters+=list(adapter.parameters())
            initial=adaptation_state(model,adapter)
            report['trainable_parameters']=sum(p.numel() for p in parameters)
            report['adapter_parameter_bytes']=sum(p.numel()*p.element_size() for p in adapter.parameters()) if adapter is not None else 0
            if not args.supervised: report['teacher_manifest_sha256']=sha256(args.teacher_dir/'manifest.json')
            if args.action=='train':
                train=[r for r in rows if r['split']=='train']
                dev=[r for r in rows if r['split']=='dev']
                print('Evaluating validation baseline',flush=True)
                baseline=evaluate_masks(model,cache,dev,mappings,args.active,device,args.alpha)
                robust_selection = bool(args.supervised or args.mask_sizes or adapter is not None)
                best=1. if robust_selection else baseline['mean_loss']
                best_state=initial; best_step=0
                report.update(dev_before=baseline,steps=[],dev_checks=[],train_ids=[r['id'] for r in train])
                optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0)
                order=[(i,name) for name in mappings for i in range(len(train))]; rng=random.Random(args.seed)
                torch.cuda.reset_peak_memory_stats()
                training_started=time.perf_counter()
                for step in range(args.steps):
                    if step%len(order)==0: rng.shuffle(order)
                    row_index,mask_name=order[step%len(order)]
                    row=train[row_index]
                    cache.set_context(mappings[mask_name][row['category']],mixture='restricted',active=args.active)
                    before=cache.stats['loads']
                    optimizer.zero_grad(set_to_none=True)
                    loss,ce,kl=objective(model,row,device,args.alpha)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite recovery loss')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    optimizer.step()
                    if cache.stats['loads']!=before: raise RuntimeError('Expert loads during training step')
                    report['steps'].append({'step':step+1,'id':row['id'],'mask':mask_name,'loss':loss.item(),'ce':ce.item(),
                                            'kl':kl.item() if not args.supervised else None,'grad_norm':float(norm)})
                    print(f"Step {step+1}: loss={loss.item():.4f}",flush=True)
                    check_rss(int(args.ram_gib*2**30))
                    if (step+1)%args.eval_interval==0 or step+1==args.steps:
                        result=evaluate_masks(model,cache,dev,mappings,args.active,device,args.alpha)
                        score=robust_score(result,baseline)
                        result['worst_group_ratio']=score
                        if not robust_selection: score=result['mean_loss']
                        print(f"Validation {step+1}: loss={result['mean_loss']:.4f}",flush=True)
                        report['dev_checks'].append({'step':step+1,**result})
                        if score<best:
                            best=score; best_state=adaptation_state(model,adapter); best_step=step+1
                    save()
                checkpoint={'schema':2,'identity':identity,**best_state,'selected_step':best_step,
                            'selection':('minimum worst relative dev loss across category/mask groups' if robust_selection else 'minimum mean dev loss')+', including step zero','alpha':args.alpha,
                            'recovery_settings':{k:getattr(args,k) for k in ('supervised','adaptation','rank','mask_sizes')}}
                prefix='router' if adapter is None else 'adaptation'
                torch.save(checkpoint,args.output/(prefix+'_best.pt'))
                torch.save({**checkpoint,**adaptation_state(model,adapter),'selected_step':args.steps,
                            'selection':'last training step; not selected using test'},args.output/(prefix+'_last.pt'))
                report.update(selected_step=best_step,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                              training_with_validation_seconds=time.perf_counter()-training_started)
                report['dev_best_worst_group_ratio' if robust_selection else 'dev_best_loss']=best
            else:
                checkpoint=loaded_checkpoint
                if checkpoint['identity']!=identity:
                    raise ValueError('Recovery checkpoint identity differs')
                report['checkpoint_sha256']=sha256(args.checkpoint)
                report['selected_step']=checkpoint['selected_step']
                args.alpha=checkpoint['alpha']
                report['settings']['alpha']=args.alpha
            # Test split is never used to select a checkpoint or optimizer steps.
            print('Final held-out comparison',flush=True)
            test=[r for r in rows if r['split']=='test']
            if not args.supervised:
                report['teacher_test_ce']=sum(-r['teacher_logp'].gather(-1,torch.tensor(r['tokens'][r['prompt_length']:])[:,None]).mean().item() for r in test)/len(test)
            restore_adaptation(model,adapter,initial)
            report['test_before']=evaluate_masks(model,cache,test,mappings,args.active,device,args.alpha)
            report['generations']=[]
            generation_rows=[]
            for category in sorted({r['category'] for r in test}):
                if not args.skip_generation:
                    generation_rows += [r for r in test if r['category']==category][:args.generation_per_category]
            for name,state in [('before',initial),('after',checkpoint)]:
                restore_adaptation(model,adapter,state)
                for mask_name,selected in mappings.items():
                    for row in generation_rows:
                        print(f"Generation {name}: {row['id']}",flush=True)
                        cache.set_context(selected[row['category']],mixture='restricted',active=args.active)
                        before=cache.snapshot()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            generation=decode(model,torch.tensor([row['tokens'][:row['prompt_length']]]),args.new_tokens,'cuda',
                                              stop_token_ids=[tokenizer.eos_token_id] if args.supervised else None)
                        after=cache.snapshot()
                        if after['loads']!=before['loads']: raise RuntimeError('Expert loads during generation')
                        report['generations'].append({'stage':name,'id':row['id'],'mask':mask_name,**generation,'text':tokenizer.decode(generation['token_ids']),
                                                      **generation_metrics(generation['token_ids'],tokenizer.decode(generation['token_ids'],skip_special_tokens=True),row),
                                                      'cache_before':before,'cache_after':after})
                        save()
            report['test_after']=evaluate_masks(model,cache,test,mappings,args.active,device,args.alpha)
            report['transition_checks']=transition_check(model,cache,test,mappings,args.active,device,args.alpha)
        report.update(completed=True,rss_bytes=check_rss(int(args.ram_gib*2**30)),cache_final=cache.snapshot(),
            limitations=['Synthetic pilot, no claim of general quality recovery.','Base weights frozen; optional persistent down-projection expert LoRA.',
                         'Adaptation is bound to the calibrated mask bank; arbitrary unseen selections are not validated.',
                         'No KL or native reference is measured in supervised-only mode.',
                         'Dev selects checkpoint including step zero; test is reserved for final before/after comparison.'])
        save()
    finally:
        cache.close()
