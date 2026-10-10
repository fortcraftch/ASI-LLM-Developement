"""Declared development-only generation proxy; never a semantic quality score."""
import statistics
import torch


def selection_key(generations, evaluation, baseline):
    if not generations or any(r['split'] != 'dev' for r in generations):
        raise ValueError('Selection requires development generations only')
    numeric=[r['task_correct'] for r in generations if r['task_correct'] is not None]
    formats=[r['format_pass'] for r in generations if r['format_pass'] is not None]
    if not numeric or not formats:
        raise ValueError('Numeric and format validation examples required')
    groups={}
    for r in generations:
        groups.setdefault((r['mask'],r['category']),[]).append(r['repeated_trigram_fraction'])
    def ce_groups(result):
        out={}
        for r in result['records']:
            out.setdefault((r['mask'],r['category']),[]).append(r['ce'])
        return {k:statistics.mean(v) for k,v in out.items()}
    current,initial=ce_groups(evaluation),ce_groups(baseline)
    if current.keys()!=initial.keys() or current.keys()!=groups.keys():
        raise ValueError('Validation category/mask groups differ')
    # Fixed lexicographic rule: mathematical answers, worst repetition, format, CE.
    # Both treatments use CE here, even if training combines CE and KL.
    return (1-sum(numeric)/len(numeric),max(statistics.mean(v) for v in groups.values()),
            1-sum(formats)/len(formats),max(current[k]/max(initial[k],1e-8) for k in current))


def dev_generations(model,cache,dev,mappings,active,tokenizer,new_tokens,per_category):
    from asi.experiments.public_moe import decode
    from asi.experiments.recovery import generation_metrics
    if any(r['split']!='dev' for r in dev):
        raise ValueError('Never generate test examples for selection')
    selected=[]
    for category in sorted({r['category'] for r in dev}):
        selected += [r for r in dev if r['category']==category][:per_category]
    records=[]
    for mask,mapping in mappings.items():
        for row in selected:
            cache.set_context(mapping[row['category']],mixture='restricted',active=active)
            before=cache.stats['loads']
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated=decode(model,torch.tensor([row['tokens'][:row['prompt_length']]]),new_tokens,'cuda',
                                 stop_token_ids=[tokenizer.eos_token_id])
            if cache.stats['loads']!=before:raise RuntimeError('Expert transfer during dev generation')
            text=tokenizer.decode(generated['token_ids'],skip_special_tokens=True)
            records.append(dict(id=row['id'],split='dev',category=row['category'],mask=mask,
                                text=text,**generated,**generation_metrics(generated['token_ids'],text,row)))
    return records
