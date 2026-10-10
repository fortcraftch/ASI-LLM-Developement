"""Paired five-way recovery evidence: keep quality proxies distinct from correctness."""
import html
import json
from pathlib import Path
import random
import statistics

from asi.experiments.recovery_validation import VARIANTS, read, fingerprint

LABELS = {'original': 'Original', 'unadapted32': '32 unadapted', 'recovered32': '32 recovered',
          'unadapted16': '16 unadapted', 'recovered16': '16 recovered'}


def validate(reports, plan, plan_hash):
    if set(reports) != set(plan.get('variants', VARIANTS)) or 'original' not in reports:
        raise ValueError('All planned variants required')
    base = reports['original']
    for planned, actual in (('new_tokens','new_tokens'), ('gpu_allocator_cap_gib','gpu_allocator_cap_gib'),
                            ('expert_ram_gib','expert_ram_gib'), ('process_ram_gib','process_ram_gib'),
                            ('native_cache_experts','native_cache_experts')):
        if planned in plan and base['protocol'].get(actual) != plan[planned]:
            raise ValueError('Execution differs from frozen '+planned)
    for name, report in reports.items():
        if not report.get('completed') or report.get('validation_plan_sha256') != plan_hash:
            raise ValueError('Incomplete or unverified run: '+name)
        if [r['id'] for r in report['records']] != plan['prompt_ids']:
            raise ValueError('Missing, duplicated or reordered test examples')
        for key in ('source_identity', 'corpus_identity', 'mapping_identity', 'protocol', 'gpu', 'torch'):
            if report[key] != base[key]:
                raise ValueError('Incomparable '+key)
        expected_variant = 'original' if name == 'original' else name[-2:]
        if report['variant'] != expected_variant or report['adapted'] != name.startswith('recovered'):
            raise ValueError('Mislabeled variant')
        if name.startswith('recovered') and report['checkpoint_sha256'] != plan['hashes']['checkpoint'+name[-2:]]:
            raise ValueError('Different adaptation checkpoint')
        for a, b in zip(base['records'], report['records']):
            if any(a[k] != b[k] for k in ('id', 'prompt', 'category', 'input_ids', 'reference_answer')):
                raise ValueError('Different input/reference')
            if name != 'original' and b['generation_transfers']['loads'] != 0:
                raise ValueError('Restricted generation unexpectedly swapped experts')
    for size in ('32', '16'):
        if 'unadapted'+size not in reports and 'recovered'+size not in reports:
            continue
        if 'unadapted'+size not in reports or 'recovered'+size not in reports:
            raise ValueError('Missing before/after control')
        if not reports['unadapted'+size]['mask_identity'] or reports['unadapted'+size]['mask_identity'] != reports['recovered'+size]['mask_identity']:
            raise ValueError('Recovery controls have different masks')


def aggregate(report):
    rows = report['records']
    def measured(key):
        values = [r['metrics'][key] for r in rows if r['metrics'].get(key) is not None]
        return dict(passed=sum(values), evaluated=len(values))
    return dict(examples=len(rows), numeric_correct=measured('task_correct'),
                format_pass=measured('format_pass'), python_syntax=measured('python_syntax_pass'),
                biology_phrase=measured('answer_phrase_present'),
                median_repetition=statistics.median(r['metrics']['repeated_trigram_fraction'] for r in rows),
                eos=sum(r['stop_reason']=='EOS' for r in rows),
                peak_gpu_gib=max(r['peak_cuda_allocated_bytes'] for r in rows)/2**30,
                peak_rss_gib=max(r['sampled_peak_rss_bytes'] for r in rows)/2**30,
                median_response_s=statistics.median(r['prepare_seconds']+r['wall_generation_seconds'] for r in rows),
                preparation_h2d_gib=sum(r['prepare_transfers']['host_to_device_bytes'] for r in rows)/2**30,
                generation_h2d_gib=sum(r['generation_transfers']['host_to_device_bytes'] for r in rows)/2**30,
                generation_loads=sum(r['generation_transfers']['loads'] for r in rows),
                category_transitions=sum(r['category_changed'] for r in rows))


def render(directory):
    directory = Path(directory)
    plan = read(directory/'plan.json')
    variants = plan['variants']
    reports = {v: read(directory/v/'report.json') for v in variants}
    validate(reports, plan, fingerprint(directory/'plan.json'))
    stats = {v: aggregate(r) for v, r in reports.items()}
    paired = {}
    for size in ('32', '16'):
        if 'unadapted'+size not in reports:
            continue
        before, after = reports['unadapted'+size], reports['recovered'+size]
        paired[size] = [dict(id=a['id'], category=a['category'],
                            repetition_delta=b['metrics']['repeated_trigram_fraction']-a['metrics']['repeated_trigram_fraction'],
                            numeric_before=a['metrics']['task_correct'], numeric_after=b['metrics']['task_correct'],
                            eos_before=a['stop_reason']=='EOS', eos_after=b['stop_reason']=='EOS')
                        for a,b in zip(before['records'],after['records'])]
    summary = dict(completed=True, kind='recovery_validation', variants=stats, paired_recovery=paired,
                   sources={v:fingerprint(directory/v/'report.json') for v in variants},
                   manual_semantic_review='pending; proxies are not semantic correctness', limitations=plan['limitations'])
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    specs = [('peak_gpu_gib','Peak PyTorch GPU allocation (GiB)'),
             ('median_response_s','Median prepare + generate time (s)'),
             ('generation_h2d_gib','Total generation H2D (GiB)'),
             ('median_repetition','Median repeated trigram fraction'),
             ('eos','Answers ending in EOS'),('generation_loads','Expert loads during generation')]
    for ax,(key,title) in zip(axes.flat,specs):
        colors=dict(zip(VARIANTS,['#64748b','#93c5fd','#2563eb','#f9a8d4','#db2777']))
        bars=ax.bar(range(len(variants)),[stats[v][key] for v in variants],color=[colors[v] for v in variants])
        ax.bar_label(bars,fmt='%.2f',fontsize=8)
        ax.set_xticks(range(len(variants)),[LABELS[v].replace(' ','\n') for v in variants],fontsize=8)
        ax.set_title(title,fontsize=10);ax.margins(y=.2)
    fig.suptitle('OLMoE: same held-out prompts, original / restricted / recovered')
    fig.text(.5,.01,'Greedy, EOS or 256 tokens; unequal response lengths; one pass. Repetition and EOS do not establish correctness.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(directory/'comparison.png',dpi=160);fig.savefig(directory/'comparison.pdf');plt.close(fig)
    lines=['# Controlled recovery validation','',
           'Same masks before/after adaptation. Held-out questions selected before generation. '
           'Oracle category labels; no classifier or chat history. Timing is exploratory, with unequal EOS lengths.','',
           '| Variant | Numeric correct | Format | Legacy response syntax (not correctness) | Biology phrase (not correctness) | EOS | Peak GPU GiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        s=stats[v]
        scores=[str(s[k]['passed'])+'/'+str(s[k]['evaluated']) for k in ('numeric_correct','format_pass','python_syntax','biology_phrase')]
        lines.append('| '+LABELS[v]+' | '+' | '.join(scores)+f" | {s['eos']}/{s['examples']} | {s['peak_gpu_gib']:.2f} |")
    lines += ['','![Resources and repetition](comparison.png)','',
              'Full outputs are in outputs.html. Blind review cards hide model labels; the separate key is for unblinding after scoring.',
              'The legacy syntax metric parses the entire response when the strict code-only format fails; '
              'it can reject a valid code block accompanied by prose. Use recovery_validation_audit for a separate first-block diagnostic.',
              'No generated Python is executed by this evaluator. Semantic correctness and coherence require review.','',
              *['- '+x for x in plan['limitations']]]
    (directory/'summary.md').write_text('\n'.join(lines),encoding='utf8')
    cards=[]; blind=[]; key={}; rng=random.Random(20261008)
    for i, row in enumerate(reports['original']['records']):
        cards.append('<h2>'+html.escape(row['id'])+'</h2><p>'+html.escape(row['prompt'])+'</p><p>Reference: '+html.escape(row['reference_answer'])+'</p>')
        shuffled=list(variants);rng.shuffle(shuffled)
        for j,v in enumerate(shuffled):
            r=reports[v]['records'][i]; code=f'Q{i+1:02d}-{j+1}'
            key[code]=dict(variant=v,id=row['id'])
            blind.append(dict(code=code,prompt=row['prompt'],reference=row['reference_answer'],text=r['text'],
                              correctness_0_2=None, coherence_0_2=None, instruction_following_0_2=None, notes=''))
        for v in variants:
            cards.append('<h3>'+LABELS[v]+'</h3><pre>'+html.escape(reports[v]['records'][i]['text'])+'</pre>')
    (directory/'outputs.html').write_text('<!doctype html><meta charset="utf-8"><title>Recovery validation outputs</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}</style>'+''.join(cards),encoding='utf8')
    # Do not overwrite human scores when regenerating plots.
    if not (directory/'blind_review.json').exists():
        (directory/'blind_review.json').write_text(json.dumps(blind,indent=2,ensure_ascii=False),encoding='utf8')
    (directory/'blind_key.json').write_text(json.dumps(key,indent=2),encoding='utf8')
    return summary


def compare_memory(first, second, output):
    """Only interpret cache/budget changes when greedy output tokens are identical."""
    first, second, output = Path(first), Path(second), Path(output)
    if output.exists():
        raise ValueError('Use a fresh output file')
    plans = [read(d/'plan.json') for d in (first,second)]
    reports = [{v:read(d/v/'report.json') for v in p['variants']} for d,p in zip((first,second),plans)]
    for d,p,r in zip((first,second),plans,reports):
        validate(r,p,fingerprint(d/'plan.json'))
    budget_keys={'gpu_allocator_cap_gib','expert_ram_gib','process_ram_gib','native_cache_experts'}
    rows=[]
    for variant in plans[1]['variants']:
        a,b=reports[0][variant],reports[1][variant]
        for field in ('source_identity','corpus_identity','mapping_identity','checkpoint_sha256','mask_identity','gpu','torch'):
            if a.get(field)!=b.get(field):raise ValueError('Different '+field)
        if {k:v for k,v in a['protocol'].items() if k not in budget_keys}!={k:v for k,v in b['protocol'].items() if k not in budget_keys}:
            raise ValueError('Non-memory protocol differs')
        if [(r['id'],r['input_ids'],r['token_ids']) for r in a['records']] != [(r['id'],r['input_ids'],r['token_ids']) for r in b['records']]:
            raise ValueError('Generated tokens differ; cannot attribute timing solely to cache budgets')
        rows.append(dict(variant=variant,outputs_identical=True,first=aggregate(a),second=aggregate(b)))
    result=dict(completed=True,comparisons=rows,
                plans={str(d):fingerprint(d/'plan.json') for d in (first,second)},
                note='Same outputs; one-pass timings remain exploratory and include OS cache/activity differences.')
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    return result
