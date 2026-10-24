"""Summarize measured mask/scale interventions without loading model weights."""
import argparse
import html
import json
from pathlib import Path
import statistics

from asi.analysis.evidence import save_figure
from asi.experiments.public_moe import sha256
from asi.experiments.recovery_protocol import durable_save


def build(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(source.read_text(encoding='utf-8'))
    if 'decoding_plan' in report:
        return build_decoding(source, report)
    out = source.parent / 'analysis'
    out.mkdir(exist_ok=True)
    scale = 'evaluation_examples' not in report and 'examples' in report
    if scale:
        from asi.experiments.routing_scale import ARMS
        variants = ARMS
        expected = {r['id'] for r in report['examples']}
        variant_key = 'arm'
        figure_name = 'routing_scale_comparison'
        title = 'Frozen OLMoE — nine-question development scale pilot'
    else:
        variants = list(report.get('masks', {}))
        expected = {r['id'] for r in report['evaluation_examples']}
        variant_key = 'variant'
        figure_name = 'mask_comparison'
        title = 'Frozen base OLMoE — exploratory six-prompt diagnostic'
    summary = {'source_sha256': sha256(source), 'completed': report['completed'], 'variants': {}}
    sections = []
    for name in variants:
        rows = [r for r in report['records'] if r[variant_key] == name]
        if {r['id'] for r in rows} != expected or len(rows) != len(expected):
            if report['completed']:
                raise ValueError('Completed report has missing/duplicated examples: ' + name)
            continue  # Never compare complete arms with partial arms.
        speeds = [r['decode_tokens_per_second'] for r in rows if r['decode_tokens_per_second'] is not None]
        summary['variants'][name] = {
            'examples': len(rows), 'mean_ce': statistics.mean(r['ce'] for r in rows),
            'eos_fraction': statistics.mean(r['eos'] for r in rows),
            'mean_repeated_trigram_fraction': statistics.mean(r['metrics']['repeated_trigram_fraction'] for r in rows),
            'median_response_seconds': statistics.median(r['generation_seconds'] for r in rows),
            'median_decode_tokens_per_second': statistics.median(speeds) if speeds else None,
            'peak_allocated_gib': max(r['peak_cuda_allocated_bytes'] for r in rows) / 2**30,
            'resident_expert_gib': max(r['ready']['resident_expert_bytes'] for r in rows) / 2**30,
            'preparation_h2d_gib': sum(r['ready'].get('host_to_device_bytes', 0) - r['before'].get('host_to_device_bytes', 0) for r in rows) / 2**30,
            'inference_h2d_gib': sum(r['after'].get('host_to_device_bytes', 0) - r['ready'].get('host_to_device_bytes', 0) for r in rows) / 2**30,
            'inference_loads': sum(r['after'].get('loads', 0) - r['ready'].get('loads', 0) for r in rows)}
        for r in rows:
            sections.append('<h2>' + html.escape(name + ' / ' + r['id']) + '</h2><pre>' + html.escape(r['text']) + '</pre>')
    durable_save(out / 'summary.json', summary)
    names = list(summary['variants'])
    if names:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        metrics = [('mean_ce', 'Reference CE (lower is better)'),
                   ('mean_repeated_trigram_fraction', 'Repeated trigrams'),
                   ('eos_fraction', 'EOS fraction'),
                   ('median_decode_tokens_per_second', 'Median decode tokens/s'),
                   ('peak_allocated_gib', 'Peak allocated VRAM (GiB)'),
                   ('resident_expert_gib', 'Resident expert weights (GiB)')]
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.bar(names, [summary['variants'][n][key] if summary['variants'][n][key] is not None else float('nan') for n in names])
            ax.set_title(label)
            ax.tick_params(axis='x', rotation=70, labelsize=8)
        fig.suptitle(title)
        if scale:
            fig.text(.5, .01, 'Original uses a swapping cache; speeds are descriptive. Lower repetition does not imply correct or coherent text.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, .96))
        else:
            fig.tight_layout()
        save_figure(fig, out, figure_name)
        plt.close(fig)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mask interventions</title>'
        '<style>body{max-width:1100px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap}img{width:100%}</style>'
        '<h1>' + html.escape(title) + '</h1><p>Only complete arms are compared. This small exploratory sample does not establish general quality. '
        'Latency excludes context preparation; weights are frozen. Repetition is not a correctness score.</p>'
        + ('<p>Original uses a capacity-limited cache and can swap. Fixed execution order, variable EOS lengths and no timing repeats: speeds are descriptive, not an all-resident baseline comparison. See the frozen plan for limitations.</p>' if scale else '')
        + ('<img src="'+figure_name+'.png">' if names else '<p>No complete arms yet.</p>')
        + ''.join(sections), encoding='utf-8')
    return summary


def build_decoding(source, report):
    """Keep sampling seeds nested within prompts, and never infer correctness."""
    import math
    plan = report['decoding_plan']
    ids = [r['id'] for r in report['examples']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate decoding questions')
    expected = {(arm, i, p['name'], seed) for arm in plan['arms'] for i in ids
                for p in plan['policies'] for seed in p['seeds']}
    rows = report['records']
    keys = [(r['arm'],r['id'],r['decoding_policy'],r['sampling_seed']) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError('Unexpected or duplicate decoding observations')
    if not report.get('completed') or set(keys) != expected:
        raise ValueError('Decoding comparison requires complete paired coverage')
    for i in ids:
        tokens = [r['input_ids'] for r in rows if r['id'] == i]
        if any(t != tokens[0] for t in tokens):
            raise ValueError('Decoding prompt mismatch')
    for r in rows:
        if not r['token_ids'] or len(r['token_ids']) > plan['protocol']['new_tokens']:
            raise ValueError('Invalid generated length')
        repetition = r['metrics']['repeated_trigram_fraction']
        if not math.isfinite(repetition) or not 0 <= repetition <= 1:
            raise ValueError('Invalid repetition fraction')
        if not isinstance(r['eos'], bool):
            raise ValueError('Invalid EOS observation')
        if r['arm'] == 'restricted' and any(r['ready'][k] != r['after'][k]
                                            for k in ('loads','host_to_device_bytes')):
            raise ValueError('Expert transfers inside restricted generation')
    summary = {'source_sha256':sha256(source), 'completed':True,
               'independent_questions':len(ids), 'responses':len(rows), 'groups':[],
               'limitations':plan['limitations'], 'quality_review_pending':True}
    for arm in plan['arms']:
        for policy in plan['policies']:
            group = [r for r in rows if r['arm']==arm and r['decoding_policy']==policy['name']]
            per_prompt = []
            for i in ids:
                examples = [r for r in group if r['id']==i]
                per_prompt.append({'id':i, 'mean_repeated_trigram_fraction':statistics.mean(
                    r['metrics']['repeated_trigram_fraction'] for r in examples),
                    'mean_generated_tokens':statistics.mean(len(r['token_ids']) for r in examples),
                    'budget_exhaustion_fraction':statistics.mean(not r['eos'] and
                        len(r['token_ids']) == plan['protocol']['new_tokens'] for r in examples),
                    'eos_fraction':statistics.mean(r['eos'] for r in examples)})
            summary['groups'].append({'arm':arm,'policy':policy['name'],'responses':len(group),
                'per_prompt':per_prompt,'mean_repeated_trigram_fraction':statistics.mean(
                    r['mean_repeated_trigram_fraction'] for r in per_prompt),
                'mean_generated_tokens':statistics.mean(r['mean_generated_tokens'] for r in per_prompt),
                'budget_exhaustion_fraction':statistics.mean(r['budget_exhaustion_fraction'] for r in per_prompt),
                'eos_fraction':statistics.mean(r['eos_fraction'] for r in per_prompt)})
    out = source.parent/'analysis'
    out.mkdir(exist_ok=True)
    durable_save(out/'summary.json', summary)
    sections = ['<!doctype html><meta charset="utf-8"><title>Decoding control</title>',
                '<style>body{max-width:1000px;margin:2em auto}pre{white-space:pre-wrap}</style>',
                '<h1>Development decoding control</h1><p>Seeds share questions. Repetition and EOS are not correctness. No generated code executed.</p>']
    for r in rows:
        sections.append('<h2>'+html.escape(f"{r['id']} / {r['arm']} / {r['decoding_policy']} / {r['sampling_seed']}")+'</h2><pre>'+html.escape(r['text'])+'</pre>')
    (out/'index.html').write_text(''.join(sections),encoding='utf-8')
    return summary


def compare_selection(reference, candidate):
    """Compare complete runs on identical diagnostic tokens; never pool timings."""
    import math
    reports = [json.loads(Path(p).read_text(encoding='utf-8')) for p in (reference, candidate)]
    base, new = reports
    if not all(r.get('completed') for r in reports):
        raise ValueError('Both selection reports must be complete')
    for key in ('source', 'files'):
        if base['identity'][key] != new['identity'][key]:
            raise ValueError('Selection report identity mismatch: ' + key)
    for key in ('new_tokens', 'active', 'max_length', 'batch_size'):
        if base['identity']['protocol'][key] != new['identity']['protocol'][key]:
            raise ValueError('Selection protocol differs: ' + key)
    if base['evaluation_examples'] != new['evaluation_examples']:
        raise ValueError('Diagnostic tokens or references differ')
    expected = {r['id'] for r in base['evaluation_examples']}
    if not expected or len(expected) != len(base['evaluation_examples']):
        raise ValueError('Invalid diagnostic identities')
    rows = []
    for report in reports:
        if set(r['variant'] for r in report['records']) != set(report['masks']):
            raise ValueError('Unexpected or missing variants')
        for name, categories in report['masks'].items():
            records = [r for r in report['records'] if r['variant'] == name]
            if len(records) != len(expected) or {r['id'] for r in records} != expected:
                raise ValueError('Incomplete or duplicate variant records')
            budgets = {sum(len(v['experts']) for v in layers.values()) for layers in categories.values()}
            if len(budgets) != 1:
                raise ValueError('Category budgets differ')
            for r in records:
                if not math.isfinite(r['ce']): raise ValueError('Nonfinite CE')
                for key in ('loads', 'host_to_device_bytes'):
                    if r['after'][key] != r['ready'][key]:
                        raise ValueError('Transfers inside restricted inference')
            rows.append({'variant': name, 'resident_pairs': budgets.pop(),
                         'mean_ce': statistics.mean(r['ce'] for r in records),
                         'eos_count': sum(bool(r['eos']) for r in records),
                         'examples': len(records),
                         'per_example': [{'id':r['id'], 'ce':r['ce']} for r in records]})
    if len({r['variant'] for r in rows}) != len(rows):
        raise ValueError('Duplicate variant names across reports')
    return {'sources_sha256': {str(p):sha256(Path(p)) for p in (reference,candidate)},
            'rows':rows, 'timing_comparable':False, 'independent_test':False,
            'limitations':['Compare selection only within equal resident budgets.',
                           'Historical runtime fingerprint incomplete; no timing speedup claim.',
                           'Seeds share prompts; 18 generations are not 18 independent questions.',
                           'CE and EOS do not establish answer correctness; inspect full outputs.']}


def selection_figure(comparison, output):
    """Show matched-budget diagnostic CE without pooling seeds as new questions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in comparison['rows'] if r['resident_pairs'] == 512]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, row in enumerate(rows):
        color = '#777777' if row['variant'].startswith('random32_') else '#2878a0'
        axes[0].barh(i, row['mean_ce'], color=color)
        axes[0].text(row['mean_ce'] + .025, i, f"{row['mean_ce']:.3f}", va='center', fontsize=9)
        axes[1].scatter([p['ce'] for p in row['per_example']], [i]*row['examples'], color=color, alpha=.7)
    for ax in axes:
        ax.set_yticks(range(len(rows)), [r['variant'] for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Reference cross-entropy (lower is better)')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_title('Mean over the same six prompts')
    axes[0].set_xlim(0, max(r['mean_ce'] for r in rows)*1.18)
    axes[1].set_title('Individual prompts; points are not confidence intervals')
    fig.suptitle('Frozen OLMoE: 512 resident layer/expert pairs, top-8 activation')
    fig.text(.5, .015, 'Exploratory diagnostic; three random masks share six questions. CE is not generation correctness.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    save_figure(fig, output, 'selection_control')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--compare-with', type=Path, help='Completed historical selection report')
    p.add_argument('--comparison-output', type=Path, help='Save validated comparison JSON and matched-budget figure')
    args = p.parse_args()
    result = compare_selection(args.compare_with, args.report) if args.compare_with else build(args.report)
    if args.comparison_output:
        if not args.compare_with:
            p.error('--comparison-output requires --compare-with')
        args.comparison_output.mkdir(parents=True, exist_ok=True)
        durable_save(args.comparison_output / 'comparison.json', result)
        selection_figure(result, args.comparison_output)
    print(json.dumps(result, indent=2))
