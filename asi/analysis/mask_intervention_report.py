"""Summarize measured mask interventions without loading model weights."""
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
    out = source.parent / 'analysis'
    out.mkdir(exist_ok=True)
    expected = {r['id'] for r in report['evaluation_examples']}
    summary = {'source_sha256': sha256(source), 'completed': report['completed'], 'variants': {}}
    sections = []
    for name in report.get('masks', {}):
        rows = [r for r in report['records'] if r['variant'] == name]
        if {r['id'] for r in rows} != expected or len(rows) != len(expected):
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
        fig.suptitle('Frozen base OLMoE — exploratory six-prompt diagnostic')
        fig.tight_layout()
        save_figure(fig, out, 'mask_comparison')
        plt.close(fig)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mask interventions</title>'
        '<style>body{max-width:1100px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap}img{width:100%}</style>'
        '<h1>Frozen OLMoE mask interventions</h1><p>Only complete arms are compared. Six prompts do not establish general quality. '
        'Latency excludes context preparation; weights are frozen. Repetition is not a correctness score.</p>'
        + ('<img src="mask_comparison.png">' if names else '<p>No complete arms yet.</p>')
        + ''.join(sections), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    print(json.dumps(build(p.parse_args().report), indent=2))
