"""Plot completed OLMoE residency traces, without changing historical evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run / 'report.json').read_text(encoding='utf8'))
    if report['status'] != 'complete' or len(report['records']) != 9:
        raise ValueError('Requires all nine completed generations')
    records = report['records']
    cases = report['config']['cases']
    modes = report['config']['modes']
    rows = []
    for case in cases:
        for mode in modes:
            found = [r for r in records if r['category'] == case['category'] and r['mode'] == mode]
            if len(found) != 1:
                raise ValueError('Duplicate or missing record')
            r = found[0]
            events = r['events']
            if len(events) != len(r['token_ids']) or not events:
                raise ValueError('Trace length mismatch')
            if mode != 'native' and r['generation_h2d_bytes']:
                raise ValueError('Unexpected restricted transfer')
            changes = sum(set(x['routes'][layer][-1]) != set(y['routes'][layer][-1])
                          for x,y in zip(events, events[1:]) for layer in x['routes'])
            rows.append(dict(mode=mode, category=case['category'], tokens=len(r['token_ids']),
                             layer_set_changes=changes, preparation_h2d_mib=r['preparation_h2d_bytes']/2**20,
                             execution_h2d_mib=r['generation_h2d_bytes']/2**20,
                             response_seconds=r['generation_seconds']+r['preparation_seconds'],
                             peak_allocated_gib=r['peak_allocated_bytes']/2**30))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(11,6),layout='constrained',gridspec_kw={'height_ratios':[1.4,1]})
    labels = {'native':'Native routing / bounded cache', '32':'32 residents per layer', '16':'16 residents per layer'}
    selected = [next(r for r in records if r['mode']==mode and r['category']==cases[0]['category']) for mode in modes]
    max_steps=max(len(r['events']) for r in selected)
    for col,(mode,r) in enumerate(zip(modes,selected)):
        events=r['events']; layers=list(events[0]['routes'])
        for j in range(1,len(events)):
            for i,layer in enumerate(layers):
                before=set(events[j-1]['routes'][layer][-1]); after=set(events[j]['routes'][layer][-1])
                count=len(after-before)
                if count:
                    axes[0,col].scatter(j,i,s=8+8*count,c=[count],vmin=0,vmax=8,cmap='YlOrBr',marker='s')
        axes[0,col].set(title=labels[mode],xlabel='Decode forward step',yticks=[0,5,10,15],yticklabels=[1,6,11,16],xlim=(.5,max_steps-.5))
        axes[0,col].invert_yaxis()
        if col==0: axes[0,col].set_ylabel('MoE layer')
        increments=np.array([e['host_to_device_bytes']/2**30 for e in events])
        axes[1,col].plot(range(len(events)),np.cumsum(increments),color='#c76a28',lw=2,label='During execution')
        axes[1,col].set(xlabel='Forward step (0 = prefill)',xlim=(-.5,max_steps-.5))
        axes[1,col].text(.03,.92,f"Preparation: {r['preparation_h2d_bytes']/2**30:.2f} GiB",transform=axes[1,col].transAxes,va='top')
        axes[1,col].grid(axis='y',alpha=.15)
    max_h2d=max(r['generation_h2d_bytes']/2**30 for r in selected)
    for ax in axes[1]: ax.set_ylim(-.03*max_h2d,1.2*max_h2d)
    axes[1,0].set_ylabel('Cumulative expert H2D (GiB)')
    fig.suptitle('OLMoE: active expert choices can change without moving expert weights\nSquares: newly selected experts (darker/larger = more, 1–8); blank = unchanged',fontsize=11)
    fig.savefig(a.run/'olmoe_residency.pdf'); fig.savefig(a.run/'olmoe_residency.png',dpi=180)
    summary={'source_report_sha256':hashlib.sha256((a.run/'report.json').read_bytes()).hexdigest(),'rows':rows,
             'figure_prompt':cases[0]['prompt'], 'limitations':report['limitations']}
    (a.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    text=['# OLMoE: residencia por categoría', '', 'Diagnóstico de mecanismo; no demuestra calidad equivalente.', '',
          '| Categoría | Modo | Tokens | Cambios activos capa/paso | Precarga MiB | H2D ejecución MiB |',
          '|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['category']} | {r['mode']} | {r['tokens']} | {r['layer_set_changes']} | {r['preparation_h2d_mib']:.2f} | {r['execution_h2d_mib']:.2f} |")
    text+=['','Los cambios activos comparan el último token del prefill con el primer paso de decode, y después cada paso de decode. No cuentan los cambios internos entre tokens del prefill agrupado.','', 'La caché nativa tiene 512 slots globales. K32 tiene 512 residentes y K16 tiene 256. Los ocho activos se seleccionan dinámicamente entre los residentes; no son ocho expertos fijos.','', '## Salidas completas']
    for r in records:
        text+=['',f"### {r['category']} — {r['mode']}",r['prompt'],'',r['output']]
    (a.run/'SUMMARY.md').write_text('\n'.join(text),encoding='utf8')


if __name__=='__main__':
    main()
