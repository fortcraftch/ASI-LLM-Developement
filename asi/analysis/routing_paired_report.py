"""Summarize paired per-token traces without rerunning any model."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from asi.analysis.evidence import read_json, save_figure


def build(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    provenance={};data=read_json(source,provenance)
    if not data['completed']:raise ValueError('Routing run is incomplete')
    arms=['original','restricted','ce','ce_kl']
    ids=[r['id'] for r in data['examples']]
    records={(r['arm'],r['id']):r for r in data['records']}
    if len(records)!=len(arms)*len(ids):raise ValueError('Missing/duplicate trace')
    results={};matrix=[]
    for arm in arms:
        counts=np.zeros((16,64));rows=[];agreement=[];categories={}
        for id in ids:
            record=records[arm,id]
            for lid,d in record['layers'].items():
                a=np.asarray(d['indices']);b=np.asarray(records['original',id]['layers'][lid]['indices'])
                if a.shape!=b.shape:raise ValueError('Token trace shapes differ')
                overlap=(a[:,:,None]==b[:,None,:]).any(-1).mean(axis=1)
                agreement.extend(overlap.tolist());counts[int(lid)]+=d['counts']
                rows.append(d)
                bucket=categories.setdefault(record['category'],dict(tokens=0,outside=0.,mass=0.,overlap=0.))
                bucket['tokens']+=d['tokens'];bucket['outside']+=d['tokens']*d['outside_topk_fraction']
                bucket['mass']+=d['tokens']*d['outside_topk_mass'];bucket['overlap']+=float(overlap.sum())
        total=sum(d['tokens'] for d in rows)
        def avg(key):return sum(d[key]*d['tokens'] for d in rows)/total
        results[arm]=dict(mean_top8_overlap_original=float(np.mean(agreement)),
            outside_top8_fraction=avg('outside_topk_fraction'),outside_top8_mass=avg('outside_topk_mass'),
            outside_total_mass=avg('outside_total_mass'),executed_mass=avg('mean_executed_mass'),
            top8_margin=avg('mean_topk_margin'),categories={c:{k:v/b['tokens'] for k,v in b.items() if k!='tokens'} for c,b in categories.items()})
        matrix.append(counts/counts.sum(axis=1,keepdims=True))
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,arm,counts in zip(axes.flat,arms,matrix):
        im=ax.imshow(counts,aspect='auto',cmap='magma',vmin=0,vmax=max(x.max() for x in matrix))
        ax.set(title=arm,xlabel='Expert',ylabel='Layer')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Share of executed top-8 assignments')
    save_figure(fig,output,'paired_usage')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['mean_top8_overlap_original','outside_top8_fraction','executed_mass'],['Executed top-8 overlap with original','Own unrestricted top-8 outside resident set','Executed routing mass']):
        bars=ax.bar(arms,[results[a][key] for a in arms]);ax.bar_label(bars,fmt='%.3f');ax.set_title(title,fontsize=10);ax.set_ylim(0,1.05)
    fig.tight_layout();save_figure(fig,output,'paired_summary')
    summary=dict(completed=True,sources_sha256=provenance,rows=results,examples=data['examples'],
        limitation='Token-weighted across 6 diagnostic sequences and 16 layers. Same tokens, different downstream hidden states; not causal attribution or quality evaluation.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/routing_paired_20261009/report.json'))
    p.add_argument('--output',type=Path,default=Path('results/routing_paired_20261009/analysis'))
    args=p.parse_args();build(args.source,args.output)


if __name__=='__main__':main()
