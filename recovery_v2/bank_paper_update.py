"""Publish the audited bank evidence without rebuilding unrelated user tables."""
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'paper'
S=P/'sources/overleaf_bank_final_20261022'
R=ROOT/'results/v2/F0/bank_active_matched_20261022'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def score(d): return d.get('accuracy_norm',d['accuracy'])*100

def main():
    report=read(R/'report.json'); gen=read(R/'generation.json')
    assert report['status']==gen['status']=='complete'
    old=read(ROOT/'results/full_suite_20261012/comparison.json')['variants']
    bank={}
    for task,item in report['tasks'].items():
        path=ROOT/item['path']; assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        bank[task]=item['result']
        assert bank[task]['total']==old['ASI classified'][task]['total']
    comparisons={name:dict(wins=sum(score(bank[t])>score(v[t])+1e-9 for t in bank),
                          ties=sum(abs(score(bank[t])-score(v[t]))<1e-9 for t in bank)) for name,v in old.items()}
    # Retain every task label and the other columns' values; refresh ranking marks.
    suite=(S/'editorial_suite.tex').read_text(encoding='utf8')
    lines=[]
    for line in suite.splitlines():
        if ' & ' not in line: lines.append(line); continue
        cells=line.removesuffix('\\\\').split(' & ')
        task=cells[0].replace('\\_','_')
        values=[float(re.search(r'\d+\.\d+',c).group()) if re.search(r'\d+\.\d+',c) else None for c in cells[1:]]
        values[2]=round(score(bank[task]),2)
        ranks=sorted(set(x for x in values if x is not None),reverse=True)
        rendered=[]
        for v in values:
            if v is None: rendered.append('--'); continue
            text=f'{v:.2f}'
            if v==ranks[0]: text='\\textbf{'+text+'}'
            elif len(ranks)>1 and v==ranks[1]: text='\\underline{'+text+'}'
            rendered.append(text)
        lines.append(cells[0]+' & '+' & '.join(rendered)+'\\\\')
    (P/'editorial_suite.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    history=read(ROOT/'results/deepseek_19072_reference_20261010/retry_cache/generation/report.json')
    rows=[]
    for key,label in [('domain_selected','Categorized MoE'),('deepseek','Free MoE'),('gpt2','GPT-2 dense')]:
        model=next(m for m in history['models'] if m['variant']==key); runs=model['runs']
        rows.append(dict(label=label,params=model['parameters']/1e6,
            weights=max(sum(v for k,v in x['memory']['weight_bytes_by_device'].items() if k.startswith('cuda')) for x in runs)/2**20,
            peak=max(x['peak_cuda_allocated_bytes'] for x in runs)/2**20,
            rate=st.median(x['decode_tokens_per_second'] for x in runs),lo=min(x['decode_tokens_per_second'] for x in runs),hi=max(x['decode_tokens_per_second'] for x in runs),
            ttft=st.median(x['first_token_seconds'] for x in runs)*1000))
    pools=list(gen['pools'].values()); runs=[r for p in pools for r in p['runs']]
    b=dict(label='Bank (93.85M active)',params=750.800896,weights=375400448/2**20,
        peak=max(p['peak_cuda_allocated_bytes'] for p in pools)/2**20,rate=st.median(x['decode_tokens_per_second'] for x in runs),
        lo=min(x['decode_tokens_per_second'] for x in runs),hi=max(x['decode_tokens_per_second'] for x in runs),ttft=st.median(x['first_token_seconds'] for x in runs)*1000)
    rows.append(b)
    perf=(S/'editorial_performance.tex').read_text(encoding='utf8')
    perf=perf.replace('Independent bank & 123.97 & -- & -- & -- & --',f"Bank (93.85M active) & 750.80 & {b['weights']:.1f} & {b['peak']:.1f} & {b['rate']:.2f} & {b['ttft']:.1f}")
    (P/'editorial_performance.tex').write_text(perf,encoding='utf8')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    for ax,key,title,unit in zip(axes.flat,['weights','peak','rate','ttft'],['GPU model weights','Peak CUDA allocation','Decode throughput (excludes first token)','Time to first token'],['MiB','MiB','tokens/s','ms']):
        vals=[r[key] for r in rows]; bars=ax.bar(range(4),vals,color=['#0072b2','#009e73','#666666','#cc79a7'])
        ax.bar_label(bars,fmt='%.1f',padding=4); ax.set_ylim(0,max(max(vals),max(r['hi'] for r in rows) if key=='rate' else 0)*1.23)
        if key=='rate': ax.errorbar(range(4),vals,yerr=[[r['rate']-r['lo'] for r in rows],[r['hi']-r['rate'] for r in rows]],fmt='none',color='black',capsize=3)
        ax.set_xticks(range(4),['Categorized\nMoE','Free MoE','GPT-2\ndense','Bank\n93.85M active'],fontsize=9)
        ax.set_title(title,fontsize=11); ax.set_ylabel(unit); ax.set_axisbelow(True); ax.grid(axis='y',alpha=.15)
    for ext in ['png','pdf']: fig.savefig(P/'figures'/('editorial_performance.'+ext),dpi=180)
    plt.close(fig)
    text=(S/'asi_paper.tex').read_text(encoding='utf8')
    def replace(a,b):
        nonlocal text
        assert a in text,a
        text=text.replace(a,b)
    replace('TTFT is reported separately. Bank measurements under this generation protocol\nare unavailable (dashes); its aggregate parameter count is shown for context.',
        'TTFT is reported separately. Bank Params M is the aggregate of eight specialists;\nonly one 93.85M model resides on GPU and generates each response. Others remain\non disk in this generation measurement. The external classifier is excluded.')
    replace('confidence intervals. Only the principal model families are shown; the bank\nhas no matched generation run and is explicitly marked as unmeasured.',
        'confidence intervals. The bank uses the same eight prefixes and timing protocol,\nmeasured in a later session; desktop load and runtime drift remain potential confounds.')
    start=text.index('The original bank matches aggregate storage')
    end=text.index('\\subsubsection{Complete suite',start)
    text=text[:start]+r'''The completed bank contains eight category-trained dense specialists with
93,850,112 parameters each (750,800,896 in total). One specialist answers each
input. All eight final checkpoints match the frozen recipes, planned update
counts and finite-weight checks. This matches ASI's participating parameter
inventory after excluding bypassed gates, including full embedding tables;
it does not match FLOPs, total storage or shared training exposure. Each dense
backbone sees its own category, while ASI's shared backbone sees all categories.

The bank completes 66,832 examples across 19 tasks. Relative to the historical
15.50M-per-specialist bank, it improves 15 tasks, decreases two (WSC and
TruthfulQA) and ties on WinoGrande and arithmetic. LAMBADA rises from 4.91 to 15.66\%, and
BoolQ from 39.45 to 52.63\%. This is descriptive evidence from single trained
banks, not an isolated causal estimate of parameter scaling.

The historical bank remains a separate aggregate-storage control: eight
15,496,576-parameter models (123,972,608 total). Its earlier placement experiment
reduced GPU weights from 472.92 to 59.11 MiB and peak allocation from 1432.39 to
948.76 MiB when selecting one model. These figures belong exclusively to that
smaller bank; they are not measurements of the new bank. Historical scores and
placement reports remain archived in the repository.

The new generation measurement gives 28.98 decode tokens/s and 35.25 ms median
TTFT. A selected model stores 358.01 MiB of parameters on GPU; generation has
no weight transfers after preparation. This excludes model loading and input
classification and does not measure a conversational cache policy.

'''+text[end:]
    replace('The bank uses one resident model with BF16 batches 4/512.',
        'Bank denotes the new 93.85M-active specialist bank. For these quality benchmarks,\nall eight models reside on GPU but only one answers each input. BF16 batches\nare 2/512 versus 4/512 historically; numerical batching is not identical.')
    text=text.replace("The bank's two complete evaluation blocks", "The historical small bank's two complete evaluation blocks")
    anchor='\\subsubsection{Supervised adaptation: separate comparison}'
    comparison='Against ASI, the new bank scores higher on nine tasks, lower on nine and ties\non arithmetic. Against DeepSeek it scores higher on five, lower on eleven and\nties on ARC-Challenge, WSC and arithmetic. Its LAMBADA score (15.66\\%) remains below\nASI (21.89\\%) and DeepSeek (24.86\\%). Similar active capacity therefore does\nnot imply equivalent quality; domain-only backbone training and differing\narchitectures remain confounded.\n\n'
    text=text.replace(anchor,comparison+anchor)
    trace='\\code{v2/F0/bank\\_active\\_matched\\_20261022}: audited 93.85M specialists, full suite and generation; '
    text=text.replace('E1 & Local architectures & ', 'E1 & Local architectures & '+trace)
    text=text.replace('Figure~\\ref{fig:cache-budget} holds', 'The paired cache control holds')
    text=text.replace('evidence snapshot: 18 October 2026', 'evidence snapshot: 22 October 2026')
    (P/'asi_paper.tex').write_text(text,encoding='utf8')
    (R/'paper_summary.json').write_text(json.dumps(dict(comparisons=comparisons,generation=rows,source_snapshot=str(S)),indent=2),encoding='utf8')
    print(json.dumps(dict(comparisons=comparisons,bank_generation=b),indent=2))

if __name__=='__main__': main()
