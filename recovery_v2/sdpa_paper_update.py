"""Publish derived SDPA table/relative figure from a preserved Overleaf snapshot."""
from pathlib import Path
import json, re, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
root=Path(__file__).resolve().parents[1]
src=root/'paper/sources/overleaf_benefits_fresh_20261024'
out=root/'paper'
r=root/'results/v2/F0/sdpa_four_models_20261023'
s=json.loads((r/'summary.json').read_text()); m=s['metrics']
assert s['status']=='complete'
table=(src/'editorial_performance.tex').read_text()
for name,k in [('Categorized MoE','asi'),('Free MoE','deepseek'),('GPT-2 dense','gpt2'),('Bank (93.85M active)','bank')]:
    old=next(x for x in table.splitlines() if x.startswith(name+' &'))
    prefix=old.split(' &')[:2]
    v=m[k]
    new=' &'.join(prefix)+f" & {v['gpu_weights_mib']:.1f} & {v['cuda_peak_mib']:.1f} & {v['decode_tokens_s']:.2f} & {v['ttft_ms']:.1f}"+r'\\'
    table=table.replace(old,new)
(out/'editorial_performance.tex').write_text(table)
refs=['deepseek','gpt2','bank']; labels=['Free MoE','GPT-2 dense','Bank']
metrics=[('gpu_weights_mib','GPU weight reduction'),('cuda_peak_mib','Peak CUDA reduction'),('decode_tokens_s','Decode throughput gain'),('ttft_ms','Time-to-first-token reduction')]
fig,axs=plt.subplots(2,2,figsize=(8.5,4.8),layout='constrained')
derived={}
for ax,(key,title) in zip(axs.flat,metrics):
    vals=[100*(m['asi'][key]/m[k][key]-1)*(1 if key=='decode_tokens_s' else -1) for k in refs]
    derived[key]=dict(zip(refs,vals))
    ax.barh(labels,vals,color=['#008678' if v>=0 else '#be6b40' for v in vals],height=.52)
    ax.axvline(0,color='#555555',lw=.8); ax.invert_yaxis()
    ax.set_title(title,fontsize=11,pad=10); ax.set_xlabel('Relative change (%)',fontsize=9)
    ax.set_axisbelow(True); ax.grid(axis='x',alpha=.15)
    for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(v,i),xytext=(4 if v>=0 else -4,0),textcoords='offset points',va='center',ha='left' if v>=0 else 'right',fontsize=9,color='black')
    low=min(0,min(vals)); high=max(0,max(vals)); span=max(high-low,1)
    ax.set_xlim(low-.55*span,high+.32*span)
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y',length=0,labelsize=10)
fig.suptitle('ASI relative to each reference under the SDPA protocol',fontsize=13)
fig.legend(handles=[Patch(color='#008678',label='Favors ASI'),Patch(color='#be6b40',label='Favors reference')],loc='outside lower center',ncol=2,frameon=False)
for ext in ['pdf','png']: fig.savefig(out/f'figures/asi_sdpa_tradeoffs.{ext}',dpi=200)
text=(src/'asi_paper.tex').read_text()
text=text.replace('recomputes the full prefix and projects logits for all its positions: no','recomputes the full prefix, uses SDPA and projects only the last-position logits: no')
text=text.replace('Matched full-context generation of the local models. GPU MiB counts','Matched SDPA full-context generation of the local models. GPU MiB counts')
text=text.replace(' Table~1 retains the historical native measurements.',' Table~\\ref{tab:localperformance} reports the same measurements.')
text=text.replace('The new generation measurement gives 28.98 decode tokens/s and 35.25 ms median','The SDPA generation measurement gives 40.20 decode tokens/s and 24.88 ms median')
anchor=r'\label{fig:localperformance}\end{figure}'
addition=r'''

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{asi_sdpa_tradeoffs.pdf}
\caption{Relative advantages and costs of ASI under the same SDPA measurement
protocol as Table~\ref{tab:localperformance}. Positive percentages favor ASI:
$(1-\mathrm{ASI}/\mathrm{reference})\times100$ for memory and latency, and
$(\mathrm{ASI}/\mathrm{reference}-1)\times100$ for throughput. Values are
ratios of the reported summaries, not quality or significance estimates.}
\label{fig:asi-tradeoffs}\end{figure}

Figure~\ref{fig:asi-tradeoffs} makes the tradeoff explicit. Compared with the
free MoE, ASI uses 24.0\% less GPU parameter memory and 19.8\% less peak CUDA
memory, while decoding 2.68 times faster and reducing TTFT by 57.5\%.
Compared with GPT-2 and the bank, its peak allocation is lower by 19.3\% and
5.6\%, respectively, but decoding is slower. Its parameter residency is
essentially equal to the selected bank model (0.1\% higher). These descriptive
comparisons include architecture and implementation differences; they do not
isolate a causal benefit of categorization or imply equal response quality.
'''
assert text.count(anchor)==1
text=text.replace(anchor,anchor+addition)
# All inline tables, including user formatting, must be byte-for-byte unchanged.
pattern=r'\\begin\{tabular\}.*?\\end\{tabular\}'
assert re.findall(pattern,text,re.S)==re.findall(pattern,(src/'asi_paper.tex').read_text(),re.S)
(out/'asi_paper.tex').write_text(text)
(r/'asi_tradeoffs.json').write_text(json.dumps({'source_sha256':hashlib.sha256((r/'summary.json').read_bytes()).hexdigest(),'relative_percent':derived,'publication_authorized':True,'manual_inline_tables_unchanged':True},indent=2))
print(table)


