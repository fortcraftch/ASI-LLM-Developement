from pathlib import Path
import re,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; src=root/'paper/sources/overleaf_benchmark_overview_20261024'; paper=root/'paper'
raw=(src/'editorial_suite.tex').read_text()
rows=[]
for line in raw.splitlines():
 if ' & ' not in line: continue
 cells=line.split(' & ')
 vals=[float(re.search(r'\d+\.\d+',c).group()) for c in cells[1:]]
 rows.append((cells[0],vals))
assert len(rows)==18
v=np.array([r[1] for r in rows]); delta=v[:,[0]]-v[:,1:]
fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
labels=['Free MoE','Bank (93.85M active)','GPT-2 historical*']; result={}
for i,label in enumerate(labels):
 d=delta[:,i]; wins=int((d>0).sum()); ties=int((d==0).sum()); losses=int((d<0).sum())
 # Displace overlapping scores vertically, keeping score coordinates exact.
 offsets=[]; used=[]
 for x in d:
  count=sum(abs(x-z)<.45 for z in used); used.append(x); offsets.append([0,.10,-.10,.20,-.20][count%5])
 ax.scatter(d,i+np.array(offsets),c=['#008678' if x>0 else '#bf6d45' if x<0 else '#777777' for x in d],s=37,alpha=.8,edgecolors='white',linewidth=.4,zorder=3)
 med=float(np.median(d)); ax.scatter([med],[i],marker='D',s=75,facecolors='none',edgecolors='black',linewidth=1.4,zorder=5)
 ax.text(12.1,i,f'{wins} / {ties} / {losses}',va='center',fontsize=10)
 result[label]={'higher':wins,'tied':ties,'lower':losses,'median_difference_pp':med,'differences_pp':dict(zip([r[0] for r in rows],d.tolist()))}
ax.axvline(0,color='#555555',lw=1); ax.set_yticks(range(3),labels); ax.invert_yaxis(); ax.set_ylim(2.55,-.65); ax.set_xlim(-9,15)
ax.set_xticks([-8,-4,0,4,8,12]); ax.set_xlabel('ASI minus reference accuracy (percentage points)')
ax.set_title('Quality overview: 18 common benchmark tasks',pad=25)
ax.text(-4.5,-.48,'Reference scores higher',ha='center',color='#a25330',fontsize=9)
ax.text(6,-.48,'ASI scores higher',ha='center',color='#007668',fontsize=9)
ax.text(12.1,-.48,'Higher / tie / lower',fontsize=8)
ax.set_axisbelow(True); ax.grid(axis='x',alpha=.15)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.supxlabel('Each circle = one task; outlined diamond = median task difference.\n* Historical GPT-2 protocol equivalence is unverified; arithmetic excluded for all.',fontsize=9)
for ext in ['pdf','png']: fig.savefig(paper/f'figures/asi_benchmark_overview.{ext}',dpi=200)
t=(src/'asi_paper.tex').read_text(); anchor='\\label{tab:full-suite-models}\n\\end{table}'
assert t.count(anchor)==1
addition=r'''

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{asi_benchmark_overview.pdf}
\caption{Descriptive quality overview of the 18 tasks shared by all four
columns of Table~\ref{tab:full-suite-models}. Each circle is ASI accuracy minus
the reference, in percentage points; the outlined diamond marks the median
across tasks. Counts indicate higher, tied and lower displayed scores.
Arithmetic is excluded uniformly because GPT-2 lacks that result. Tasks receive
equal visual weight despite differing sizes and metrics; this is not a combined
accuracy estimate or a significance test. Historical GPT-2 provenance and bank
batching limitations from the table also apply here.}
\label{fig:benchmark-overview}\end{figure}

Figure~\ref{fig:benchmark-overview} separates execution gains from task quality.
ASI scores higher on 5 of 18 shared tasks against the free MoE, 9 against the
bank, and 3 against historical GPT-2 (with one displayed tie). Median task
differences are -0.60, +0.07 and -1.12 percentage points, respectively.
The pattern is mixed rather than a uniform quality gain. These task scores do
not establish better coherence, repetition control or instruction following in
open-ended generation; those require the separate generation evaluations.
'''
t=t.replace(anchor,anchor+addition)
t=t.replace(r'Table~\ref{tab:localperformance} and Figure~\ref{fig:localperformance} summarize',r'Table~\ref{tab:localperformance} summarizes')
assert re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',t,re.S)==re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',(src/'asi_paper.tex').read_text(),re.S)
(paper/'asi_paper.tex').write_text(t)
out=root/'results/v2/F0/benchmark_overview_20261024'; out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'scores_from_displayed_table':True,'comparisons':result},indent=2))
print(json.dumps({k:{a:b for a,b in val.items() if a!='differences_pp'} for k,val in result.items()},indent=2))

