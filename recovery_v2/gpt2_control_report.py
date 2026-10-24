import json,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/v2/F0/gpt2_attention_control_20261023');p=root/'report.json';r=json.loads(p.read_text());assert r['status']=='complete' and len(r['results'])==8
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('recovery_v2/gpt2_attention_control.py')==r['settings']['script_sha256']
for path,h in r['settings']['source_hashes'].items():assert sha(path)==h
assert sha('../Creating-GPT-2-From-0/eval.py')==r['settings']['evaluator_sha256']
rng=np.random.default_rng(1337);ix=rng.integers(0,8,(10000,8));stats={}
for name in ['baseline','math_attention']:
 ms=[v['modes'][name] for v in r['results'].values()];assert all(len(m['runs'])==3 for m in ms)
 a=np.array([[z['decode_tokens_per_second'] for z in m['runs']] for m in ms]);ci=np.quantile(np.median(a[ix],axis=(1,2)),[.025,.975])
 runs=[z for m in ms for z in m['runs']];assert all(z['generated_tokens']==64 for z in runs)
 stats[name]=dict(rate=float(np.median(a)),ci95=ci.tolist(),ttft=float(np.median([z['first_token_seconds'] for z in runs])*1000),peak=max(z['peak_cuda_allocated_bytes'] for z in runs)/2**20,weights=sum(runs[0]['memory']['weight_bytes_by_device'].values())/2**20,fixed_identical=sum(m['fixed_tokens_identical'] for m in ms),eos_identical=sum(m['eos_tokens_identical'] for m in ms),fp32_allclose=sum(m['logit_checks']['fp32']['allclose'] for m in ms),rss_peak=max(z['sampled_peaks']['rss_bytes'] for z in runs),nvml_available=all(z['sampled_peaks']['nvml_process_bytes'] is not None for z in runs))
(root/'summary.json').write_text(json.dumps(dict(status='complete',input_sha256=sha(p),bootstrap='10000 prompt-cluster resamples, seed1337',metrics=stats),indent=2))
rows=json.loads(Path('results/v2/F0/bank_active_matched_20261022/paper_summary.json').read_text())['generation']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
for ax,key,title,unit in zip(axes.flat,['weights','peak','rate','ttft'],['GPU model weights','Peak CUDA allocation','Decode throughput (excludes first token)','Time to first token'],['MiB','MiB','tokens/s','ms']):
 vals=[x[key] for x in rows];bars=ax.bar(range(4),vals,color=['#0072b2','#009e73','#666666','#cc79a7'],alpha=.85);ax.bar_label(bars,fmt='%.1f',padding=3)
 if key=='rate': ax.errorbar(range(4),vals,yerr=[[x['rate']-x['lo'] for x in rows],[x['hi']-x['rate'] for x in rows]],fmt='none',color='black',capsize=3)
 for name,marker,color,label,offset in [('baseline','o','#202020','GPT-2 paired SDPA auto',-.13),('math_attention','D','#bd5d00','GPT-2 paired MATH (unfused)',.13)]:
  val=stats[name][key];ax.scatter(2+offset,val,marker=marker,color=color,s=40,zorder=5,label=label)
  if key in ['rate','ttft']:ax.annotate(f'{val:.1f}',(2+offset,val),xytext=(14 if offset>0 else -30,-13),textcoords='offset points',color=color,fontsize=9)
 ax.set_xticks(range(4),['Categorized\nMoE','Free MoE','GPT-2\ndense','Bank\n93.85M active']);ax.set_ylabel(unit);ax.set_title(title);ax.set_ylim(0,max(vals)*1.27);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
for ext in ['pdf','png']:fig.savefig('paper/figures/editorial_performance.'+ext,dpi=180)
src=Path('paper/sources/overleaf_gptmath_20261023/asi_paper.tex');s=src.read_text()
anchor='\\paragraph{Dense inference implementation control.}'
note=r'''\paragraph{GPT-2 attention backend control.}
The added markers in the memory/throughput overview compare GPT-2 SDPA automatic
selection with the unfused MATH backend in a paired follow-up session, using the
same eight prefixes, 64 tokens, three repetitions, FP32 weights and BF16 autocast.
Median decode throughput is 85.58 versus 53.77 tokens/s (descriptive 95\%
prompt-bootstrap intervals 83.74--87.77 and 52.64--54.38); TTFT is 11.85 versus
18.64 ms. Peak allocated CUDA memory is 573.83 MiB in both follow-up arms.
The original bars and table retain their historical measurements; markers are
new observations, not a simultaneous rerun of all four families. Forcing MATH
does not reproduce MLA or equalize architectural work. FP32 logit checks pass
on all eight prompts, but fixed-length and EOS-aware tokens agree with automatic
SDPA in only five of eight prompts under BF16. This is a numerical/backend
intervention, not an output-equivalent optimization or a quality comparison.
Evidence is archived in \code{results/v2/F0/gpt2\_attention\_control\_20261023}.

'''
assert anchor in s;s=s.replace(anchor,note+anchor)
s=s.replace('Local memory and generation measurements under one protocol. Error bars','Local memory and generation measurements under one protocol. Bars retain the\nhistorical results; circle/diamond markers show the paired GPT-2 backend control\ndescribed below. Error bars')
pat=r'\\begin\{tabular\}.*?\\end\{tabular\}';assert re.findall(pat,s,re.S)==re.findall(pat,src.read_text(),re.S)
Path('paper/asi_paper.tex').write_text(s)
