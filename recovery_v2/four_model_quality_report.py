"""Summarize a completed generation diagnostic; bootstrap whole prompts."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from asi.experiments.recovery_protocol import durable_save
root=Path('results/v2/F0/four_model_quality_20261024')
r=json.loads((root/'report.json').read_text()); assert r['status']=='complete'
qs=r['config']['prompts']; families=['asi','deepseek','bank','gpt2']; names=['ASI','Free MoE','Bank','GPT-2']
rng=np.random.default_rng(1337); indices=rng.integers(0,len(qs),(10000,len(qs)))
s={}; md=['# Matched base-model generation diagnostic','',r['config']['protocol'],'','Exploratory, unblinded analysis. The automatic format check is a proxy, not a coherence score. Repeated word four-grams are measured only in the answer before the first EOS. No generated code is executed.','']
for f in families:
 rows=[r['results'][f+'/'+q['id']] for q in qs]
 metrics={}
 for metric in ['format_or_exact_proxy_pass','eos','limit_reached','repeated_word_4gram_fraction']:
  arr=np.array([x['automatic'][metric] if x['automatic'][metric] is not None else np.nan for x in rows],float)
  with np.errstate(invalid='ignore'): dist=np.nanmean(arr[indices],axis=1)
  metrics[metric]={'mean':float(np.nanmean(arr)),'ci95':np.nanquantile(dist,[.025,.975]).tolist(),'valid_prompts':int(np.isfinite(arr).sum())}
 s[f]=metrics
md+=['## Automatic overview','', '| Model | Format/exact proxy | EOS | Mean repeated word 4-grams |','|---|---:|---:|---:|']
for f,n in zip(families,names):
 m=s[f]; md.append(f"| {n} | {round(m['format_or_exact_proxy_pass']['mean']*16)}/16 | {round(m['eos']['mean']*16)}/16 | {m['repeated_word_4gram_fraction']['mean']*100:.1f}% |")
md+=['','These are base models, with no SFT adaptation in this comparison. EOS/short answers are not evidence of correctness. Missing repetition values for answers shorter than four words are excluded rather than treated as zero.','']
for q in qs:
 md += [f"## {q['id']}",'',q['prompt'],'',f"Expected content: {q['reference']}",'']
 for f,n in zip(families,names):
  x=r['results'][f+'/'+q['id']]; md += [f'### {n}', '',f"Stop: {x['stop_reason']}; tokens: {x['generated_tokens']}; format/exact proxy: {x['automatic']['format_or_exact_proxy_pass']}",'','```text',x['text'].split('<|endoftext|>')[0],'```','']
(root/'OUTPUTS.md').write_text('\n'.join(md),encoding='utf8')
fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
for ax,key,title in zip(axs,['format_or_exact_proxy_pass','repeated_word_4gram_fraction'],['Format / exact-answer proxy (higher is better)','Repeated word 4-grams (lower is better)']):
 vals=np.array([s[f][key]['mean']*100 for f in families]); ci=np.array([s[f][key]['ci95'] for f in families])*100
 ax.bar(names,vals,color=['#0075a9','#009b78','#cb7eac','#666666']); ax.errorbar(names,vals,yerr=np.array([vals-ci[:,0],ci[:,1]-vals]),fmt='none',color='black',capsize=3)
 ax.set_title(title,fontsize=10); ax.set_ylabel('%'); ax.set_ylim(0,105); ax.spines[['top','right']].set_visible(False)
fig.supxlabel('16 matched prompts; descriptive 95% prompt-bootstrap intervals. Neither metric measures coherence.',fontsize=9)
fig.savefig(root/'automatic_overview.png',dpi=200); fig.savefig(root/'automatic_overview.pdf')
summary={'status':'complete','report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'bootstrap':{'seed':1337,'resamples':10000,'unit':'prompt'},'metrics':s,'semantic_review':'pending'}
durable_save(root/'summary.json',summary)
# Preserve raw records; attach derived metrics and their hashes to the run report.
r['aggregate_metrics']=s; r['bootstrap']=summary['bootstrap']; r['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'OUTPUTS.md',root/'automatic_overview.png',root/'automatic_overview.pdf']}; durable_save(root/'report.json',r)
print(json.dumps(s,indent=2))
