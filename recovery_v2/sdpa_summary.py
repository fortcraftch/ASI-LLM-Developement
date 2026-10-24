import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/v2/F0/sdpa_four_models_20261023');p=root/'report.json';r=json.loads(p.read_text());assert r['status']=='complete' and len(r['results'])==32
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('recovery_v2/sdpa_four_model_audit.py')==r['settings']['script_sha256']
for path,h in r['settings']['source_hashes'].items():assert sha(path)==h
assert sha('../Creating-GPT-2-From-0/eval.py')==r['settings']['evaluator_sha256']
rng=np.random.default_rng(1337);ix=rng.integers(0,8,(10000,8));stats={};changes=[]
for family in ['asi','deepseek','bank','gpt2']:
 rows=[(k,v) for k,v in r['results'].items() if k.startswith(family+'/')];assert len(rows)==8
 mode='last' if family=='gpt2' else 'sdpa_last';ms=[v['modes'][mode] for k,v in rows]
 a=np.array([[run['decode_tokens_per_second'] for run in m['runs']] for m in ms]);assert a.shape==(8,3)
 runs=[z for m in ms for z in m['runs']];assert all(z['generated_tokens']==64 for z in runs)
 stats[family]={'decode_tokens_s':float(np.median(a)),'ci95':np.quantile(np.median(a[ix],axis=(1,2)),[.025,.975]).tolist(),'ttft_ms':float(np.median([z['first_token_seconds'] for z in runs])*1000),'cuda_peak_mib':max(z['peak_cuda_allocated_bytes'] for z in runs)/2**20,'gpu_weights_mib':sum(v for k,v in runs[0]['memory']['weight_bytes_by_device'].items() if k.startswith('cuda'))/2**20,'rss_peak_mib':max(z['sampled_peaks']['rss_bytes'] for z in runs)/2**20,'nvml_peak_mib':max((z['sampled_peaks']['nvml_process_bytes']/2**20 for z in runs if z['sampled_peaks']['nvml_process_bytes'] is not None),default=None),'expert_h2d_bytes':sum(z['measured_expert_h2d_bytes'] for z in runs),'fp32_allclose':sum(m['logit_checks']['fp32']['allclose'] for m in ms),'fixed_identical':sum(m['fixed_tokens_identical'] for m in ms),'eos_identical':sum(m['eos_tokens_identical'] for m in ms),'repeat_matches_optimized':sum(z['output_ids']==m['fixed_output']['output_ids'] for m in ms for z in m['runs'])}
 for key,v in rows:
  m=v['modes'][mode]
  if not(m['fixed_tokens_identical'] and m['eos_tokens_identical']):
   aa=v['reference']['output_ids'];bb=m['fixed_output']['output_ids'];first=next((i+1 for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
   changes.append(dict(case=key,first_changed_token=first,prompt=v['prompt'],baseline=v['reference']['text'],sdpa=m['fixed_output']['text'],logit_checks=m['logit_checks'],fixed_identical=m['fixed_tokens_identical'],eos_identical=m['eos_tokens_identical']))
summary=dict(status='complete',publication='awaiting_user_decision',source_sha256=sha(p),bootstrap={'unit':'prompt, all three repetitions retained','resamples':10000,'seed':1337},metrics=stats,changes=changes,limitations=['Descriptive single-checkpoint diagnostic; no quality or significance claim.','Full-prefix generation, FP32 weights/BF16 autocast, batch1; SDPA kernels may differ by shape.','RSS/NVML sampled at 50ms; null NVML means unavailable.','Output changes imply timings can follow different trajectories.'])
(root/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained');labels=['Categorized\nMoE','Free MoE','Bank\n93.85M active','GPT-2\ndense'];colors=['#0072b2','#009e73','#cc79a7','#666666']
for ax,key,title,unit in zip(axs.flat,['gpu_weights_mib','cuda_peak_mib','decode_tokens_s','ttft_ms'],['GPU model weights','Peak CUDA allocation','SDPA decode (first token excluded)','Time to first token'],['MiB','MiB','tokens/s','ms']):
 vals=[s[key] for s in stats.values()];bars=ax.bar(range(4),vals,color=colors);ax.bar_label(bars,fmt='%.1f',padding=4);ax.set_xticks(range(4),labels);ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(0,max(vals)*1.25);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 if key=='decode_tokens_s':ax.errorbar(range(4),vals,yerr=[[s[key]-s['ci95'][0] for s in stats.values()],[s['ci95'][1]-s[key] for s in stats.values()]],fmt='none',color='black',capsize=3)
for ext in ['png','pdf']:fig.savefig(root/('sdpa_comparison.'+ext),dpi=180)
(root/'DECISION.md').write_text('Measurement complete; publication pending user decision.\n\nChanged cases: '+', '.join(c['case'] for c in changes)+'.\nSee summary.json for complete baseline/SDPA outputs. No Overleaf changes. Proposed publication: label as matched SDPA timing, explicitly report changed outputs; do not claim equivalent quality. Alternative: full FP32 comparison as a separate protocol.\n')
print(json.dumps(stats,indent=2)); print('Changes',[(x['case'],x['first_changed_token']) for x in changes])
