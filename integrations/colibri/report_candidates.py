import json,re,statistics as st,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/colibri_candidates_vulkan_20261014');runs=json.loads((p/'runs.json').read_text());assert len(runs)==33
for i in range(3):assert runs[2*i]['tokens']==runs[2*i+1]['tokens']
rows=[]
for r in runs:
 log=(p/(r['label']+'.log')).read_text();assert '[VK] ready:' in log
 trace=(p/(r['label']+'.trace')).read_text();count=0
 cats=['life_sciences_biology','mathematics_statistics','computer_science_software_engineering'];n=r['candidates']
 ids=list(map(int,(p/f'{cats[r["prompt_index"]]}_{n}.txt').read_text().split()))[3:]
 for line in trace.splitlines():
  f=line.split()
  if len(f)<11 or line.startswith('#'):continue
  layer=int(f[2]);chosen=[int(x.split(':')[0]) for x in f[3:]]
  assert len(chosen)==8 and len(set(chosen))==8 and set(chosen)<=set(ids[layer*n:(layer+1)*n]);count+=1
 assert count==r['audited_rows']
 if not r['label'].startswith('repeat'):continue
 r.update(disk_loads=int(re.search(r'disk loads (\d+)',log)[1]),evictions=int(re.search(r'evictions (\d+)',log)[1]),staged_mib=float(re.search(r'([\d.]+) MiB staged in',log)[1]))
 rows.append(r)
summary=[]
for i in range(3):
 base=st.median(r['seconds'] for r in rows if r['prompt_index']==i and r['candidates']==64)
 for n in [64,32,16]:
  group=[r for r in rows if r['prompt_index']==i and r['candidates']==n];values=[r['seconds'] for r in group]
  summary.append(dict(prompt=i,candidates=n,median=st.median(values),minimum=min(values),maximum=max(values),speedup=base/st.median(values),median_process=st.median(r['process_seconds'] for r in group),median_disk_loads=st.median(r['disk_loads'] for r in group),median_evictions=st.median(r['evictions'] for r in group),median_staged_mib=st.median(r['staged_mib'] for r in group)))
(p/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Vulkan category timing screen — completed','', 'Same patched build, same INT8 weights, top8 active, CPU cache8/layer, GPU expert-tier budget 2 GiB (dense weights additional). Six all64 controls passed; 27 shuffled runs audited, three repetitions per prompt/arm.','', '| Prompt | Candidates/layer | Median seconds [min, max] | Speedup | Median process seconds |','|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {cats[r['prompt']]} | {r['candidates']} | {r['median']:.3f} [{r['minimum']:.3f}, {r['maximum']:.3f}] | {r['speedup']:.2f}x | {r['median_process']:.3f} |")
lines+=['','## Interpretation','','A consistent timing advantage exists in this constrained GPU-tier regime, warranting recovery experiments. It is not a quality-equivalent speedup: restricted routing changes outputs and computation paths. No recovery was trained here. The engine remains hybrid CPU/GPU, not all computation on GPU. With 16 candidates/layer there are at most 256 unique candidates versus a reported tier capacity of 272; this helps explain reduced churn. No conclusion for unrestricted VRAM or larger models follows.','','Timings include prefill plus 64 generated tokens; process times additionally include startup and trace parsing. Trace logging is enabled in every arm. No per-token prefill/decode timing separation was available. OS page cache is uncontrolled; disk-load counters mean logical expert loads, not physical SSD reads. Staged MiB are device upload counters, not disk bytes. Three repetitions describe observed spread, not statistical significance. Candidate selection was precomputed and its production cost is not timed.','','Shader compatibility: all runs executed GPU expert work successfully, with same-build all64 token equivalence. Precompiled release shaders were reused; this is a runtime smoke check, not independent numerical validation against a shader rebuild or HF logits.','','## Cache evidence','','| Prompt | Candidates | Median logical loads | Median evictions | Median staged MiB |','|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['prompt']} | {r['candidates']} | {r['median_disk_loads']} | {r['median_evictions']} | {r['median_staged_mib']:.1f} |")
(p/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
fig,ax=plt.subplots(figsize=(8,4))
for j,n in enumerate([64,32,16]):
 g=[r for r in summary if r['candidates']==n];v=[r['median'] for r in g]
 ax.bar([i+(j-1)*.25 for i in range(3)],v,width=.24,label=f'{n} candidates',yerr=[[r['median']-r['minimum'] for r in g],[r['maximum']-r['median'] for r in g]],capsize=3)
ax.set_xticks(range(3),['Gene','Derivative','Python']);ax.set_ylabel('Seconds, prefill + 64 tokens');ax.set_title('Vulkan INT8, 2 GiB expert tier: median and range');ax.legend();fig.tight_layout();fig.savefig(p/'timing.png',dpi=160)
files=list(p.glob('*.log'))+list(p.glob('*.trace'))+[p/'runs.json']
(p/'validation.json').write_text(json.dumps({'all64_equivalent':True,'audited_runs':33,'all_masks_valid':True,'hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}},indent=2))
print([(r['prompt'],r['candidates'],round(r['median'],2),round(r['speedup'],2)) for r in summary])
