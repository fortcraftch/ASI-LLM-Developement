"""Descriptive single-prompt comparison; never infer a scaling law from two runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

paths=[Path('results/v2/F0/olmoe_base_timing_20261021'),Path('results/v2/F0/olmoe_base_timing_256_20261021')]
reports=[json.loads((p/'report.json').read_text(encoding='utf8')) for p in paths]
assert all(r['status']=='complete' and len(r['records'])==1 for r in reports)
a,b=[r['records'][0] for r in reports]
for key in ('source','torch','transformers','gpu'):
    assert reports[0][key]==reports[1][key], key
for key in ('prompt','input_ids','token_ids'):
    assert a[key]==b[key],key
assert [e['routes'] for e in a['events']]==[e['routes'] for e in b['events']]
out=Path('results/v2/F0/olmoe_cache_comparison_20261021');out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
rows=[]
for i,(record,slots,color) in enumerate(zip((a,b),(512,256),('#197b9b','#c26a29'))):
    events=record['events'];amount=np.cumsum([e['host_to_device_bytes']/2**30 for e in events])
    ax.plot(amount,label=f'{slots} global cache slots',color=color,lw=2)
    fetch=sum(e['timing_delta']['weight_fetch_seconds'] for e in events)
    copy=sum(e['timing_delta']['h2d_wall_seconds'] for e in events)
    bx.barh(i,fetch,color='#596b7a',label='Prepare weights in RAM\n(including storage access)' if i==0 else None)
    bx.barh(i,copy,left=fetch,color='#c26a29',label='H2D copy calls + wait' if i==0 else None)
    bx.text(fetch+copy+.3,i,f'{fetch+copy:.1f} s',va='center')
    rows.append({'slots':slots,'h2d_gib':float(amount[-1]),'weight_preparation_seconds':fetch,'h2d_wall_seconds':copy,'generation_seconds':record['generation_seconds']})
ax.set(xlabel='Forward step (0 = batched prefill)',ylabel='Cumulative H2D (GiB)',ylim=(0,None));ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
bx.set(yticks=[0,1],yticklabels=['512 slots','256 slots'],xlabel='Measured loading components (seconds)');bx.invert_yaxis();bx.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=8,ncol=2)
bx.set_xlim(0,max(r['weight_preparation_seconds']+r['h2d_wall_seconds'] for r in rows)*1.22)
fig.suptitle('OLMoE base: same prompt, weights and routing; different GPU cache budgets')
fig.savefig(out/'olmoe_cache_comparison.pdf');fig.savefig(out/'olmoe_cache_comparison.png',dpi=180)
summary={'status':'complete','same_tokens_and_routes':True,'rows':rows,'ratios_256_over_512':{k:rows[1][k]/rows[0][k] for k in ('h2d_gib','weight_preparation_seconds','h2d_wall_seconds','generation_seconds')},'limitations':['One sequential run per budget; OS cache uncontrolled.','Profiling synchronizes loads and adds overhead.','Two budgets do not establish a superlinear scaling law.']}
(out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
