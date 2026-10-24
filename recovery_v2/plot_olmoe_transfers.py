"""One OLMoE trace: active-set changes, H2D bytes and measured loading time."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

root=Path('results/v2/F0/olmoe_base_timing_20261021')
out=root/'legend_revision';out.mkdir(exist_ok=True)
report=json.loads((root/'report.json').read_text(encoding='utf8'))
assert report['status']=='complete' and len(report['records'])==1
r=report['records'][0]; events=r['events']; layers=list(events[0]['routes'])
assert r['mode']=='native'
old=json.loads(Path('results/v2/F0/olmoe_base_trace_20261021/report.json').read_text(encoding='utf8'))['trace']
same_tokens=r['token_ids']==old['token_ids']
same_routes=[e['routes'] for e in events]==[e['routes'] for e in old['events']]
if not same_tokens or not same_routes:
    raise ValueError('Instrumentation changed outputs or routes; investigate before publishing')
prefill=len(events[0]['routes'][layers[0]])
seq={l:[ids for e in events for ids in e['routes'][l]] for l in layers}
n=len(seq[layers[0]]);changes=np.zeros((len(layers),n))
for i,l in enumerate(layers):
    for j in range(1,n):changes[i,j]=len(set(seq[l][j])-set(seq[l][j-1]))
colors=['#f0f3f6','#d8e7ef','#b1d1e1','#82b6cf','#5598bb','#3179a4','#1d5c86','#103f66','#082945']
cmap=ListedColormap(colors);cmap.set_bad('white')
data=np.ma.array(changes,mask=np.zeros_like(changes,dtype=bool));data.mask[:,0]=True
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(10,8.1),layout='constrained')
grid=fig.add_gridspec(3,2,width_ratios=[1,.025],height_ratios=[2.2,1,1.1])
ax=fig.add_subplot(grid[0,0]);cb=fig.add_subplot(grid[0,1]);bx=fig.add_subplot(grid[1,0]);cx=fig.add_subplot(grid[2,0])
im=ax.imshow(data,aspect='auto',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,9),9),interpolation='nearest')
ax.set(yticks=np.arange(16),yticklabels=np.arange(1,17),ylabel='MoE layer',xlabel='Processed token position',title='OLMoE: expert selection and loading in one response')
ax.axvline(prefill-.5,color='#bc5b28',ls='--',lw=1.5)
fig.colorbar(im,cax=cb,ticks=[0,2,4,6,8]).set_label('New active experts (out of 8)')
x=np.arange(prefill-1,n);amount=np.array([e['host_to_device_bytes'] for e in events])/2**30
bx.plot(x,np.cumsum(amount),color='#c76a28',lw=2)
bx.scatter([x[0]],[amount[0]],color='#c76a28',s=25)
bx.axvline(prefill-.5,color='#bc5b28',ls='--',lw=1.5)
bx.set(xlim=(-.5,n-.5),ylabel='Cumulative H2D (GiB)',xlabel='Processed token position')
bx.set_ylim(0,amount.sum()*1.12)
bx.grid(axis='y',alpha=.15)
bx.annotate(f'Prefill: {amount[0]:.2f} GiB\n(one batched forward)',(x[0],amount[0]),xytext=(1,amount.sum()*.65),arrowprops={'arrowstyle':'-','color':'#666'},fontsize=9)
bx.text(.98,.1,f'Total: {amount.sum():.2f} GiB',ha='right',transform=bx.transAxes)
phases=[events[:1],events[1:]]
fetch=np.array([sum(e['timing_delta']['weight_fetch_seconds'] for e in phase) for phase in phases])
copy=np.array([sum(e['timing_delta']['h2d_wall_seconds'] for e in phase) for phase in phases])
gpu=np.array([sum(e['timing_delta']['h2d_cuda_seconds'] for e in phase) for phase in phases])
cx.barh([0,1],fetch,color='#596b7a',label='Prepare weights in RAM\n(including storage access)')
cx.barh([0,1],copy,left=fetch,color='#c76a28',label='H2D copy calls + wait (wall time)')
for i,v in enumerate(fetch+copy):cx.text(v+.3,i,f'{v:.2f} s',va='center',fontsize=9)
cx.set(yticks=[0,1],yticklabels=['Prefill','Decode'],xlabel='Measured loading components (seconds)',xlim=(0,max(fetch+copy)*1.22))
cx.invert_yaxis();cx.legend(frameon=False,fontsize=8,loc='lower right');cx.grid(axis='x',alpha=.15)
fig.savefig(out/'olmoe_base_transfers.pdf');fig.savefig(out/'olmoe_base_transfers.png',dpi=180)
summary={'same_tokens_as_unprofiled':same_tokens,'same_routes_as_unprofiled':same_routes,'h2d_gib':float(amount.sum()),
         'host_weight_fetch_seconds':float(fetch.sum()),'h2d_wall_seconds':float(copy.sum()),'h2d_cuda_event_seconds':float(gpu.sum()),
         'profiled_generation_seconds':r['generation_seconds'],'phase_weight_fetch_seconds':fetch.tolist(),'phase_copy_wall_seconds':copy.tolist(),
         'limitations':['Synchronized profiling changes timing; not an estimate of speedup without transfers.',
                        'Host fetch includes cache lookup, mapped reads, allocations and tensor assembly; not pure physical SSD IO.',
                        'CUDA-event intervals bracket copy calls and can include host launch gaps; wall time includes allocation and waits.',
                        'Timing components are not total inference time; synchronization overhead and model compute remain outside them.',
                        'One prompt. No repeated-run uncertainty estimate.']}
(out/'timing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
