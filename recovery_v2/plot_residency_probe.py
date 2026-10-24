from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/v2/F0/deepseek_residency_20261021');r=json.loads((p/'report.json').read_text());a=next(x for x in r['records'] if x['mode']=='native_cache' and x['category']=='programming');e=a['events'];layers=list(e[0]['routes']);changes=np.zeros((len(layers),len(e)))
for j in range(1,len(e)):
 for i,l in enumerate(layers): changes[i,j]=sorted(np.asarray(e[j]['routes'][l]).ravel())!=sorted(np.asarray(e[j-1]['routes'][l]).ravel())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(2,1,sharex=True,figsize=(9,5.6),gridspec_kw={'height_ratios':[1.5,1]},layout='constrained')
y,x=np.where(changes);ax.scatter(x,y,marker='|',s=80,color='#c76a28',linewidth=2)
ax.set(yticks=range(len(layers)),yticklabels=layers,ylabel='MoE layer',xlim=(-.5,len(e)-.5));ax.invert_yaxis();ax.set_title('DeepSeek final: programming prompt\nMarks show expert-pair changes from the preceding step',fontsize=11,pad=10)
bound=sum(t['phase']=='prefill' for t in e)-.5
for axis in (ax,bx):axis.axvline(bound,ls=':',color='#555555',lw=1)
for mode,label,color in [('native_cache','Native routing, 2 cache slots/layer','#c76a28'),('fixed_pair','Fixed pair; 28.9 MiB preloaded','#177d96')]:
 row=next(x for x in r['records'] if x['mode']==mode and x['category']=='programming');bx.plot(np.cumsum([x['cache_delta']['host_to_device_bytes']/2**20 for x in row['events']]),label=label,color=color,lw=2)
bx.set(xlabel='Compute step (serial prefill, then decoding)',ylabel='Cumulative H2D (MiB)');bx.legend(frameon=False,fontsize=9);bx.grid(axis='y',alpha=.15)
fig.savefig(p/'deepseek_swapping.pdf');fig.savefig(p/'deepseek_swapping.png',dpi=180)
