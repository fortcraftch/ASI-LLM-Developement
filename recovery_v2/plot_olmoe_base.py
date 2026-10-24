"""Single completed base-model trace; no restricted-model comparison."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

source=Path('results/v2/F0/olmoe_residency_20261021/retry_autocast/report.json')
report=json.loads(source.read_text(encoding='utf8'))
record=next(r for r in report['records'] if r['mode']=='native' and r['category']=='computer_science_software_engineering')
assert len(record['events'])==len(record['token_ids'])==32
layers=list(record['events'][0]['routes'])
sequences={layer:[ids for event in record['events'] for ids in event['routes'][layer]] for layer in layers}
count=len(sequences[layers[0]])
prefill=len(record['events'][0]['routes'][layers[0]])
changes=np.zeros((len(layers),count),dtype=int)
for i,layer in enumerate(layers):
    assert len(sequences[layer])==count
    for j in range(1,count):
        assert len(set(sequences[layer][j]))==8
        changes[i,j]=len(set(sequences[layer][j])-set(sequences[layer][j-1]))
out=Path('results/v2/F0/olmoe_base_trace_20261021');out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(10,4.9),layout='constrained')
colors=['#f0f3f6','#d8e7ef','#b1d1e1','#82b6cf','#5598bb','#3179a4','#1d5c86','#103f66','#082945']
cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,9),9)
data=np.ma.array(changes,mask=np.zeros_like(changes,dtype=bool));data.mask[:,0]=True
cmap.set_bad('#ffffff')
im=ax.imshow(data,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
ax.axvline(prefill-.5,color='#bc5b28',lw=1.8,ls='--')
ax.set(yticks=np.arange(16),yticklabels=np.arange(1,17),ylabel='MoE layer',xlabel='Processed token position')
ax.set_title('OLMoE: changing expert selections during one response',pad=28,fontsize=13)
ax.text((prefill-1)/2,-1,'Prompt processing',ha='center',fontsize=10)
ax.text((prefill+count-1)/2,-1,'Autoregressive decoding',ha='center',fontsize=10)
bar=fig.colorbar(im,ax=ax,ticks=[0,2,4,6,8],shrink=.85,pad=.025)
bar.set_label('New experts in the active set (out of 8)')
fig.savefig(out/'olmoe_base_routing.pdf');fig.savefig(out/'olmoe_base_routing.png',dpi=180)
summary={'status':'complete','scope':'One completed native trace extracted from a user-cancelled wider experiment.',
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model':report['source'],
         'prompt':record['prompt'],'output':record['output'],'prefill_tokens':prefill,'generated_tokens':32,
         'processed_positions':count,'layer_transitions':int(changes[:,1:].size),
         'changed_layer_transitions':int((changes[:,1:]>0).sum()),'trace':record,
         'limitations':['Illustration, not an estimate across prompts.', 'Changes in active selections do not imply weight transfers.','First column has no predecessor; last generated token is not forwarded.']}
(out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print({k:v for k,v in summary.items() if k in ['prefill_tokens','processed_positions','layer_transitions','changed_layer_transitions']})
