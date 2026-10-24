"""Reproducible CPU-only summary of the isolated dense inference audit."""
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    root=Path('results/v2/F0'); source=root/'dense_inference_20261023/report.json'
    isolated=root/'dense_inference_20261023_isolated_technology/report.json'
    old=json.loads(source.read_text()); r=json.loads(isolated.read_text())
    assert r['status']==old['status']=='complete' and len(r['results'])==16
    assert sha(source)==r['supplement_provenance']['sha256']
    assert all(v==old['results'][k] for k,v in r['results'].items() if k!='bank/technology')
    assert r['settings']==old['settings']
    assert sha('recovery_v2/dense_inference_audit.py')==r['settings']['script_sha256']
    assert sha('../Creating-GPT-2-From-0/eval.py')==r['settings']['evaluator_sha256']
    out=root/'dense_inference_20261023_analysis'; out.mkdir(exist_ok=True)
    rng=np.random.default_rng(1337); samples=rng.integers(0,8,(10000,8)); stats={}; arrays={}
    for family in ['bank','gpt2']:
        rows=[v for k,v in r['results'].items() if k.startswith(family+'/')]
        assert len(rows)==8
        for mode in rows[0]['modes']:
            ms=[v['modes'][mode] for v in rows]
            for row,m in zip(rows,ms):
                assert m['logit_checks']['fp32']['allclose']
                assert m['fixed_tokens_identical'] and m['eos_tokens_identical']
                assert m['fixed_output']['output_ids']==row['reference']['output_ids']
                assert m['eos_output']['output_ids']==row['eos_reference']['output_ids']
                assert len(m['runs'])==3
                assert all(run['output_ids']==row['reference']['output_ids'] and run['generated_tokens']==64 for run in m['runs'])
            a=np.array([[run['decode_tokens_per_second'] for run in m['runs']] for m in ms]); arrays[family+'/'+mode]=a
            boot=np.median(a[samples],axis=(1,2))
            stats[family+'/'+mode]={'median_decode_tokens_s':float(np.median(a)), 'ci95':np.quantile(boot,[.025,.975]).tolist(),
             'fp32_allclose_prompts':8,'tokens_identical_prompts':8,'bf16_allclose_prompts':sum(m['logit_checks']['bf16']['allclose'] for m in ms),
             'fp32_max_abs':max(m['logit_checks']['fp32']['max_abs'] for m in ms), 'bf16_max_abs':max(m['logit_checks']['bf16']['max_abs'] for m in ms)}
    a=arrays['bank/sdpa_last']; b=arrays['bank/baseline']; ratio=np.median(a[samples],axis=(1,2))/np.median(b[samples],axis=(1,2))
    report={'status':'complete','gate':'pass on 8 diagnostic prompts per family; not universal equivalence','input_sha256':{str(source):sha(source),str(isolated):sha(isolated)},'settings':r['settings'],'git_commit':r['git_commit'],'versions':r['versions'],'gpu':r['gpu'],'bootstrap':{'seed':1337,'resamples':10000,'unit':'prompt; all three repetitions retained','statistic':'median decode tokens/s across prompt repetitions'},'metrics':stats,'bank_sdpa_speed_ratio':float(np.median(a)/np.median(b)),'bank_sdpa_ratio_ci95':np.quantile(ratio,[.025,.975]).tolist(),'limitations':r['limitations']+['BF16 logits fail allclose in optimized variants despite exact generated tokens.','No RSS/NVML peak comparison; previous memory results retained.','Bank technology retimed in isolation; other 15 cases unchanged.']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.4),sharey=True,layout='constrained')
    for ax,family,title in zip(axs,['bank','gpt2'],['Independent bank (93.85M active)','GPT-2 dense (124.48M)']):
        modes=['baseline','last']+(['sdpa_last'] if family=='bank' else [])
        for i,m in enumerate(modes):
            d=stats[family+'/'+m]; value=d['median_decode_tokens_s']; low,high=d['ci95']
            ax.bar(i,value,color=['#64748b','#58a6a6','#086788'][i],width=.6)
            ax.errorbar(i,value,yerr=[[value-low],[high-value]],fmt='none',color='#172033',capsize=4)
            ax.text(i,high+3,f'{value:.1f}',ha='center')
        ax.set_xticks(range(len(modes)),['Baseline','Last logit']+(['SDPA +\nlast logit'] if family=='bank' else [])); ax.set_title(title,fontsize=11); ax.set_ylim(0,105); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    axs[0].set_ylabel('Decode throughput (tokens/s; TTFT excluded)')
    for ext in ['pdf','png']: fig.savefig(out/f'dense_inference_ablation.{ext}',dpi=200)
    report['output_sha256']={p.name:sha(p) for p in out.glob('dense_inference_ablation.*')}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['metrics'],indent=2)); print('Speed ratio',report['bank_sdpa_speed_ratio'],report['bank_sdpa_ratio_ci95'])
if __name__=='__main__': main()
