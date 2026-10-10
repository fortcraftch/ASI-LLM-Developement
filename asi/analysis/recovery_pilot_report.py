"""CPU-only audit of the completed 200-step pilot and exploratory t-SNE maps."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import html
import json
import statistics as st
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from asi.analysis.evidence import read_json, save_figure


def repetition(tokens, n=4):
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1-len(set(grams))/len(grams) if grams else 0.


def mean_ce(result):
    # mean_loss mixes CE and KL in the distillation arm: never compare it to CE.
    return st.mean(row['ce'] for row in result['records'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/recovery_200_screened_20261008'))
    p.add_argument('--output', type=Path, default=Path('results/recovery_pilot_analysis_20261009'))
    p.add_argument('--tsne', action='store_true')
    p.add_argument('--analysis-deps', type=Path, default=Path('results/analysis_dependencies'))
    args = p.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    sources = {}
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        sources[str(path)] = h.hexdigest()
        return h.hexdigest()
    def read(path):
        return read_json(path, sources)
    def save(fig, name):
        fig.tight_layout()
        save_figure(fig, out, name)
    reports = {a:read(args.root/a/'report.json') for a in ('ce','ce_kl','native')}
    comparison = read(args.root/'comparison.json')
    plan = read(args.root/'plan.json')
    for path, expected in plan['source_hashes'].items():
        if sha(path)!=expected: raise ValueError('Pilot input changed: '+path)
    for arm in ('ce','ce_kl'):
        if sources[str(args.root/arm/'report.json')]!=comparison['arms'][arm]['report_sha256']:
            raise ValueError('Report hash differs')
        if not reports[arm]['completed'] or len(reports[arm]['steps'])!=200: raise ValueError('Incomplete arm')
    if not reports['native']['completed']: raise ValueError('Native reference incomplete')
    for a,b in zip(reports['ce']['steps'],reports['ce_kl']['steps']):
        if (a['id'],a['mask'])!=(b['id'],b['mask']): raise ValueError('Training order differs')
    corpus = read('results/recovery_corpus_v3/corpus.json')
    rows = {r['id']:r for r in corpus['examples']}
    eos = read('public_models/olmoe/config.json')['eos_token_id']
    variants = {
        'Original':reports['native']['generations'],
        'Restricted 32':[g for g in reports['ce']['generations'] if g['stage']=='before'],
        'CE step 200':[g for g in reports['ce']['generations'] if g['stage']=='after'],
        'CE+KL step 50':[g for g in reports['ce_kl']['generations'] if g['stage']=='after']}
    expected = {g['id'] for g in variants['Original']}
    for gs in variants.values():
        if len(gs)!=18 or {g['id'] for g in gs}!=expected: raise ValueError('Generation ID mismatch')
        if any(rows[g['id']]['split']!='test' for g in gs): raise ValueError('Non-test example')
    before_kl = {g['id']:g for g in reports['ce_kl']['generations'] if g['stage']=='before'}
    assert all(g['token_ids']==before_kl[g['id']]['token_ids'] for g in variants['Restricted 32'])
    metrics = {}
    for name, gs in variants.items():
        numeric = [g['task_correct'] for g in gs if g['task_correct'] is not None]
        metrics[name] = dict(n=len(gs),math_correct=sum(numeric),math_n=len(numeric),
            repetition4_median=st.median(repetition(g['token_ids']) for g in gs),
            repetition4_mean=st.mean(repetition(g['token_ids']) for g in gs),
            eos_count=sum(bool(g['token_ids']) and g['token_ids'][-1]==eos for g in gs),
            median_tokens=st.median(len(g['token_ids']) for g in gs),
            median_seconds=st.median(g['generation_seconds'] for g in gs),
            median_decode_tps=st.median(g['decode_tokens_per_second'] for g in gs),
            peak_generation_gib=max(g['peak_cuda_allocated_bytes'] for g in gs)/2**30,
            generation_loads=sum(g['cache_after']['loads']-g['cache_before']['loads'] for g in gs),
            generation_h2d_gib=sum(g['cache_after']['host_to_device_bytes']-g['cache_before']['host_to_device_bytes'] for g in gs)/2**30,
            categories={})
        for cat in sorted({rows[g['id']]['category'] for g in gs}):
            group=[g for g in gs if rows[g['id']]['category']==cat]
            metrics[name]['categories'][cat]=dict(repetition4_mean=st.mean(repetition(g['token_ids']) for g in group),
                eos_count=sum(g['token_ids'][-1]==eos for g in group),n=len(group))
    ces=[reports['native']['mean_ce'],mean_ce(reports['ce']['test_before']),
         mean_ce(reports['ce']['test_after']),mean_ce(reports['ce_kl']['test_after'])]
    for name, ce in zip(variants,ces): metrics[name]['test_ce']=ce
    colors=['#64748b','#ef4444','#2563eb','#16a34a']
    fig, axes=plt.subplots(2,3,figsize=(14,8))
    specs=[('test_ce','Test reference CE (lower better)',1),('math_correct','Numeric answer match (of 6)',1),
           ('repetition4_median','Median repeated token 4-grams (%)',100),('eos_count','EOS-stopped answers (of 18)',1),
           ('median_tokens','Median output tokens',1),('repetition4_mean','Mean repeated token 4-grams (%)',100)]
    for ax,(key,title,scale) in zip(axes.flat,specs):
        vals=[metrics[n][key]*scale for n in variants]; bars=ax.bar(list(variants),vals,color=colors)
        ax.bar_label(bars,fmt='%.2f',fontsize=8);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=22,labelsize=8);ax.margins(y=.2)
    fig.suptitle('Recovery pilot: 18 held-out prompts; fluency and correctness are separate outcomes')
    save(fig,'quality')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(key,title) in zip(axes.flat,[('median_seconds','Median response seconds (variable length)'),('median_decode_tps','Median decode tokens/s (variable contexts)'),('peak_generation_gib','CUDA GiB: training state may remain in adapted arms'),('generation_loads','Expert loads during 18 generations'),('generation_h2d_gib','H2D GiB during 18 generations'),('median_tokens','Median output tokens')]):
        bars=ax.bar(list(variants),[metrics[n][key] for n in variants],color=colors);ax.bar_label(bars,fmt='%.2f',fontsize=8)
        ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=22,labelsize=8);ax.margins(y=.2)
    fig.suptitle('Observed process costs, NOT clean inference memory; context preparation excluded from transfers')
    save(fig,'resources')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm,color in [('ce','#2563eb'),('ce_kl','#16a34a')]:
        d=reports[arm];checks=[{'step':0,**d['dev_before']}]+d['dev_checks']
        axes[0].plot([c['step'] for c in checks],[mean_ce(c) for c in checks],'-o',color=color,label=arm)
        for i,key in [(1,0),(2,1)]:axes[i].plot([c['step'] for c in d['dev_generation_checks']],[c['selection_key'][key] for c in d['dev_generation_checks']],'-o',color=color,label=arm)
        for ax in axes:ax.axvline(d['selected_step'],color=color,ls=':',alpha=.5)
    for ax,title in zip(axes,['Dev CE (18 examples)','Dev numeric error (2 examples)','Worst-category mean trigram repetition (2/category)']):
        ax.set(title=title,xlabel='Training step');ax.legend()
    save(fig,'validation')
    # Complete texts, including failures: no cherry-picking or generated-code execution.
    page=['<!doctype html><meta charset="utf-8"><title>Recovery pilot audit</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto}pre{white-space:pre-wrap}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}td{width:25%}</style><h1>Recovery pilot: paired evidence</h1><p>18 test prompts; one seed. CE measures reference fit, not semantic correctness. Times have variable output lengths. Router maps are weight geometry, not activations.</p>']
    for name in ('quality','resources','validation'):page.append(f'<h2>{name}</h2><img src="{name}.png">')
    for id in sorted(expected):
        row=rows[id];page.append('<h2>'+html.escape(id)+'</h2><pre>'+html.escape(row['prompt'])+'</pre><details><summary>Dataset reference</summary><pre>'+html.escape(row['answer'])+'</pre></details><table><tr>')
        for name in variants:page.append('<th>'+html.escape(name)+'</th>')
        page.append('</tr><tr>')
        for name,gs in variants.items():
            g=next(g for g in gs if g['id']==id)
            page.append('<td><pre>'+html.escape(g['text'])+'</pre><p>'+f"Tokens: {len(g['token_ids'])}; repetition4: {repetition(g['token_ids']):.1%}; numeric match: {g['task_correct']}"+'</p></td>')
        page.append('</tr></table>')
    map_report={}
    if args.tsne:
        if args.analysis_deps.exists():sys.path.insert(0,str(args.analysis_deps.resolve()))
        import sklearn
        from sklearn.manifold import TSNE, trustworthiness
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
        import torch
        torch.set_num_threads(1)
        def load_router(path):
            sha(path);d=torch.load(path,map_location='cpu',weights_only=True)
            return np.stack([d['routers'][str(i)].float().numpy() for i in range(16)])
        base=load_router(args.root/'ce/adaptation_step_0000.pt')
        base2=load_router(args.root/'ce_kl/adaptation_step_0000.pt')
        np.testing.assert_array_equal(base,base2)
        learned=[load_router(args.root/a/'adaptation_best.pt') for a in ('ce','ce_kl')]
        drift=np.stack([np.linalg.norm(x-base,axis=2)/np.maximum(np.linalg.norm(base,axis=2),1e-12) for x in learned])
        np.savez_compressed(out/'router_features.npz',before=base,ce=learned[0],ce_kl=learned[1],relative_drift=drift)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,x,name in zip(axes,drift,['CE step 200','CE+KL step 50']):
            im=ax.imshow(x,aspect='auto',vmin=0,vmax=drift.max(),cmap='magma');fig.colorbar(im,ax=ax,label='L2 change / initial L2 norm');ax.set(title=name,xlabel='Expert',ylabel='Layer')
        save(fig,'router_drift')
        map_report={'sklearn':sklearn.__version__,'router_relative_drift_median':np.median(drift,axis=(1,2)).tolist(),'tsne_runs':[]}
        with threadpool_limits(limits=1):
            # Separate layer spaces. Each projection jointly fits before and both selected adapters.
            fig,axes=plt.subplots(2,3,figsize=(14,9))
            for column,layer in enumerate((0,7,15)):
                x=np.concatenate([base[layer],learned[0][layer],learned[1][layer]])
                x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
                z=PCA(n_components=30,random_state=0).fit_transform(x)
                km=KMeans(n_clusters=4,n_init=10,random_state=0).fit_predict(z)
                for ri,(seed,perplexity) in enumerate(((0,15),(42,30))):
                    tsne=TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',random_state=seed,max_iter=1500,n_jobs=1)
                    xy=tsne.fit_transform(z);ax=axes[ri,column]
                    for start,marker,label in [(0,'o','Before'),(64,'^','CE 200'),(128,'x','CE+KL 50')]:
                        ax.scatter(*xy[start:start+64].T,c=km[start:start+64],cmap='tab10',vmin=0,vmax=9,marker=marker,s=18,label=label,alpha=.65)
                    ax.set_title(f'Layer {layer}; perplexity {perplexity}; seed {seed}',fontsize=10)
                    ax.legend(fontsize=7);ax.set_xticks([]);ax.set_yticks([])
                    score=trustworthiness(x,xy,n_neighbors=5)
                    map_report['tsne_runs'].append(dict(layer=layer,seed=seed,perplexity=perplexity,trustworthiness5=score,kl=tsne.kl_divergence_))
                    np.savetxt(out/f'router_layer{layer}_seed{seed}.csv',np.column_stack([np.repeat(range(3),64),np.tile(range(64),3),xy,km]),delimiter=',',header='stage,expert,x,y,kmeans4',comments='')
            fig.suptitle('OLMoE router weights: joint before/after t-SNE per layer\nColors: exploratory KMeans k=4 in PCA space; shapes: stage; not domain/causal labels')
            save(fig,'router_tsne')
            # Existing DeepSeek routing profiles; no new inference.
            path=Path('results/expert_maps_20261009/deepseek_profiles.csv');sha(path)
            features=np.loadtxt(path,delimiter=',',skiprows=1);x=features[:,2:]
            clusters=KMeans(n_clusters=4,n_init=10,random_state=0).fit_predict(x)
            fig,axes=plt.subplots(2,2,figsize=(10,9))
            for ax,(seed,perplexity) in zip(axes.flat,[(0,10),(42,10),(0,30),(42,30)]):
                t=TSNE(perplexity=perplexity,init='random',learning_rate='auto',random_state=seed,max_iter=1500,n_jobs=1)
                xy=t.fit_transform(x);ax.scatter(*xy.T,c=clusters,cmap='tab10',vmin=0,vmax=9,s=22)
                score=trustworthiness(x,xy,n_neighbors=5)
                ax.set_title(f'Perplexity {perplexity}; seed {seed}; trust@5={score:.3f}')
                ax.set_xticks([]);ax.set_yticks([])
                map_report['tsne_runs'].append(dict(model='DeepSeek',seed=seed,perplexity=perplexity,trustworthiness5=score,kl=t.kl_divergence_))
                np.savetxt(out/f'deepseek_p{perplexity}_seed{seed}.csv',np.column_stack([features[:,:2],xy,clusters]),delimiter=',',header='layer,expert,x,y,kmeans4',comments='')
            fig.suptitle('DeepSeek: t-SNE of six routing-share features from three probes\nColors: KMeans k=4 in original feature space; not validated specialization')
            save(fig,'deepseek_tsne')
        for name in ('router_drift','router_tsne','deepseek_tsne'):page.insert(3,f'<h2>{name}</h2><img src="{name}.png">')
    sha(__file__)
    result=dict(completed=True,metrics=metrics,maps=map_report,sources_sha256=sources,
        limitations=['Single seed; 18 test prompts; no semantic aggregate score.','CE+KL selected at step 50, CE at 200; final comparisons differ in update count.','Teacher-forced CE and short fluent outputs do not establish reasoning recovery.','Router weight geometry is not expert output representations or causal influence.','CUDA memory in training arms may retain optimizer/gradients; not clean inference-only measurements.'])
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'index.html').write_text(''.join(page),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
