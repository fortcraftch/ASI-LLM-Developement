"""Offline expert maps from archived JSON only; never imports torch or loads weights."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from asi.analysis.evidence import read_json, save_figure


def assignment_share(counts):
    counts = np.asarray(counts, dtype=float)
    if (counts < 0).any():
        raise ValueError('Negative assignment count')
    return counts / counts.sum() if counts.sum() else np.zeros_like(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/expert_maps_20261009')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    def read(path):
        return read_json(path, provenance)
    def save(fig, name):
        save_figure(fig, out, name)

    deep = read('results/deepseek_07874_eval_100.json')
    probes = deep['routing_audit']['probes']
    layers = sorted(probes[0]['layers'], key=int)
    n = len(probes[0]['layers'][layers[0]]['prefill_assignments'])
    features = np.array([[assignment_share(p['layers'][layer][phase])[expert]
                         for p in probes for phase in ('prefill_assignments', 'last_position_assignments')]
                         for layer in layers for expert in range(n)])
    np.savetxt(out / 'deepseek_profiles.csv',
               np.column_stack(([int(l) for l in layers for _ in range(n)], list(range(n))*len(layers), features)),
               delimiter=',', header='layer,expert,' + ','.join(f'probe{i}_{p}' for i in range(len(probes)) for p in ('prefill','decode')), comments='')
    fig, axes = plt.subplots(2, len(probes), figsize=(13, 7), squeeze=False)
    for i, prompt in enumerate(probes):
        for phase in range(2):
            ax = axes[phase, i]
            im = ax.imshow(features[:, 2*i+phase].reshape(len(layers), n), vmin=0, vmax=features.max(), cmap='magma', aspect='auto')
            ax.set(title=f'Probe {i+1}: ' + ('prefill' if phase == 0 else 'decode'), xlabel='Expert ID', ylabel='MoE layer')
            ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Share of top-2 assignments', shrink=.8)
    fig.suptitle('DeepSeek checkpoint 07874: observed routing on three short probes')
    save(fig, 'deepseek_routing')
    centered = features - features.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    xy = u[:, :2] * s[:2]
    variance = s[:2]**2 / np.sum(s**2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c = axes[0].scatter(*xy.T, c=np.repeat(list(map(int, layers)), n), cmap='viridis', s=25, alpha=.75)
    fig.colorbar(c, ax=axes[0], label='Layer')
    axes[0].set_title('Same routing profiles: colored by layer')
    preference = features.reshape(-1, len(probes), 2).mean(axis=2).argmax(axis=1)
    for i in range(len(probes)):
        mask = (preference == i) & (features.sum(axis=1) > 0)
        axes[1].scatter(*xy[mask].T, label=f'Largest share: probe {i+1}', s=25, alpha=.75)
    absent = features.sum(axis=1) == 0
    if absent.any(): axes[1].scatter(*xy[absent].T, marker='x', color='grey', label='Never observed')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Probe preference is not validated specialization')
    for ax in axes:
        ax.set(xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})')
    fig.suptitle('88 routed experts; PCA of 6 assignment-share features (not hidden representations)')
    fig.tight_layout()
    save(fig, 'deepseek_pca')

    olmo = read('results/recovery_calibration_batched_v3/report.json')
    categories = list(olmo['mappings']['32'])
    labels = [dict(computer_science_software_engineering='Programming', life_sciences_biology='Biology', mathematics_statistics='Mathematics').get(c,c) for c in categories]
    stats = {}
    for k in ('32', '16'):
        mapping = olmo['mappings'][k]
        ls = sorted(mapping[categories[0]], key=int)
        masks = np.zeros((len(categories), len(ls), 64), dtype=bool)
        for ci, cat in enumerate(categories):
            for li, layer in enumerate(ls):
                ids = mapping[cat][layer]['experts']
                assert len(ids) == int(k) and len(set(ids)) == int(k)
                masks[ci, li, ids] = True
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ci, ax in enumerate(axes):
            ax.imshow(masks[ci], cmap=ListedColormap(['#eeeeee','#2563eb']), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
            ax.set(title=labels[ci], xlabel='Expert ID', ylabel='MoE layer')
            ax.set_yticks(range(len(ls)), ls)
        fig.suptitle(f'OLMoE: top-{k} selection membership per category (blue = selected; grey = excluded)')
        fig.tight_layout()
        save(fig, f'olmoe_selection_{k}')
        overlap = np.empty((3,3))
        for a in range(3):
            for b in range(3):
                overlap[a,b] = (masks[a] & masks[b]).sum() / (masks[a] | masks[b]).sum()
        stats[k] = {'jaccard':overlap.tolist(), 'selected_by_all_categories':int(masks.all(axis=0).sum()), 'selected_by_any_category':int(masks.any(axis=0).sum())}
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(overlap, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(3),labels); ax.set_yticks(range(3),labels)
        for a in range(3):
            for b in range(3): ax.text(b,a,f'{overlap[a,b]:.1%}',ha='center',va='center',color='white' if overlap[a,b]>.6 else 'black')
        fig.colorbar(im,ax=ax,label='Intersection / union of (layer, expert) pairs')
        ax.set_title(f'OLMoE top-{k}: category selection overlap')
        fig.tight_layout(); save(fig,f'olmoe_overlap_{k}')

    archive = read('results/local_124m_generation_20261008/report.json')
    model = next(m for m in archive['models'] if m['variant']=='domain_selected')
    cfg, names = model['config'], model['metadata']['pool_names']
    pool_ids = np.arange(cfg['n_routed_experts']) // cfg['experts_per_pool']
    fig, ax = plt.subplots(figsize=(11,5))
    ls = list(range(cfg['n_dense_layers'],cfg['n_layer']))
    im = ax.imshow(np.tile(pool_ids,(len(ls),1)),cmap=plt.get_cmap('tab10',len(names)),vmin=-.5,vmax=len(names)-.5,aspect='auto')
    ax.set_xticks(range(len(pool_ids))); ax.set_yticks(range(len(ls)),ls)
    ax.set(xlabel='Expert ID',ylabel='MoE layer',title='ASI checkpoint 37595: configured pool membership (not measured activations)')
    bar=fig.colorbar(im,ax=ax,ticks=range(len(names))); bar.ax.set_yticklabels(names)
    fig.tight_layout(); save(fig,'asi_pool_structure')
    report={'sources_sha256':provenance,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'deepseek_prompts':[p['prompt'] for p in probes], 'deepseek_pca_variance':variance.tolist(),
            'olmoe_categories':categories,'olmoe':stats,'asi_pool_names':names,
            'scope':'Offline archived JSON only. No checkpoints, torch, CUDA or pilot files loaded or modified.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
