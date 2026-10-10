"""Compare paired capacity pilots across initialization seeds, with frozen data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def routing_metrics(layers,previous=None):
    effective=[];dependence=[];mass=[]
    for key,layer in layers.items():
        before=(previous or {}).get(key,{})
        counts=np.asarray(layer['assignments_by_input_pool'],dtype=float)-np.asarray(before.get('assignments_by_input_pool',0))
        if (counts<0).any() or counts.sum()<=0:raise ValueError('Invalid routing interval')
        joint=counts/counts.sum();experts=joint.sum(0);pools=joint.sum(1)
        effective.append(float(np.exp(-sum(p*np.log(p) for p in experts if p))))
        expected=pools[:,None]*experts[None,:];valid=joint>0
        mi=float((joint[valid]*np.log(joint[valid]/expected[valid])).sum())
        pool_entropy=-sum(p*np.log(p) for p in pools if p)
        dependence.append(mi/pool_entropy if pool_entropy else 0.)
        observed=layer['observed_tokens']-before.get('observed_tokens',0)
        mass.append((layer['mixture_weight_sum']-before.get('mixture_weight_sum',0))/observed)
    return {'effective_experts':float(np.mean(effective)),
            'normalized_pool_information':float(np.mean(dependence)),
            'selected_weight_mass':float(np.mean(mass))}


def window_identity(path):
    h=hashlib.sha256()
    with np.load(path) as data:
        for key in sorted(data.files):
            arr=data[key];h.update(key.encode());h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode());h.update(arr.tobytes())
    return h.hexdigest()


def compare(paths,output):
    rows=[];identities=set();seeds=set();common_settings=None;source_identities=set()
    for path in paths:
        plan=json.loads((path/'plan.json').read_text(encoding='utf8'))
        if plan['status']!='completed':raise ValueError(f'Incomplete pilot: {path}')
        seed=plan['settings']['seed']
        if seed in seeds:raise ValueError('Duplicate initialization seed')
        seeds.add(seed)
        source_identities.add(tuple(plan[k] for k in ('source_recipe_sha256','pool_manifest_sha256','local_manifest_sha256')))
        settings={k:v for k,v in plan['settings'].items() if k!='seed'}
        if common_settings is not None and settings!=common_settings:raise ValueError('Unmatched settings')
        common_settings=settings
        identities.add(window_identity(path/'windows.npz'))
        pair=[]
        for name in plan['variants']:
            r=json.loads((path/(name+'.json')).read_text(encoding='utf8'))
            if r['status']!='completed' or not r.get('usage_history'):raise ValueError('Missing completed usage history')
            history=[];previous=None
            for snapshot in r['usage_history']:
                history.append({'updates':snapshot['updates'],**routing_metrics(snapshot['layers'],previous)})
                previous=snapshot['layers']
            earlier=r['usage_history'][-2]['layers'] if len(r['usage_history'])>1 else {}
            final_counts={key:(np.asarray(layer['assignments_by_input_pool'])-
                np.asarray(earlier.get(key,{}).get('assignments_by_input_pool',0))).sum(0).tolist()
                for key,layer in r['usage_history'][-1]['layers'].items()}
            pair.append(r)
            rows.append({'seed':seed,'name':name,'routing':r['variant']['model']['routing_mode'],
                'tokens':r['tokens'],'validation':r['validation'],'routing_intervals':history,
                'last_interval_assignments_by_layer':final_counts,
                'source':str(path/(name+'.json'))})
        if len(pair)!=2 or {r['variant']['model']['routing_mode'] for r in pair}!={'uniform_pool','learned'}:
            raise ValueError('Expected one restricted/free pair per seed')
        if len({r['common_initial_weights_sha256'] for r in pair})!=1:raise ValueError('Initial common weights differ')
        configs=[{k:v for k,v in r['variant']['model'].items() if k!='routing_mode'} for r in pair]
        if configs[0]!=configs[1]:raise ValueError('Architectures differ beyond routing')
    if len(identities)!=1:raise ValueError('Frozen windows differ across seeds')
    if len(source_identities)!=1:raise ValueError('Source identities differ across seeds')
    if len({r['tokens'] for r in rows})!=1:raise ValueError('Token budgets differ')
    output.mkdir(parents=True,exist_ok=True)
    differences=[]
    for seed in sorted(seeds):
        pair={r['routing']:r for r in rows if r['seed']==seed}
        differences.append(pair['uniform_pool']['validation'][-1]['mean']-pair['learned']['validation'][-1]['mean'])
    report={'kind':'paired_early_training_followup','seeds':sorted(seeds),'settings':common_settings,
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'window_content_sha256':next(iter(identities)),'rows':rows,
        'nll_restricted_minus_free':differences,'mean_nll_difference':float(np.mean(differences)),
        'sample_std_nll_difference':float(np.std(differences,ddof=1)) if len(differences)>1 else None,
        'limitation':'Early optimization, not language quality; descriptive seed variation, not a significance claim.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# Punto 9: seguimiento pareado del routing','',
        f"{len(seeds)} semillas de inicialización, ventanas idénticas, {rows[0]['tokens']:,} tokens por variante.",
        'El estado final de cada fila incluye solo el último intervalo de entrenamiento para las métricas de routing.','',
        '| Semilla | Política | NLL final | Expertos efectivos | Asociación con pool | Suma de pesos |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        m=r['routing_intervals'][-1]
        lines.append(f"| {r['seed']} | {r['routing']} | {r['validation'][-1]['mean']:.4f} | {m['effective_experts']:.2f} | {m['normalized_pool_information']:.4f} | {m['selected_weight_mass']:.3f} |")
    lines+=['',f"Diferencia pareada NLL (restringido − libre): {differences}. Media: {np.mean(differences):.4f}. No es una prueba de significación.",
        '', 'Expertos efectivos: concentración del reparto, no cantidad activada por token. Asociación: información mutua normalizada por la entropía del pool; cero indica reparto independiente de la categoría, uno separación completa. No demuestra especialización semántica.',
        ('Control de normalización activo: los pesos top-k aprendidos se renormalizan a route_scale, igualando la suma de la rama uniforme.' if common_settings.get('normalize_selected') else 'Suma de pesos: cuánto peso total recibe la rama de expertos enrutados. La política softmax libre puede sumar menos de uno; el pool uniforme suma uno. La comparación incluye esta diferencia, además de la selección.'),
        'Las curvas muestran medias y rangos entre semillas, no intervalos de confianza. No usar este ensayo corto para elegir un mínimo de parámetros para lenguaje.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    specs=[('validation','mean','NLL de validación'),('routing_intervals','effective_experts','Expertos efectivos por capa'),
           ('routing_intervals','normalized_pool_information','Asociación entre pool y experto'),('routing_intervals','selected_weight_mass','Suma de pesos seleccionados')]
    for ax,(field,key,label) in zip(axes.flat,specs):
        for policy,title in [('uniform_pool','Pool restringido'),('learned','Routing libre')]:
            series=[r[field] for r in rows if r['routing']==policy]
            x=[s['updates'] for s in series[0]]
            if any([s['updates'] for s in v]!=x for v in series):raise ValueError('Mismatched evaluation points')
            y=np.array([[s[key] for s in v] for v in series])
            line=ax.plot(x,y.mean(0),'-o',label=title)[0]
            ax.fill_between(x,y.min(0),y.max(0),color=line.get_color(),alpha=.15)
        ax.set_xlabel('Actualizaciones');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Aprendizaje temprano: media y rango entre semillas; no evalúa lenguaje')
    fig.tight_layout();fig.savefig(output/'routing_followup.png',dpi=160);plt.close(fig)
    lines+=['','![Evolución](routing_followup.png)']
    free=[r for r in rows if r['routing']=='learned']
    fig,axes=plt.subplots(1,len(free),figsize=(5*len(free),4),squeeze=False,layout='constrained')
    for ax,r in zip(axes.flat,free):
        keys=sorted(r['last_interval_assignments_by_layer'],key=int)
        values=np.array([r['last_interval_assignments_by_layer'][k] for k in keys],dtype=float)
        shares=values/values.sum(1,keepdims=True)
        im=ax.imshow(shares,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set_title(f"Routing libre, semilla {r['seed']}");ax.set_xlabel('Índice de experto (propio de esta semilla)')
        ax.set_yticks(range(len(keys)),keys);ax.set_ylabel('Capa MoE')
    fig.colorbar(im,ax=list(axes.flat),label='Fracción de asignaciones en el último intervalo')
    fig.savefig(output/'expert_concentration.png',dpi=160);plt.close(fig)
    lines+=['','![Reparto por capa](expert_concentration.png)',
            'Los índices de expertos no se alinean semánticamente entre semillas. Cada mapa se interpreta por separado.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return report


def compare_normalization(baselines, normalized, output):
    """Paired intervention comparison; keep every setting except normalization fixed."""
    def index(paths):
        result={}
        for path in paths:
            plan=json.loads((path/'plan.json').read_text(encoding='utf8'))
            seed=plan['settings']['seed']
            if seed in result or plan['status']!='completed': raise ValueError('Duplicate seed or incomplete normalization run')
            result[seed]=(path,plan)
        return result
    old,new=index(baselines),index(normalized)
    if not old or set(old)!=set(new): raise ValueError('Normalization seeds differ')
    rows=[]
    for seed in sorted(old):
        bp,b=old[seed];np_,n=new[seed]
        if b['settings'].get('normalize_selected',False) or not n['settings'].get('normalize_selected',False):
            raise ValueError('Expected native baseline and normalized intervention')
        clean=lambda p:{k:v for k,v in p['settings'].items() if k!='normalize_selected'}
        if clean(b)!=clean(n): raise ValueError('Settings differ beyond normalization')
        if any(b[k]!=n[k] for k in ('source_recipe_sha256','pool_manifest_sha256','local_manifest_sha256','variants')):
            raise ValueError('Input identities differ')
        if window_identity(bp/'windows.npz')!=window_identity(np_/'windows.npz'):
            raise ValueError('Normalization windows differ')
        for name in b['variants']:
            before=json.loads((bp/(name+'.json')).read_text(encoding='utf8'))
            after=json.loads((np_/(name+'.json')).read_text(encoding='utf8'))
            if before['status']!='completed' or after['status']!='completed': raise ValueError('Incomplete variant')
            if any(before[k]!=after[k] for k in ('variant','common_initial_weights_sha256','tokens')):
                raise ValueError('Architecture, initialization or token budget differs')
            metrics=[]
            for r in (before,after):
                history=r['usage_history']
                metrics.append(routing_metrics(history[-1]['layers'],history[-2]['layers'] if len(history)>1 else None))
            rows.append({'seed':seed,'routing':before['variant']['model']['routing_mode'],
                         'baseline_nll':before['validation'][-1]['mean'],'normalized_nll':after['validation'][-1]['mean'],
                         'nll_delta':after['validation'][-1]['mean']-before['validation'][-1]['mean'],
                         'baseline_routing':metrics[0],'normalized_routing':metrics[1],
                         'baseline_report':str(bp/(name+'.json')),'normalized_report':str(np_/(name+'.json'))})
    learned=[r for r in rows if r['routing']=='learned']
    result={'kind':'normalization_intervention','rows':rows,'mean_learned_nll_delta':float(np.mean([r['nll_delta'] for r in learned])),
            'limitations':'Early optimization only. Seeds share one corpus sample; no language threshold or statistical significance claim.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'normalization_effect.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    lines=['# Efecto de normalizar los pesos seleccionados','','Mismos datos, semillas y presupuesto. Delta = normalizado menos original; menor NLL es mejor.','',
           '| Semilla | Routing | NLL original | NLL normalizada | Delta |','|---|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['seed']} | {r['routing']} | {r['baseline_nll']:.4f} | {r['normalized_nll']:.4f} | {r['nll_delta']:+.4f} |")
    lines+=['',f"Delta medio del router aprendido: {result['mean_learned_nll_delta']:+.4f}.",
            'El pool uniforme es un control sin intervención: su diferencia permite detectar variación ajena a la normalización.',
            'No establece un mínimo de parámetros ni mide calidad de conversación. Ver normalization_effect.json para concentración y suma de pesos.']
    (output/'normalization_effect.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-runs',type=Path,nargs='+',help='Optional matched native runs for the normalization intervention')
    a=p.parse_args();compare(a.runs,a.output)
    if a.baseline_runs: compare_normalization(a.baseline_runs,a.runs,a.output)


if __name__=='__main__':main()
