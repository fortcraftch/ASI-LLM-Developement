"""Plot matched local-model generations without loading checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

LABELS = {'deepseek':'DeepSeek\nfree top-2', 'domain_selected':'ASI pool\n2 resident / 2 active',
          'domain_all_resident':'ASI pool\n16 resident / 2 active',
          'domain_all_uniform':'ASI all-uniform\n16 resident / 16 active', 'gpt2':'GPT-2\ndense'}


def model_label(model, compact=False):
    kind=model['variant']
    if kind=='gpt2': return LABELS[kind]
    config=model['config']
    resident=config['experts_per_pool'] if kind=='domain_selected' else config['n_routed_experts']
    active=config['n_routed_experts'] if kind=='domain_all_uniform' else config['n_activated_experts']
    name='DeepSeek free' if kind=='deepseek' else ('ASI all-uniform' if kind=='domain_all_uniform' else 'ASI pool')
    if compact:
        name='DeepSeek' if kind=='deepseek' else ('ASI uniform' if kind=='domain_all_uniform' else 'ASI pool')
        return f'{name}\n{resident}R / {active}A'
    return f'{name}\n{resident} resident\n{active} active'


def summarize(data):
    if not data.get('completed') or data.get('kind')!='local_124m_generation':
        raise ValueError('Expected a completed local generation report')
    expected = [(p['prompt'], repeat) for p in data['probes'] for repeat in range(data['settings']['repeats'])]
    reference = None
    rows = []
    for model in data['models']:
        runs = model['runs']
        if [(r['prompt'],r['repeat']) for r in runs] != expected:
            raise ValueError('Mismatched prompts or repeats')
        inputs = [r['input_ids'] for r in runs]
        if reference is None: reference = inputs
        if inputs != reference: raise ValueError('Mismatched tokenized inputs')
        if any(r['stop_at_eos'] or r['generated_tokens']!=data['settings']['new_tokens'] for r in runs):
            raise ValueError('Timing workload must have identical fixed token count')
        rates=[r['tokens_per_second'] for r in runs]
        peaks=[r['peak_cuda_allocated_bytes'] for r in runs if r['peak_cuda_allocated_bytes'] is not None]
        rows.append({'variant':model['variant'], 'parameters':model['parameters'],
                     'weight_mib':model['parameter_bytes']/2**20,
                     'gpu_weight_mib':max(sum(v for k,v in r['memory']['weight_bytes_by_device'].items() if k.startswith('cuda')) for r in runs)/2**20,
                     'peak_mib':max(peaks)/2**20 if peaks else None,
                     'checkpoint_gib':model['checkpoint_file_bytes']/2**30,
                     'median_tokens_s':statistics.median(rates),'min_tokens_s':min(rates),'max_tokens_s':max(rates),
                     'median_first_token_ms':1000*statistics.median(r['first_token_seconds'] for r in runs),
                     'loads':sum(r['loads_during_generation'] for r in runs),
                     'generation_h2d_bytes':sum(r['h2d_bytes_during_generation'] for r in runs)})
    if not rows: raise ValueError('No models')
    return rows


def build(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    source,output=Path(source),Path(output)
    payload=source.read_bytes();data=json.loads(payload)
    rows=summarize(data);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':13,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.2))
    labels=[model_label(m,compact=True) for m in data['models']];x=np.arange(len(rows))
    colors=['#5b6472','#0072b2','#56b4e9','#d89000','#009e73'][:len(rows)]
    for ax,key,title,unit in zip(axes.flat,['gpu_weight_mib','peak_mib','median_tokens_s','median_first_token_ms'],
            ['GPU-resident model weights','Peak allocated CUDA memory','Generation throughput','Time to first token'],
            ['MiB','MiB','tokens/s','ms']):
        vals=[r[key] if r[key] is not None else 0 for r in rows]
        bars=ax.bar(x,vals,color=colors)
        if key=='median_tokens_s':
            ax.errorbar(x,vals,yerr=[[r[key]-r['min_tokens_s'] for r in rows],[r['max_tokens_s']-r[key] for r in rows]],fmt='none',color='black',capsize=3)
        ax.bar_label(bars,labels=['N/A' if r[key] is None else f'{r[key]:.1f}' for r in rows],padding=3,fontsize=12)
        ax.set_xticks(x,labels,fontsize=12);ax.set_title(title);ax.set_ylabel(unit)
        upper=max([r['max_tokens_s'] for r in rows]) if key=='median_tokens_s' else max(vals+[1])
        ax.set_ylim(0,upper*1.24);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Local ~124M models: matched full-context generation',fontsize=14)
    fig.text(.5,.015, f"{len(data['probes'])} prompts × {data['settings']['repeats']} repeats × {data['settings']['new_tokens']} tokens; FP32 / BF16 autocast; batch 1; no KV fast path.\nR = resident routed experts; A = active routed experts. Shared branches are additional.\nObserved ranges, not confidence intervals. Classification and preparation excluded. Unequal training budgets.",ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.095,1,.96))
    for extension in ('pdf','png'): fig.savefig(output/f'local_124m_performance.{extension}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    by_kind={m['variant']:m for m in data['models']}
    controls={}
    if {'domain_selected','domain_all_resident'} <= by_kind.keys():
        selected=by_kind['domain_selected'];resident=by_kind['domain_all_resident']
        if selected['metadata']['checkpoint_sha256'] != resident['metadata']['checkpoint_sha256']:
            raise ValueError('Residency control must use identical checkpoint weights')
        controls['pool_residency_identical_tokens']=all(a['output_ids']==b['output_ids']
            for a,b in zip(selected['runs'],resident['runs']))
        controls['pool_residency_identical_eos_outputs']=all(a['output_ids']==b['output_ids']
            for a,b in zip(selected['outputs'],resident['outputs']))
    summary={'source_sha256':hashlib.sha256(payload).hexdigest(),'protocol':data['protocol'],'rows':rows,
             'controls':controls}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    lines=['# Modelos locales: memoria y generación','',data['protocol'],'',
           '| Variante | Parámetros M | Pesos totales MiB | Pesos GPU MiB | Pico CUDA MiB | Checkpoint GiB | tokens/s | TTFT ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        peak='N/A' if r['peak_mib'] is None else f"{r['peak_mib']:.2f}"
        lines.append(f"| {r['variant']} | {r['parameters']/1e6:.2f} | {r['weight_mib']:.2f} | {r['gpu_weight_mib']:.2f} | {peak} | {r['checkpoint_gib']:.3f} | {r['median_tokens_s']:.2f} | {r['median_first_token_ms']:.2f} |")
    lines+=['','![Mediciones](local_124m_performance.png)','','Control de residencia (mismo checkpoint, mismo pool): '+json.dumps(controls),'','## Límites','']+['- '+s for s in data['limitations']]
    lines+=['','No se deduce corrección ni inteligencia de estos tiempos. Los cambios de expertos activos no son transferencias si sus pesos ya están en GPU.','',
            '## Generaciones separadas con parada EOS','',
            'Los textos siguientes respetan EOS; las secuencias del microbenchmark de tiempo no se usan como evidencia de calidad.']
    for m in data['models']:
        lines+=['','### '+m['variant']]
        for r in m['outputs']:
            lines+=['','**Prompt:** '+r['prompt'],'','    '+r['text'].replace('\n','\n    '),'',
                    f"Parada: {r['stop_reason']}; {r['generated_tokens']} tokens."]
    (output/'review.md').write_text('\n'.join(lines),encoding='utf8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.report,a.output)


if __name__=='__main__':main()
