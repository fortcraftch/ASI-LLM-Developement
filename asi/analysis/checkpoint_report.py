"""Summarize completed checkpoint benchmarks without loading any model."""
import argparse
import json
import math
import statistics
import copy
from pathlib import Path


def recover_native_suite(blocks, reference, expected_tasks):
    """Recover full saved tasks from interrupted native reports without editing them.

    Coverage and per-task counts must match an independently completed reference.
    This validates recorded evidence, not rerunning the original predictions.
    """
    if not blocks or reference.get('completed') is not True:
        raise ValueError('Completed reference and native blocks required')
    first = blocks[0]
    if not blocks[-1].get('completed'):
        raise ValueError('Continuation has not completed')
    result = copy.deepcopy(first)
    result['tasks'] = {}; result['datasets'] = {}
    for block in blocks:
        if block.get('kind') != 'native_deepseek_benchmark':
            raise ValueError('Only native reports can be recovered')
        for field in ('checkpoint_sha256', 'evaluator_sha256', 'precision', 'routing', 'runtime', 'parameters', 'limit'):
            if field not in first or block.get(field) != first[field]:
                raise ValueError('Native protocol mismatch: '+field)
        if block.get('metadata', {}).get('architecture_sha256') != first.get('metadata', {}).get('architecture_sha256') or not first.get('metadata', {}).get('architecture_sha256'):
            raise ValueError('Native architecture mismatch')
        if result['tasks'].keys() & block['tasks'].keys():
            raise ValueError('Overlapping task blocks')
        result['tasks'].update(copy.deepcopy(block['tasks']))
        for key, value in block.get('datasets', {}).items():
            if key in result['datasets'] and result['datasets'][key] != value:
                raise ValueError('Conflicting dataset identity')
            result['datasets'][key] = copy.deepcopy(value)
    result['completed'] = True
    compare_full_suite({'recovered': [result], 'reference': [reference]}, expected_tasks)
    result['recovery_note'] = 'Aggregated saved full tasks; original interrupted reports remain unchanged. Not a single uninterrupted timing run.'
    for field in ('memory', 'benchmark_peak_cuda_bytes', 'routing_audit'):
        result.pop(field, None)
    return result


def compare_full_suite(variants, expected_tasks):
    """Validate complete local suites before comparing quality, never timing.

    Each variant supplies disjoint report blocks. Checkpoint-eval blocks must
    contain exactly one mode so the caller cannot silently select a policy.
    This verifies recorded provenance, not training equivalence or input order.
    """
    expected = set(expected_tasks)
    if not expected or len(variants) < 2:
        raise ValueError('Require tasks and at least two variants')
    merged = {}
    common_evaluator = None
    common_datasets = None
    for label, blocks in variants.items():
        tasks, datasets = {}, {}
        identity = None
        for report in blocks:
            if report.get('completed') is not True or report.get('limit', 'missing') is not None:
                raise ValueError('Require completed full reports: '+label)
            evaluator = report.get('evaluator_sha256')
            checkpoint = report.get('checkpoint_sha256')
            if not evaluator or not checkpoint:
                raise ValueError('Missing evaluator/checkpoint identity: '+label)
            if common_evaluator is None:
                common_evaluator = evaluator
            if evaluator != common_evaluator:
                raise ValueError('Different evaluators')
            if identity is None:
                identity = checkpoint
            if checkpoint != identity:
                raise ValueError('Different checkpoints within variant: '+label)
            if 'results' in report:
                if len(report['results']) != 1:
                    raise ValueError('Select exactly one mode: '+label)
                values = next(iter(report['results'].values()))['tasks']
            else:
                values = report['tasks']
            if tasks.keys() & values.keys():
                raise ValueError('Overlapping task blocks: '+label)
            for task, result in values.items():
                if 'error' in result or result.get('total', 0) <= 0:
                    raise ValueError('Invalid task result: '+task)
                score = result.get('accuracy_norm', result.get('accuracy'))
                if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid accuracy: '+task)
            tasks.update(values)
            for key, fingerprint in report.get('datasets', {}).items():
                if key in datasets and datasets[key] != fingerprint:
                    raise ValueError('Dataset changed across blocks: '+key)
                datasets[key] = fingerprint
        if set(tasks) != expected or not datasets:
            raise ValueError('Incomplete task or dataset coverage: '+label)
        if common_datasets is None:
            common_datasets = datasets
        if datasets != common_datasets:
            raise ValueError('Different dataset provenance: '+label)
        merged[label] = tasks
    for task in expected:
        if len({tasks[task]['total'] for tasks in merged.values()}) != 1:
            raise ValueError('Different example counts: '+task)
        if len({'accuracy_norm' in tasks[task] for tasks in merged.values()}) != 1:
            raise ValueError('Different accuracy definitions: '+task)
    return merged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--gpt2-report',type=Path)
    p.add_argument('--sft-report',type=Path,help='Additional evaluation of the fine-tuned checkpoint, with identical benchmark settings')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.report.read_text(encoding='utf-8'))
    if not data.get('completed'): p.error('Checkpoint evaluation is incomplete')
    modes=data['results']
    if a.sft_report:
        sft=json.loads(a.sft_report.read_text(encoding='utf-8'))
        if not sft.get('completed'): p.error('SFT evaluation is incomplete')
        if sft.get('probe_format','completion')!=data.get('probe_format','completion'):
            p.error('Different qualitative prompt formats')
        for key in ['limit','evaluator_sha256','datasets','trained_routing','input_routes']:
            if sft.get(key)!=data.get(key): p.error('SFT comparison differs in '+key)
        # The original evaluator used these fixed settings before recording flags.
        for key,legacy_default in [('probe_categories','predicted'),('generation_repeats',3),('new_tokens',64)]:
            if sft.get(key,legacy_default)!=data.get(key,legacy_default):
                p.error('SFT comparison differs in '+key)
        provenance=sft.get('metadata',{}).get('sft_provenance',{})
        if provenance.get('source_sha256')!=data['checkpoint_sha256']:
            p.error('SFT checkpoint does not identify this base checkpoint')
        modes={**modes,**{'sft_'+k:v for k,v in sft['results'].items()}}
    scores={k:v['tasks'] for k,v in modes.items()}
    if a.gpt2_report:
        baseline=json.loads(a.gpt2_report.read_text(encoding='utf-8'))
        if baseline['meta']['limit']!=data['limit']: p.error('Different benchmark limits')
        scores.update({Path(k).stem:v['tasks'] for k,v in baseline['models'].items()})
    tasks=list(next(iter(scores.values())))
    if not tasks: p.error('This summary requires benchmark tasks, not probe-only reports')
    if any(set(v)!=set(tasks) for v in scores.values()): p.error('Different task sets')
    if any('error' in task for model in scores.values() for task in model.values()):
        p.error('Resolve benchmark errors before comparing')
    for task in tasks:
        if len({model[task]['total'] for model in scores.values()})!=1:
            p.error('Different evaluated example counts')
    a.output.mkdir(parents=True,exist_ok=True)
    labels={'classified':'Categoría (2 activos)', 'all_uniform':'Uniforme (16 activos)',
            'all_router':'Router sin entrenar (2 activos)', 'gpt2Base':'GPT-2 propio'}
    if data['trained_routing']!='uniform_pool':
        labels['all_router']='Router aprendido (2 activos)'
    labels.update({'sft_'+k:'SFT · '+v for k,v in list(labels.items())})
    checkpoint_step=data['metadata']['step']
    sample_description='Datasets completos (aritmética: 500 casos)' if data['limit'] is None else f"Primeros {data['limit']} ejemplos por tarea"
    lines=[f'# Evaluación del checkpoint {checkpoint_step}', '',
           sample_description+', mismas funciones del evaluador GPT-2.',
           'HellaSwag usa accuracy_norm; el resto accuracy. Son las métricas del evaluador local, no necesariamente el protocolo oficial de cada benchmark.', '',
           '| Variante | '+' | '.join(tasks)+' |','|---|'+'---:|'*len(tasks)]
    for mode,values in scores.items():
        lines.append('| '+labels.get(mode,mode)+' | '+' | '.join(f"{100*values[t].get('accuracy_norm',values[t]['accuracy']):.1f}%" for t in tasks)+' |')
    if a.sft_report:
        lines+=['',f"SFT: checkpoint seleccionado en actualización {sft['metadata']['step']}; procedencia verificada contra el checkpoint original. Datos OASST y ajuste de pesos completos con pools fijos. Su pérdida de validación usa tokens del asistente y no es directamente comparable con la pérdida de preentrenamiento."]
        if 'new_tokens' not in data:
            lines+=['El informe original precede a los campos de configuración de generación; se usan los valores fijos históricos: categorías predichas, tres repeticiones y 64 tokens.']
    if 'lambada' in tasks:
        lines+=['','Perplejidad sobre la palabra objetivo de LAMBADA (menor es mejor): '+
                '; '.join(f"{labels.get(mode,mode)} = {values['lambada']['perplexity']:.2f}" for mode,values in scores.items())+'.']
    lines+=['','![Comparación de aciertos](accuracy.png)']
    lines+=['','## Memoria y velocidad de generación','',
            '| Variante | Expertos GPU/capa | Pesos GPU MiB | Pico asignado MiB | Mediana tokens/s | Cargas durante respuesta |',
            '|---|---:|---:|---:|---:|---:|']
    summary={}
    for mode,r in modes.items():
        inv=r['inventory']
        hot=[e for e in inv['experts'] if e['device'].startswith('cuda')]
        layers={e['layer'] for e in hot}
        weights=sum(e['execution_weight_bytes'] for e in hot)+sum(v for k,v in inv['backbone_and_shared_weight_bytes'].items() if k.startswith('cuda'))
        gen=r['generations']
        speed=statistics.median(g['tokens_per_second'] for g in gen)
        peak=inv.get('cuda',{}).get('peak_allocated_bytes',0)
        loads=sum(g['loads_during_generation'] for g in gen)
        summary[mode]={'gpu_weights_bytes':weights,'peak_cuda_bytes':peak,'median_tokens_per_second':speed,
                       'loads_during_generation':loads,'resident_per_layer':len(hot)/len(layers) if layers else 0}
        lines.append(f"| {labels.get(mode,mode)} | {summary[mode]['resident_per_layer']:g} | {weights/2**20:.2f} | {peak/2**20:.2f} | {speed:.2f} | {loads} |")
    lines+=['','![Residencia y velocidad](residency_speed.png)','',f"Generación greedy incremental: 8 prefijos × {data.get('generation_repeats',3)} repeticiones, hasta {data.get('new_tokens',64)} tokens. Las repeticiones miden latencia; no son preguntas independientes.",
            'Tiempo de generación excluye clasificación y carga inicial. Pesos en FP32, autocast BF16, backend bmm. Pico incluye buffers y temporales; no es solo el modelo.',
            'Las mediciones de variantes pueden proceder de sesiones distintas. Una diferencia de tokens/s entre original y SFT no demuestra aceleración causada por el ajuste; faltan repeticiones intercaladas bajo la misma carga del equipo.',
            'En all_router todos los expertos residen en GPU, aunque solo se activen dos por token. No se ha entrenado ese router.',
            'El GPT-2 se compara en calidad; su implementación y precisión son distintas, por lo que sus tiempos de benchmark no son un control de velocidad.',
            'Las barras de error son intervalos Wilson marginales del 95%, descriptivos. No sustituyen una prueba pareada; los primeros N ejemplos no son una muestra aleatoria.', '',
            '## Textos para revisión', '', 'No se han asignado puntuaciones automáticas de coherencia ni inteligencia.']
    review=[]
    for mode,r in modes.items():
        lines+=['',f'### {labels.get(mode,mode)}','']
        for g in r['generations']:
            if g['repeat']: continue
            lines += ['**'+g['prompt']+'**','',g['text'].replace('\n','  \n'),'']
            review.append({'mode':mode,'prompt':g['prompt'],'response':g['text'],
                           'correctness':None,'coherence':None})
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.output/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(12,5))
    width=.8/len(scores); x=np.arange(len(tasks))
    for i,(mode,values) in enumerate(scores.items()):
        means=[]; low=[]; high=[]
        for task in tasks:
            r=values[task]; v=r.get('accuracy_norm',r['accuracy']); n=r['total']; z=1.96
            center=(v+z*z/(2*n))/(1+z*z/n)
            half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/(1+z*z/n)
            means.append(100*v); low.append(max(0,100*(v-center+half))); high.append(max(0,100*(center+half-v)))
        ax.bar(x-.4+width/2+i*width,means,width,label=labels.get(mode,mode),yerr=[low,high],capsize=2)
    ax.set_xticks(x,tasks); ax.set_ylim(0,100); ax.set_ylabel('Aciertos (%)')
    ax.set_title(f"Checkpoint {checkpoint_step} · {sample_description}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(a.output/'accuracy.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=[labels.get(k,k) for k in summary]
    axes[0].bar(range(len(names)),[v['gpu_weights_bytes']/2**20 for v in summary.values()])
    axes[0].set_ylabel('Pesos residentes GPU (MiB)')
    axes[1].bar(range(len(names)),[v['median_tokens_per_second'] for v in summary.values()])
    axes[1].set_ylabel('Mediana tokens/s (generación)')
    for ax in axes: ax.set_xticks(range(len(names)),names,rotation=15,ha='right',fontsize=8)
    fig.tight_layout();fig.savefig(a.output/'residency_speed.png',dpi=170);plt.close(fig)
    print(a.output/'summary.md')


if __name__=='__main__': main()
