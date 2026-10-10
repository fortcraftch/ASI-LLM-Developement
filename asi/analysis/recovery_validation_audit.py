"""Offline integrity, code-format and category-transition audit of recovery runs.

Never modifies raw reports, executes generated code, or assigns semantic scores.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import statistics

from asi.analysis.recovery_validation_report import validate
from asi.experiments.recovery_protocol import task_metrics
from asi.experiments.recovery_validation import read, fingerprint
from asi.taxonomy import digest


def code_diagnostics(text):
    blocks = re.findall(r'^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?$)', text, re.M | re.S)
    valid = None
    if blocks:
        try:
            if not blocks[0].strip():
                raise SyntaxError('Empty block')
            ast.parse(blocks[0])
            valid = True
        except (SyntaxError, ValueError, TypeError):
            valid = False
    return {'complete_python_blocks': len(blocks), 'first_block_syntax_pass': valid,
            'note': 'Syntax of first complete Python fence only; not format compliance or functional correctness.'}


def audit(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError('Use a fresh audit directory')
    plan = read(directory/'plan.json')
    reports = {v:read(directory/v/'report.json') for v in plan['variants']}
    validate(reports, plan, fingerprint(directory/'plan.json'))
    for name,path in plan['sources'].items():
        if fingerprint(path) != plan['hashes'][name]:
            raise ValueError('Frozen source changed: '+name)
    corpus = read(plan['sources']['corpus'])
    study = read(plan['sources']['study'])
    rows = {r['id']:r for r in corpus['examples']}
    subset = set(read(plan['sources']['subset']))
    config = read('public_models/olmoe/config.json')
    eos = config['eos_token_id']
    results = {}
    for name,report in reports.items():
        if report['corpus_identity'] != digest(corpus) or report['mapping_identity'] != digest(study['mappings']):
            raise ValueError('Wrong corpus or calibration identity')
        if report['source_identity'] != study['source_identity']:
            raise ValueError('Wrong model source')
        if name != 'original' and report['mask_identity'] != digest(study['mappings'][name[-2:]]):
            raise ValueError('Wrong resident mask')
        if name == 'original' or name.startswith('unadapted'):
            if report['checkpoint_sha256'] is not None or report['checkpoint'] is not None:
                raise ValueError('Unadapted control contains an adaptation')
        expected_capacity = plan['native_cache_experts'] if name=='original' else int(name[-2:])*config['num_hidden_layers']
        if report['cache_capacity_pairs'] != expected_capacity:
            raise ValueError('Unexpected cache capacity')
        code=[]; transitions=[]
        previous=None
        for row in report['records']:
            reference=rows[row['id']]
            if reference['split']!='test' or reference['id'] in subset:
                raise ValueError('Not an independent held-out test example')
            if any(row[k]!=reference[k] for k in ('prompt','category')) or row['reference_answer']!=reference['answer']:
                raise ValueError('Changed corpus content')
            tokens=row['token_ids']
            if not 0<len(tokens)<=plan['new_tokens']:
                raise ValueError('Invalid generated length')
            expected_stop='EOS' if tokens[-1]==eos else 'token_budget'
            if row['stop_reason']!=expected_stop or (expected_stop=='token_budget' and len(tokens)!=plan['new_tokens']):
                raise ValueError('Invalid stopping record')
            if eos in tokens[:-1]:
                raise ValueError('Generation continued past EOS')
            triples=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
            repetition=1-len(set(triples))/len(triples) if triples else 0.
            if abs(repetition-row['metrics']['repeated_trigram_fraction'])>1e-12:
                raise ValueError('Repetition metric differs from token IDs')
            for key,value in task_metrics(row['text'],reference).items():
                if row['metrics'][key]!=value:
                    raise ValueError('Task metric differs: '+key)
            if row['peak_cuda_allocated_bytes']>plan['gpu_allocator_cap_gib']*2**30:
                raise ValueError('Measured GPU allocation exceeds budget')
            cache=row['cache_after']
            if cache['disk_ram']['ram_weight_bytes']>plan['expert_ram_gib']*2**30:
                raise ValueError('Expert RAM cache exceeds budget')
            if cache['resident_experts']>expected_capacity:
                raise ValueError('Expert cache exceeds capacity')
            if row['previous_category']!=previous or row['category_changed']!=(previous is not None and previous!=row['category']):
                raise ValueError('Incorrect category transition metadata')
            if reference['evaluation']=='python_syntax_and_manual_tests':
                code.append({'id':row['id'],'historical_syntax':row['metrics']['python_syntax_pass'],
                             'strict_format':row['metrics']['format_pass'],**code_diagnostics(row['text'])})
            transitions.append({'id':row['id'],'from':previous,'to':row['category'],
                                'kind':'initial_load' if previous is None else 'category_change' if row['category_changed'] else 'same_category',
                                'prepare_seconds':row['prepare_seconds'],
                                'prepare_h2d_gib':row['prepare_transfers']['host_to_device_bytes']/2**30,
                                'prepare_loads':row['prepare_transfers']['loads'],
                                'generation_h2d_gib':row['generation_transfers']['host_to_device_bytes']/2**30,
                                'generation_loads':row['generation_transfers']['loads']})
            previous=row['category']
        results[name]={'code_diagnostics':code,'transitions':transitions}
    paired={}
    for size in ('32','16'):
        if 'unadapted'+size not in reports:continue
        a,b=reports['unadapted'+size]['records'],reports['recovered'+size]['records']
        deltas=[y['metrics']['repeated_trigram_fraction']-x['metrics']['repeated_trigram_fraction'] for x,y in zip(a,b)]
        paired[size]={'repetition_decreased':sum(d<0 for d in deltas),'repetition_increased':sum(d>0 for d in deltas),
                      'median_paired_delta':statistics.median(deltas),'examples':len(deltas)}
    result={'completed':True,'raw_reports_modified':False,'generated_code_executed':False,
            'sources':{str(directory/v/'report.json'):fingerprint(directory/v/'report.json') for v in reports},
            'plan_sha256':fingerprint(directory/'plan.json'),'variants':results,'paired_repetition':paired,
            'semantic_quality':'Not scored automatically. Syntax, EOS and phrase presence are not correctness.'}
    output.mkdir(parents=True)
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['# Auditoría de recuperación','',
           'Integridad, fuentes congeladas, test independiente, máscaras, métricas, EOS y presupuestos comprobados. '
           'Los informes originales no se modifican. No se ejecuta código generado.','',
           '## Sintaxis y formato son indicadores distintos','',
           'El indicador histórico intenta analizar toda la respuesta cuando incumple el formato. '
           'El nuevo diagnóstico analiza solo el primer bloque Python completo; tampoco demuestra corrección funcional.','',
           '| Variante | Formato estricto | Sintaxis del primer bloque completo | Bloque ausente |',
           '|---|---:|---:|---:|']
    for name,value in results.items():
        code=value['code_diagnostics'];n=len(code)
        lines.append(f"| {name} | {sum(x['strict_format'] for x in code)}/{n} | {sum(x['first_block_syntax_pass'] is True for x in code)}/{n} | {sum(x['first_block_syntax_pass'] is None for x in code)} |")
    lines+=['','## Cambios de categoría','',
            'Las transferencias de preparación incluyen carga inicial y cambios de categoría; '
            'se separan de las transferencias dentro de cada respuesta. Son preguntas independientes, no chat con historial.','',
            '| Variante | Preparación total GiB | Generación total GiB | Cargas al generar |',
            '|---|---:|---:|---:|']
    for name,value in results.items():
        t=value['transitions']
        lines.append(f"| {name} | {sum(x['prepare_h2d_gib'] for x in t):.2f} | {sum(x['generation_h2d_gib'] for x in t):.2f} | {sum(x['generation_loads'] for x in t)} |")
    lines+=['','## Repetición antes y después del finetuning','']
    for size,p in paired.items():
        lines.append(f"- {size}: disminuye en {p['repetition_decreased']}/{p['examples']}, aumenta en {p['repetition_increased']}/{p['examples']}; cambio mediano emparejado {100*p['median_paired_delta']:+.2f} puntos porcentuales.")
    (output/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from asi.analysis.recovery_validation_report import LABELS
    names=list(results);x=list(range(len(names)))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    preparation=[sum(t['prepare_h2d_gib'] for t in results[v]['transitions']) for v in names]
    generation=[sum(t['generation_h2d_gib'] for t in results[v]['transitions']) for v in names]
    axes[0].bar(x,preparation,label='Prepare / category change',color='#38bdf8')
    axes[0].bar(x,generation,bottom=preparation,label='During generation',color='#64748b')
    for i,(a,b) in enumerate(zip(preparation,generation)):
        axes[0].text(i,a+b+8,f'{a+b:.1f}',ha='center',fontsize=9)
    axes[0].set_title('Total logical H2D transfers (GiB)');axes[0].legend(fontsize=8)
    axes[0].margins(y=.2)
    width=.35
    axes[1].bar([i-width/2 for i in x],[sum(c['strict_format'] for c in results[v]['code_diagnostics']) for v in names],width,label='Strict code-only format',color='#64748b')
    axes[1].bar([i+width/2 for i in x],[sum(c['first_block_syntax_pass'] is True for c in results[v]['code_diagnostics']) for v in names],width,label='First complete block parses',color='#38bdf8')
    axes[1].set_title('Code diagnostics (3 prompts; not correctness)');axes[1].set_ylim(0,4);axes[1].set_yticks(range(4));axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,[LABELS[v].replace(' ','\n') for v in names],fontsize=9)
    fig.suptitle('Recovery audit: include preparation; separate syntax from format')
    fig.text(.5,.01,'9 independent prompts, oracle categories, one pass, unequal EOS lengths. Transfers are cumulative, not model size.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(output/'audit.png',dpi=160);fig.savefig(output/'audit.pdf');plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.directory,a.output)


if __name__=='__main__':main()
