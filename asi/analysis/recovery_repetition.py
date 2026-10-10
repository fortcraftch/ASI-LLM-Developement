"""Offline audit of repetition; does not change decoding or rerun a model."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from asi.analysis.recovery_generation_report import read_reports


def longest_loop(sequence, max_period=8):
    """Longest adjacent exact repeated block, requiring at least three copies.

    Positions are zero-based and end-exclusive. This diagnoses surface loops,
    not semantic quality. Legitimate repeated punctuation can also be detected.
    """
    best = {'start': None, 'end': None, 'period': 0, 'copies': 0, 'span': 0, 'block': []}
    for start in range(len(sequence)):
        for period in range(1, min(max_period, (len(sequence)-start)//3)+1):
            block = sequence[start:start+period]
            end = start+period
            while sequence[end:end+period] == block:
                end += period
            copies = (end-start)//period
            if copies >= 3 and end-start > best['span']:
                best = {'start': start, 'end': end, 'period': period, 'copies': copies,
                        'span': end-start, 'block': list(block)}
    return best


def words(text):
    return re.findall(r"\w+(?:['’]\w+)*", text.lower())


def audit(tokens, text):
    token_loop = longest_loop(tokens)
    word_loop = longest_loop(words(text))
    triples = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    checkpoints = []
    for size in (32,64,128,192,256):
        if len(tokens) >= size:
            grams=triples[:size-2]
            checkpoints.append({'tokens':size, 'repeated_trigram_fraction':1-len(set(grams))/len(grams)})
    ws=words(text)
    repeated_phrases=Counter(tuple(ws[i:i+5]) for i in range(len(ws)-4))
    return {'token_loop':token_loop, 'word_loop':word_loop, 'prefixes':checkpoints,
            'repeated_five_word_phrases':[{'phrase':' '.join(k),'count':v}
                for k,v in repeated_phrases.most_common(5) if v >= 3]}


def analyze(directory, previous, corpus_path, subset_path):
    reports=read_reports(directory)
    historical=json.loads(previous.read_text(encoding='utf8'))
    if not historical['completed'] or historical['identity']['source']!=reports[1]['source_identity']:
        raise ValueError('Historical baseline is not from this model')
    before={r['id']:r for r in historical['generations'] if r['stage']=='before' and r['mask']=='32'}
    after={r['id']:r for r in historical['generations'] if r['stage']=='after' and r['mask']=='32'}
    # Reproduce the historic adapted outputs exactly before using its before/after
    # contrast. Do not merge historic timing/memory with fresh-process measures.
    if any(after[r['id']]['token_ids']!=r['token_ids'] for r in reports[1]['records']):
        raise ValueError('Historic adapted tokens differ; inspect protocol before comparing')
    corpus=json.loads(corpus_path.read_text(encoding='utf8'))
    subset=set(json.loads(subset_path.read_text(encoding='utf8')))
    training=[r for r in corpus['examples'] if r['id'] in subset and r['split']=='train']
    references=[{'id':r['id'], 'loop':longest_loop(words(r['answer']))} for r in training]
    records=[]
    for report in reports:
        for row in report['records']:
            records.append({'variant':report['variant'], 'id':row['id'], 'tokens':len(row['token_ids']),
                'stop_reason':row['stop_reason'], 'repeated_trigram_fraction':row['metrics']['repeated_trigram_fraction'],
                'generation_loads':row['generation_transfers']['loads'], **audit(row['token_ids'],row['text'])})
    for row in before.values():
        tokens=row['token_ids']
        grams=[tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        records.append({'variant':'32_before_historical', 'id':row['id'], 'tokens':len(tokens),
            'repeated_trigram_fraction':1-len(set(grams))/len(grams), **audit(tokens,row['text'])})
    sources=[directory/v/'report.json' for v in ('original','32','16')]+[previous,corpus_path,subset_path]
    result={'completed':True,'kind':'offline analysis, no new inference',
        'historical_after_tokens_exact_match':True,'records':records,
        'training_reference_audit':{'count':len(training),'longest_adjacent_word_loops':
                                   sorted(references,key=lambda r:r['loop']['span'],reverse=True)[:5]},
        'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'limitations':['Three prompts; no causal decomposition of decoding, expert choice and training.',
                       'Exact repetition is not semantic correctness; token lengths differ.',
                       'Historical before/after is used for text only, not current hardware timing.',
                       'A repeated-word audit does not establish training-data quality or teacher calibration.']}
    (directory/'repetition_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,4.8))
    for variant,color,label in [('original','#64748b','Original'),('32_before_historical','#d97706','32 sin ajuste (previo)'),
                                ('32','#2563eb','32 recuperado'),('16','#db2777','16 recuperado')]:
        for ax,ident in zip(axes,('gsm8k_train_4607','sciq_test_772')):
            row=next(r for r in records if r['variant']==variant and r['id']==ident)
            ax.plot([r['tokens'] for r in row['prefixes']],
                    [100*r['repeated_trigram_fraction'] for r in row['prefixes']],
                    marker='o',label=label,color=color)
    for ax,title in zip(axes,('Matemáticas','Biología')):
        ax.set_title(title)
        ax.set_xlabel('Primeros N tokens de la respuesta')
        ax.set_ylabel('Trigramas de tokens repetidos (%)')
        ax.set_ylim(0,100)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Cómo se acumula la repetición: comparación a igual longitud de prefijo')
    fig.tight_layout()
    fig.savefig(directory/'repetition_prefixes.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'completed':True,'historical_exact_match':True,'training_references':len(training),
                      'output':str(directory/'repetition_audit.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('results/generation_recovery_20261007'))
    parser.add_argument('--previous',type=Path,default=Path('results/recovery_quick_20261006/joint32_generation256_retry/report.json'))
    parser.add_argument('--corpus',type=Path,default=Path('results/recovery_corpus_v3/corpus.json'))
    parser.add_argument('--subset',type=Path,default=Path('results/recovery_quick_20261006/ids.json'))
    args=parser.parse_args()
    analyze(args.directory,args.previous,args.corpus,args.subset)
