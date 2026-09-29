"""Pinned public recovery corpus, topic holdouts and resource estimates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCES = {
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866', 'main', 'MIT'),
    'mbpp': ('google-research-datasets/mbpp', '4bb6404fdc6cacfda99d4ac4205087b89d32030c', 'full', 'CC-BY-4.0'),
    'sciq': ('allenai/sciq', '2c94ad3e1aafab77146f384e23536f97a4849815', 'data', 'CC-BY-NC-3.0'),
}
CATEGORIES = {'gsm8k': 'mathematics_statistics', 'mbpp': 'computer_science_software_engineering',
              'sciq': 'life_sciences_biology'}
# First match wins. These are transparent lexical partitions, not semantic proofs.
HOLDOUTS = {
    'gsm8k': [('test', 'rates', r'\b(speed|miles per|kilometers per|mph)\b'),
              ('dev', 'geometry', r'\b(area|perimeter|rectangle|square feet|square meters)\b')],
    'mbpp': [('test', 'strings', r'\b(string|strings|substring|character|characters)\b'),
             ('dev', 'sorting', r'\b(sort|sorted|sorting)\b')],
    'sciq': [('test', 'ecology', r'\b(ecosystem|ecosystems|food web|food chain|biodiversity|population|populations)\b'),
             ('dev', 'genetics', r'\b(dna|rna|gene|genes|genetic|chromosome|chromosomes|heredity)\b')],
}
BIOLOGY = r'\b(cell|cells|organism|organisms|photosynthesis|protein|proteins|enzyme|enzymes|plant|plants|animal|animals|bacteria|fungi|evolution|respiration|blood|neuron|neurons|reproduction|species|ecosystem|dna|rna|genetic|chromosome)\b'


def signature(text):
    return re.sub(r'\s+', ' ', re.sub(r'\d+(?:\.\d+)?', '<n>', text.casefold())).strip()


def convert(source, row, origin, index):
    question = row.get('question', row.get('text', '')).strip()
    support = row.get('support', '').strip()
    if source == 'sciq' and (not support or not re.search(BIOLOGY, question+' '+support, re.I)):
        return None
    split, topic = 'train', 'remaining_topics'
    for candidate, label, pattern in HOLDOUTS[source]:
        if re.search(pattern, question+' '+(support if source == 'sciq' else ''), re.I):
            split, topic = candidate, label
            break
    result = {'id': f'{source}_{origin}_{index}', 'split': split, 'topic': source+'_'+topic,
              'category': CATEGORIES[source], 'source': source, 'source_split': origin,
              'source_id': str(row.get('task_id', index)), 'template_group': signature(question)}
    if source == 'gsm8k':
        answer = re.sub(r'<<.*?>>', '', row['answer'])
        if '####' not in answer: return None
        explanation, number = answer.rsplit('####', 1)
        result.update(prompt=question+'\nExplain your calculation and end with "Final answer: <number>".',
                      answer=explanation.strip()+'\nFinal answer: '+number.strip(),
                      expected_number=number.strip(), evaluation='final_number')
    elif source == 'mbpp':
        result.update(prompt=question+'\nReturn only Python code in one fenced code block.',
                      answer='```python\n'+row['code'].strip()+'\n```',
                      reference_tests=row.get('test_list', []), evaluation='python_syntax_and_manual_tests')
    else:
        result.update(prompt=question+'\nGive a short answer, then explain why.',
                      answer=row['correct_answer'].strip()+'. '+support,
                      expected_phrase=row['correct_answer'].strip(), evaluation='answer_phrase_and_manual')
    return result


def build(raw, tokenizer, max_length=384, train_cap=1500, eval_cap=60):
    from asi.experiments.public_moe import input_ids
    candidates, rejected = [], Counter()
    for source, origin, index, row in raw:
        converted = convert(source, row, origin, index)
        if converted is None:
            rejected['not_eligible'] += 1
            continue
        try:
            prompt = input_ids(tokenizer, converted['prompt'],limit=max_length)[0].tolist()
        except ValueError:
            rejected['length'] += 1
            continue
        answer = tokenizer.encode(converted['answer'], add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prompt)+len(answer)>max_length:
            rejected['length'] += 1
            continue
        converted.update(token_count=len(prompt)+len(answer), answer_tokens=len(answer))
        candidates.append(converted)
    # Reject every member of a scaffold duplicated across partitions, not just one side.
    memberships = {}
    for row in candidates:
        memberships.setdefault(row['template_group'], set()).add(row['split'])
    seen, counts, rows = set(), Counter(), []
    for row in sorted(candidates, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest()):
        group = row['template_group']
        if len(memberships[group])>1 or group in seen:
            rejected['duplicate_or_cross_split_template'] += 1
            continue
        seen.add(group)
        key = (row['category'], row['split'])
        if counts[key] >= (train_cap if row['split']=='train' else eval_cap):
            rejected['cap'] += 1
            continue
        counts[key] += 1
        rows.append(row)
    for category in CATEGORIES.values():
        for split in ('train', 'dev', 'test'):
            if not counts[(category,split)]:
                raise ValueError(f'Empty category/split: {category}/{split}')
    return {'schema':3, 'description':'Public topic-held-out recovery corpus; not official benchmark splits',
            'sources':{name:dict(repository=v[0],revision=v[1],license=v[3],url='https://huggingface.co/datasets/'+v[0]) for name,v in SOURCES.items()},
            'split_policy':HOLDOUTS, 'rejected':dict(rejected),
            'counts':{f'{c}/{s}':n for (c,s),n in counts.items()},
            'limitations':['Lexical topics and number-normalized templates do not prove semantic independence.',
                'Public datasets may have appeared in model pretraining.',
                'SciQ biology filter is heuristic; manually audit examples. SciQ license is noncommercial.',
                'Original source splits are regrouped; these are not official benchmark scores.'],
            'examples':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,default=Path('data/recovery_sources_v3'))
    p.add_argument('--model-dir',type=Path,default=Path('public_models/olmoe'))
    p.add_argument('--download',action='store_true')
    p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--train-per-category',type=int,default=1500)
    p.add_argument('--eval-per-category',type=int,default=60)
    args=p.parse_args()
    if args.output.exists() or min(args.max_length,args.train_per_category,args.eval_per_category)<1:
        p.error('Use a fresh output and positive limits')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    raw=[]; files={}
    for source,(repo,revision,config,license_name) in SOURCES.items():
        directory=args.source_dir/source
        if args.download:
            snapshot_download(repo,repo_type='dataset',revision=revision,local_dir=directory,
                              allow_patterns=[config+'/*.parquet','README.md','.gitattributes'])
            (directory/'revision.json').write_text(json.dumps({'repository':repo,'revision':revision}))
        lock=json.loads((directory/'revision.json').read_text())
        if lock!={'repository':repo,'revision':revision}: raise ValueError('Source revision differs')
        paths=sorted((directory/config).glob('*.parquet'))
        if not paths: raise ValueError('No source parquet files: '+source)
        for path in paths:
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            origin=path.stem.split('-')[0]
            for i,row in enumerate(pq.read_table(path).to_pylist()): raw.append((source,origin,i,row))
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    corpus=build(raw,tokenizer,args.max_length,args.train_per_category,args.eval_per_category)
    corpus['source_file_sha256']=files
    corpus['teacher_dense_fp32_bytes']=sum(r['answer_tokens'] for r in corpus['examples'])*len(tokenizer)*4
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(corpus,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in corpus.items() if k not in ('examples','source_file_sha256')},indent=2))
