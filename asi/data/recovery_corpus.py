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


def reviewed_draft(corpus_path, review_path, replacement_paths):
    """Apply explicitly hashed proposals to a new train-only draft, never promote it."""
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def text_sha(value): return hashlib.sha256(value.encode()).hexdigest()
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if sha(corpus_path) not in review['sources_sha256'].values():
        raise ValueError('Corpus is not pinned by the reference review')
    source = {r['id']: r for r in corpus['examples']}
    if len(source) != len(corpus['examples']): raise ValueError('Duplicate source IDs')
    records = {r['id']: r for r in review['records']}
    if len(records) != len(review['records']): raise ValueError('Duplicate review IDs')
    changes = {}
    for path in replacement_paths:
        for replacement in json.loads(path.read_text(encoding='utf-8'))['replacements']:
            parent = replacement.get('parent_id', replacement.get('id'))
            if parent not in records or parent in changes:
                raise ValueError('Unknown or multiply replaced parent: '+str(parent))
            row = source[parent]
            if replacement['old_answer_sha256'] != text_sha(row['answer']):
                raise ValueError('Old answer hash differs: '+parent)
            if replacement.get('old_prompt_sha256', text_sha(row['prompt'])) != text_sha(row['prompt']):
                raise ValueError('Old prompt hash differs: '+parent)
            answer = replacement['new_answer']
            prompt = replacement.get('new_prompt', row['prompt'])
            if not answer.strip() or not prompt.strip(): raise ValueError('Empty replacement')
            for field, value in [('answer',answer), ('prompt',prompt)]:
                if replacement.get('new_'+field+'_sha256', text_sha(value)) != text_sha(value):
                    raise ValueError('New text hash differs: '+parent)
            if replacement.get('allowed_split', 'train') != 'train':
                raise ValueError('Reviewed historical training examples must remain train-only')
            targets = {}
            for field in ('reference_tests', 'expected_number', 'expected_phrase'):
                key = 'new_' + field
                if key in replacement:
                    value = replacement[key]
                    valid = (isinstance(value, list) and bool(value) and
                             all(isinstance(t, str) and t.strip() for t in value)) if field == 'reference_tests' else (
                                 isinstance(value, str) and bool(value.strip()))
                    if not valid: raise ValueError('Invalid replacement evaluation target: '+key)
                    targets[field] = value
            changes[parent] = (prompt, answer, str(path), targets)
    rows = []
    for parent, record in records.items():
        row = source[parent]
        if hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest() != record['row_sha256']:
            raise ValueError('Reviewed row differs: '+parent)
        if row['split'] != 'train': raise ValueError('This draft accepts reviewed training rows only')
        revised = dict(row, parent_id=parent, train_only=True, review_status=record['status'])
        if parent in changes:
            prompt, answer, path, targets = changes[parent]
            revised.update(id=parent+'_review_v1', prompt=prompt, answer=answer,
                           revision_source=path, review_status='replacement_pending_validation',
                           teacher_recompute_required=True)
            # These values belong to old text. Never carry them across tokenization changes.
            for field in ('token_count','answer_tokens','tokens','prompt_length'):
                revised.pop(field, None)
            # Old tests/answers may contradict the revised task. Retain them only
            # as provenance unless the proposal explicitly supplies a new target.
            for field in ('reference_tests', 'expected_number', 'expected_phrase'):
                if field in revised:
                    revised['parent_' + field] = revised.pop(field)
            revised.update(targets)
            revised['evaluation_targets_require_review'] = True
            revised['template_group'] = signature(prompt)
            revised['parent_template_group'] = row.get('template_group')
        rows.append(revised)
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Invalid draft IDs')
    paths = [corpus_path, review_path, *replacement_paths]
    return dict(schema=4, ready_for_training=False, status='reviewed_training_draft',
                description='Historical training references and proposed corrections; no new dev/test.',
                sources=corpus.get('sources', {}),
                source_file_sha256={str(p):sha(p) for p in paths},
                replacements_applied=len(changes), examples=rows,
                limitations=['Corrections retain parent exposure; never treat them as clean holdout data.',
                             'Functional tests, factual review, tokenization and split freezing remain separate gates.',
                             'Original corpus and teacher targets are unchanged.'])


def revision_main():
    parser = argparse.ArgumentParser(description='Build a hash-checked train-only draft from reference-review proposals')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--replacements', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output; never overwrite a versioned draft')
    result = reviewed_draft(args.corpus, args.review, args.replacements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','ready_for_training','replacements_applied')}))
