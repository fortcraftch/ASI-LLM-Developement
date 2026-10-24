"""Conservative local exposure audit and candidate staging; no sealed test access."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import yaml
from recovery_v2.preflight import save

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {'prompt', 'question', 'text', 'input', 'instruction', 'answer', 'response',
          'output', 'content', 'support', 'code', 'reference', 'correct_answer'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return ' '.join(re.findall(r'\w+|[^\w\s]', text.casefold()))


def grams(text):
    words = normalize(text).split()
    return {hashlib.sha256(' '.join(words[i:i+13]).encode()).digest() for i in range(len(words)-12)}


def texts(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FIELDS and isinstance(child, str):
                yield child
            elif isinstance(child, (dict, list)):
                yield from texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from texts(child)


def read_documents(path):
    ext = path.suffix.lower()
    if ext == '.json':
        yield json.loads(path.read_text(encoding='utf-8-sig'))
    elif ext == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip(): yield json.loads(line)
    elif ext in ('.yaml', '.yml'):
        yield yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    elif ext == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            yield from csv.DictReader(stream)
    elif ext == '.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        yield {'text': path.read_text(encoding='utf-8', errors='replace')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results/v2/F0/data_audit'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        if args.resume: return
        raise SystemExit('Existing run preserved; use --resume.')
    save(out/'STATUS.json', {'status':'partial', 'stage':'historical_scan'})
    exact, ngrams, ledger, errors = set(), set(), [], []
    def ingest(text):
        if text.strip():
            exact.add(hashlib.sha256(normalize(text).encode()).hexdigest())
            ngrams.update(grams(text))
    roots = ('results', 'configs', 'data')
    supported = {'.json', '.jsonl', '.csv', '.yaml', '.yml', '.log', '.txt', '.md', '.parquet', '.trace'}
    for directory in roots:
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file(): continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(('results/v2/', 'data/v2/')): continue
            # These are downloaded source caches or third-party code, not execution evidence.
            if any(part in rel for part in ('hf_hub_cache/', 'hf_datasets_cache/', 'analysis_dependencies/',
                                            'recovery_sources_v3/', '__pycache__/', '.cache/')):
                continue
            if '.no_exist/' in rel: continue
            if path.suffix.lower() not in supported:
                ledger.append({'path':rel, 'bytes':path.stat().st_size,
                               'handling':'nontext_artifact_not_independent_source'})
                continue
            item = {'path':rel, 'sha256':sha(path), 'bytes':path.stat().st_size, 'texts':0}
            try:
                for document in read_documents(path):
                    for text in texts(document):
                        ingest(text); item['texts'] += 1
                item['handling'] = 'indexed'
            except Exception as error:
                item['handling'] = 'quarantined'
                item['error'] = type(error).__name__
                errors.append(rel)
            ledger.append(item)
    # Corrupt teacher metadata / token-only shards: exclude the entire old source pool
    # used by recovery, rather than attempting unsafe pickle loads or guessing contents.
    old = ROOT/'results/recovery_corpus_v3/corpus.json'
    for text in texts(json.loads(old.read_text(encoding='utf-8-sig'))): ingest(text)
    save(out/'STATUS.json', {'status':'partial', 'stage':'source_filtering'})
    candidates, rejected, sources = [], Counter(), []
    source_root = ROOT/'data/recovery_sources_v3'
    # All test and demonstration splits are excluded, including rows without text in reports.
    paths = sorted(source_root.rglob('*.parquet'))
    for path in paths:
        if path.name.startswith(('test-', 'prompt-')):
            for row in read_documents(path):
                for text in texts(row): ingest(text)
    for path in paths:
        source = path.relative_to(source_root).parts[0]
        split = path.name.split('-')[0]
        sources.append({'path':path.relative_to(ROOT).as_posix(), 'sha256':sha(path),
                        'source':source, 'split':split, 'rows':pq.ParquetFile(path).metadata.num_rows})
        if split in ('test', 'prompt'):
            rejected['excluded_source_split'] += pq.ParquetFile(path).metadata.num_rows
            continue
        for index, row in enumerate(read_documents(path)):
            content = list(texts(row))
            if any(hashlib.sha256(normalize(t).encode()).hexdigest() in exact for t in content if t.strip()):
                rejected['exact_exposure'] += 1; continue
            if any(grams(t) & ngrams for t in content):
                rejected['13gram_exposure'] += 1; continue
            candidates.append({'id':f'{source}_{split}_{index}', 'source':source, 'source_split':split,
                               'source_id':str(row.get('task_id', index)), 'row':row})
    # Candidates are deliberately not assigned train/dev/test until semantic review and quotas.
    save(out/'candidates.json', {'status':'quarantined_pending_review', 'examples':candidates})
    save(out/'files.json', {'files':ledger, 'sources':sources})
    burned = {'status':'conservative_local_inventory', 'normalization':'casefold unicode words and punctuation',
              'exact_sha256':sorted(exact), 'ngram_size':13,
              'blocked_source_splits':['gsm8k/test', 'mbpp/test', 'mbpp/prompt', 'sciq/test'],
              'legacy_corpus_quarantined':str(old.relative_to(ROOT)),
              'ledger_sha256':sha(out/'files.json')}
    save(out/'burned.json', burned)
    save(out/'report.json', {'status':'partial', 'inventory_finished':True, 'files':len(ledger),
                             'quarantined_files':errors, 'exact_texts':len(exact), 'ngrams':len(ngrams),
                             'candidate_counts':dict(Counter(r['source'] for r in candidates)),
                             'rejected':dict(rejected), 'inputs':sources,
                             'script_sha256':sha(Path(__file__)),
                             'outputs_sha256':{p.name:sha(p) for p in [out/'candidates.json', out/'files.json', out/'burned.json']},
                             'limitations':['Candidate preparation only: no independent evaluation split is certified.',
                                            'Token-only artifacts are covered by conservative corpus quarantine, not decoded.',
                                            'Exact and 13-gram matching cannot rule out semantic overlap or pretraining exposure.',
                                            'All historical local test splits excluded; new eval sources still required if quotas fail.']})
    save(out/'STATUS.json', {'status':'partial', 'stage':'candidates_prepared'})


if __name__ == '__main__': main()
