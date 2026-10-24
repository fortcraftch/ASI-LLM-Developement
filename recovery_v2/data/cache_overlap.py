"""Exclude any candidate overlapping locally cached benchmark text."""
from collections import defaultdict, Counter
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from recovery_v2.data.prepare_candidates import ROOT, grams, normalize, sha, texts, read_documents
from recovery_v2.preflight import save
from recovery_v2.data.scan_checkpoint import ScanCheckpoint


def strings(value):
    if isinstance(value, str): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from strings(child)
    elif isinstance(value, list):
        for child in value: yield from strings(child)


def batches(path):
    if path.suffix == '.parquet':
        yield from pq.ParquetFile(path).iter_batches(batch_size=256)
    else:
        with pa.memory_map(str(path), 'r') as stream:
            try:
                reader = pa.ipc.open_stream(stream)
            except pa.ArrowInvalid:
                stream.seek(0)
                reader = pa.ipc.open_file(stream)
                for i in range(reader.num_record_batches): yield reader.get_batch(i)
            else:
                yield from reader


def documents(path):
    if path.suffix in ('.arrow', '.parquet'):
        for batch in batches(path): yield from batch.to_pylist()
    else:
        yield from read_documents(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default='results/v2/F0/cache_overlap')
    parser.add_argument('--source', default='results/v2/F0/data_audit/candidates.json')
    parser.add_argument('--include-history', action='store_true')
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if ROOT/'results/v2/F0' not in out.parents:
        raise SystemExit('Output must remain below results/v2/F0.')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        if args.resume: return
        raise SystemExit('Existing output preserved; use --resume.')
    source = (ROOT/args.source).resolve()
    if ROOT/'data/v2/test' == source or ROOT/'data/v2/test' in source.parents:
        raise SystemExit('Sealed test access forbidden.')
    source_hash = sha(source)
    checkpoint = ScanCheckpoint(out/'checkpoints', {'candidate_sha256':source_hash,
                                                   'algorithm':'normalized_exact_and_13grams_v1',
                                                   'include_history':args.include_history})
    rows = json.loads(source.read_text(encoding='utf-8'))['examples']
    exact, index = defaultdict(set), defaultdict(set)
    for row in rows:
        for text in texts(row['row']):
            if not text.strip(): continue
            exact[normalize(text)].add(row['id'])
            for gram in grams(text): index[gram].add(row['id'])
    excluded, ledger = defaultdict(set), []
    files = sorted(p for folder in ('results/hf_datasets_cache', 'results/hf_hub_cache')
                   for p in (ROOT/folder).rglob('*') if p.suffix in ('.arrow', '.parquet'))
    if args.include_history:
        history = json.loads((ROOT/'results/v2/F0/data_audit/files.json').read_text(encoding='utf-8'))
        files.extend(ROOT/row['path'] for row in history['files'] if row['handling'] == 'indexed')
        files.extend((ROOT/'data/recovery_sources_v3').rglob('*.parquet'))
        files = sorted(set(files))
    for number, path in enumerate(files):
        relative = path.relative_to(ROOT).as_posix()
        save(out/'STATUS.json', {'status':'partial', 'stage':'cache_scan', 'file':number, 'total':len(files)})
        item = {'path':relative, 'sha256':sha(path), 'rows':0}
        cached = checkpoint.read(relative, item['sha256']) if args.resume else None
        if cached is not None:
            ledger.append(cached['item'])
            for key in cached['excluded_ids']: excluded[key].add(relative)
            continue
        affected = set()
        try:
            for row in documents(path):
                    item['rows'] += 1
                    for text in strings(row):
                        matches = set(exact.get(normalize(text), ()))
                        for gram in grams(text): matches.update(index.get(gram, ()))
                        for key in matches:
                            excluded[key].add(relative)
                            affected.add(key)
            item['status'] = 'complete'
        except Exception as error:
            item.update(status='failed', error=type(error).__name__)
        ledger.append(item)
        if item['status'] == 'complete':
            checkpoint.write(relative, item['sha256'], {'status':'complete','item':item,
                                                       'excluded_ids':sorted(affected)})
    remaining = [r for r in rows if r['id'] not in excluded]
    save(out/'candidates.json', {'status':'pending_semantic_and_split_review', 'examples':remaining})
    save(out/'exclusions.json', {key:sorted(value) for key,value in excluded.items()})
    save(out/'report.json', {'status':'partial', 'scan_finished':True,
                             'inputs_sha256':{str(source.relative_to(ROOT)):sha(source)},
                             'files':ledger, 'excluded_candidates':len(excluded),
                             'remaining':dict(Counter(r['source'] for r in remaining)),
                             'outputs_sha256':{name:sha(out/name) for name in ('candidates.json','exclusions.json')},
                             'limitations':['Cache presence is not evidence of evaluation: exclusions are conservative.',
                                            'No independent splits created; semantic review, reference validation and quotas pending.']})
    save(out/'STATUS.json', {'status':'partial', 'stage':'cache_scan_finished'})


if __name__ == '__main__': main()
