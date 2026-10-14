"""Conservative corpus exposure inventory; never treats candidates as a frozen test."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def mentions(value, known_ids, prompts):
    found = set()
    if isinstance(value, str):
        if value in known_ids:
            found.add(value)
        found.update(prompts.get(normalize(value), []))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in known_ids:
                found.add(key)
            found.update(mentions(item, known_ids, prompts))
    elif isinstance(value, list):
        for item in value:
            found.update(mentions(item, known_ids, prompts))
    return found


def inventory(examples, exposure, quarantined):
    by_id = {x['id']: x for x in examples}
    if len(by_id) != len(examples):
        raise ValueError('Duplicate corpus IDs')
    groups = defaultdict(set)
    for row in examples:
        groups[('prompt', normalize(row['prompt']))].add(row['id'])
        if row.get('template_group'):
            groups[('template', normalize(row['template_group']))].add(row['id'])
    cross_split = [{'kind': kind, 'ids': sorted(ids)} for (kind, _), ids in groups.items()
                   if len({by_id[i]['split'] for i in ids}) > 1]
    cross_split_ids = {key for group in cross_split for key in group['ids']}
    excluded = set(exposure) | set(quarantined) | cross_split_ids
    # Transitive closure handles templates connecting different prompt duplicates.
    changed = True
    while changed:
        changed = False
        for ids in groups.values():
            if ids & excluded and ids - excluded:
                excluded.update(ids)
                changed = True
    records = []
    for row in examples:
        key = row['id']
        reasons = []
        if key in exposure: reasons.append('historical_reference')
        if key in quarantined: reasons.append('quarantined_reference')
        if key in cross_split_ids: reasons.append('cross_split_prompt_or_template')
        if key in excluded and not reasons: reasons.append('duplicate_or_template_of_excluded')
        records.append({'id': key, 'split': row['split'], 'category': row['category'],
                        'topic': row.get('topic'), 'reasons': reasons,
                        'historical_sources': sorted(exposure.get(key, [])),
                        'candidate_only': not reasons})
    return records, cross_split


def build(corpus_path, results, output, quarantine_path):
    if output.exists():
        raise ValueError('Use a fresh audit output directory; historical reports may be hash-pinned')
    raw = corpus_path.read_bytes()
    examples = json.loads(raw)['examples']
    quarantine_raw = quarantine_path.read_bytes()
    quarantine = json.loads(quarantine_raw)['excluded']
    known = {x['id'] for x in examples}
    if set(quarantine) - known:
        raise ValueError('Quarantine contains unknown corpus IDs')
    prompts = defaultdict(set)
    for x in examples:
        prompts[normalize(x['prompt'])].add(x['id'])
    exposure = defaultdict(set)
    hashes, errors, declared_absent = {}, [], {}
    prefixes = ('recovery_', 'routing_', 'mask_', 'generation_recovery')
    files = []
    for directory in sorted(results.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefixes): continue
        if directory.name.startswith(('recovery_corpus', 'recovery_data_audit')): continue
        if directory.resolve() == output.resolve(): continue
        files.extend(sorted(directory.rglob('*.json')))
    for path in files:
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        try:
            data = json.loads(content)
        except (ValueError, UnicodeError) as error:
            errors.append({'path': str(path), 'error': str(error)})
            continue
        # An audit's missing list is evidence of absence, not a completed run.
        # Keep it separately instead of treating every corpus ID as exposed.
        if isinstance(data, dict) and 'manifest_valid_json' in data and 'missing_examples' in data:
            declared_absent[str(path)] = sorted(mentions(data['missing_examples'], known, {}))
            data = {k: v for k, v in data.items() if k != 'missing_examples'}
        for key in mentions(data, known, prompts):
            exposure[key].add(str(path))
    records, duplicates = inventory(examples, exposure, quarantine)
    counts = Counter((r['split'], r['category'], r['candidate_only']) for r in records)
    report = {'schema': 1, 'audit_complete': not errors, 'test_frozen': False,
              'corpus_sha256': hashlib.sha256(raw).hexdigest(),
              'quarantine_sha256': hashlib.sha256(quarantine_raw).hexdigest(),
              'sources_sha256': hashes, 'errors': errors, 'declared_absent': declared_absent,
              'cross_split_groups': duplicates,
              'counts': [{'split': s, 'category': c, 'candidate_only': candidate, 'count': n}
                         for (s, c, candidate), n in sorted(counts.items())],
              'records': records, 'limitations': [
                  'Historical reference is conservative: planned or prepared IDs may not have been executed.',
                  'Only JSON artifacts in named recovery/routing/mask directories are scanned; no PT or external training logs.',
                  'Candidate IDs require source/reference review and topic separation before freezing new dev/test.',
                  'Exact normalized prompts and declared template groups do not detect all semantic duplicates.',
                  'This inventory cannot establish absence of original model pretraining contamination.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Recovery data exposure audit', '',
             f'Scanned {len(hashes)} JSON artifacts. Corpus: {len(records)} examples. Test frozen: no.', '',
             '| Split | Category | Referenced/excluded | Candidate only |', '|---|---|---:|---:|']
    for s, c in sorted({(r['split'], r['category']) for r in records}):
        lines.append(f'| {s} | {c} | {counts[s,c,False]} | {counts[s,c,True]} |')
    lines += ['', '## Limitations', ''] + ['- '+x for x in report['limitations']]
    if errors: lines += ['', 'ERROR: malformed artifacts; audit is incomplete.']
    (output / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=Path('results/recovery_corpus_v3/corpus.json'))
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--output', type=Path, required=True, help='New directory; existing audits are never overwritten')
    p.add_argument('--quarantine', type=Path, default=Path('configs/recovery_reference_quarantine.json'))
    a = p.parse_args()
    report = build(a.corpus, a.results, a.output, a.quarantine)
    print((a.output / 'README.md').read_text(encoding='utf-8'))
    if not report['audit_complete']: raise SystemExit(1)


if __name__ == '__main__': main()
