"""Conservative inventory of identifiable historical prompts, not decontamination."""
import argparse
import hashlib
import json
from pathlib import Path
from recovery_v2.preflight import save, digest


def records(value):
    if isinstance(value, dict):
        if isinstance(value.get('prompt'), str) and value['prompt'].strip():
            yield {k: value[k] for k in ('id', 'parent_id', 'prompt', 'source', 'source_id', 'source_split') if k in value}
        for child in value.values():
            yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / 'results/v2/F0/exposure_inventory'
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'report.json'
    if destination.exists():
        if args.resume:
            print('Existing inventory preserved; see report.json.')
            return
        raise SystemExit('Existing inventory preserved; use --resume to inspect it.')
    save(out / 'STATUS.json', {'status': 'partial', 'stage': 'inventory'})
    entries, sources, skipped = {}, {}, []
    for folder in ('results', 'configs'):
        for path in sorted((root / folder).rglob('*.json')):
            if root / 'results/v2' in path.parents:
                continue
            relative = path.relative_to(root).as_posix()
            if path.stat().st_size > 20_000_000:
                skipped.append({'path': relative, 'reason': 'over_20MB'})
                continue
            try:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError, OSError) as error:
                skipped.append({'path': relative, 'reason': type(error).__name__})
                continue
            found = list(records(value))
            if not found:
                continue
            sources[relative] = digest(path)
            for row in found:
                key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                entry = entries.setdefault(key, {'record': row, 'found_in': []})
                if relative not in entry['found_in']:
                    entry['found_in'].append(relative)
    save(out / 'identified_exposure.json', {'records': list(entries.values()), 'sources_sha256': sources})
    save(destination, {'status': 'partial', 'inventory_finished': True,
                       'identified_records': len(entries), 'source_files': len(sources),
                       'skipped': skipped, 'script_sha256': digest(Path(__file__)),
                       'output_sha256': digest(out / 'identified_exposure.json'),
                       'limitations': ['Conservative candidates, including prepared but possibly unexecuted prompts.',
                                       'Only JSON prompt fields; missing records, benchmark splits and JSONL need separate audit.',
                                       'Not burned.json yet; no claim of complete exposure coverage or clean holdout.']})
    save(out / 'STATUS.json', {'status': 'partial', 'stage': 'inventory_finished'})


if __name__ == '__main__':
    main()
