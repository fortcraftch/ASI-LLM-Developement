"""Summarize exposure without declaring cached-only text to be evaluated."""
import argparse
import json
from collections import Counter
from recovery_v2.data.prepare_candidates import ROOT, sha
from recovery_v2.preflight import save


def classify(paths, mapping):
    if any(mapping.get(path) == 'recorded_evaluation_split' for path in paths):
        return 'recorded_evaluation_overlap'
    if paths:
        return 'cached_overlap_unresolved'
    return 'no_cached_overlap_detected'


def verify_scan(cache, candidate_path, exclusions_path):
    """Do not merge stale or modified artifacts into an exposure decision."""
    if not cache.get('scan_finished') or any(row['status'] != 'complete' for row in cache['files']):
        raise ValueError('Incomplete/failed cache scan')
    candidate_key = str(candidate_path.relative_to(ROOT))
    if cache.get('inputs_sha256', {}).get(candidate_key) != sha(candidate_path):
        raise ValueError('Candidate fingerprint differs')
    if cache.get('outputs_sha256', {}).get('exclusions.json') != sha(exclusions_path):
        raise ValueError('Exclusions fingerprint differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--identities', default='results/v2/F0/evaluation_sources_v2/report.json')
    args = parser.parse_args()
    base = ROOT/'results/v2/F0'
    out = base/'inventory_consolidated'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        if args.resume: return
        raise SystemExit('Existing report preserved.')
    paths = [base/'cache_overlap/report.json', base/'cache_overlap/exclusions.json',
             ROOT/args.identities, base/'data_audit/candidates.json']
    if any(not path.exists() for path in paths):
        raise SystemExit('Cache scan is not finished; no inventory closure produced.')
    cache, exclusions, identities, candidates = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    verify_scan(cache, paths[3], paths[1])
    mapping = {row['path']:row['status'] for row in identities['cache_mapping']}
    groups = Counter()
    records = []
    for row in candidates['examples']:
        matches = exclusions.get(row['id'], [])
        status = classify(matches, mapping)
        groups[(row['source'], status)] += 1
        records.append({'id':row['id'], 'source':row['source'], 'status':status,
                        'matching_files':matches})
    save(out/'candidate_review.json', {'records':records})
    save(out/'report.json', {'status':'partial', 'cache_scan_validated':True,
                             'counts':[{'source':source,'classification':status,'count':count}
                                       for (source,status),count in sorted(groups.items())],
                             'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
                             'outputs_sha256':{'candidate_review.json':sha(out/'candidate_review.json')},
                             'gate':'pending', 'training_authorized':False,
                             'remaining':['Resolve cached-only matches using source identity, not file presence.',
                                          'Audit semantic overlap and duplicate components before splitting.',
                                          'Validate task references and obtain sufficient unseen code examples.',
                                          'Inventory closure is not certification of independent test data.']})


if __name__ == '__main__': main()
