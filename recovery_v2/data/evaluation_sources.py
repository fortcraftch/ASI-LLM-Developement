"""Resolve benchmark cache provenance from recorded historical dataset identities."""
import ast
import json
from collections import defaultdict
from recovery_v2.data.prepare_candidates import ROOT, sha
from recovery_v2.preflight import save


def dataset_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'datasets' and isinstance(child, dict):
                for name in child:
                    try: identity = ast.literal_eval(name)
                    except (ValueError, SyntaxError): continue
                    if isinstance(identity, tuple) and len(identity) == 3 and isinstance(identity[0], str):
                        yield identity
            if isinstance(child, (dict, list)): yield from dataset_keys(child)
    elif isinstance(value, list):
        for child in value: yield from dataset_keys(child)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/v2/F0/evaluation_sources')
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if ROOT/'results/v2/F0' not in out.parents:
        raise SystemExit('Output must remain below results/v2/F0.')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists(): raise SystemExit('Existing evidence preserved.')
    sources, inputs = defaultdict(list), {}
    for path in sorted((ROOT/'results').rglob('*.json')):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith('results/v2/') or '/hf_' in relative or '/analysis_dependencies/' in relative: continue
        try: value = json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError): continue
        keys = set(dataset_keys(value))
        if keys: inputs[relative] = sha(path)
        for key in keys: sources[key].append(relative)
    mappings = []
    for path in sorted((ROOT/'results/hf_datasets_cache').rglob('*.arrow')):
        parts = path.relative_to(ROOT/'results/hf_datasets_cache').parts
        split = path.stem.rsplit('-', 1)[-1]
        matches = [key for key in sources if key[0].replace('/', '___').casefold() == parts[0].casefold()
                   and key[2] == split and (key[1] is None or key[1] == parts[1])]
        mappings.append({'path':path.relative_to(ROOT).as_posix(),
                         'status':'recorded_evaluation_split' if matches else 'no_split_evidence_in_manifests',
                         'identities':[list(key) for key in matches]})
    for path in sorted((ROOT/'results/hf_hub_cache').rglob('*.parquet')):
        parts = path.relative_to(ROOT/'results/hf_hub_cache').parts
        repository = parts[0].removeprefix('datasets--').replace('--', '/')
        split = path.name.split('-')[0]
        config = parts[-2]
        matches = [key for key in sources if key[0].casefold() == repository.casefold()
                   and key[2] == split and (key[1] is None or key[1] == config)]
        mappings.append({'path':path.relative_to(ROOT).as_posix(),
                         'status':'recorded_evaluation_split' if matches else 'no_split_evidence_in_manifests',
                         'identities':[list(key) for key in matches]})
    save(out/'report.json', {'status':'partial', 'sources':[{'identity':list(key), 'reports':paths}
                                                         for key,paths in sorted(sources.items(), key=lambda x:str(x[0]))],
                             'inputs_sha256':inputs, 'cache_mapping':mappings,
                             'limitation':'Dataset manifest presence conservatively blocks the entire split; absent evidence is not proof of non-use.'})


if __name__ == '__main__': main()
