"""Conservative connected components that must never cross split boundaries."""
import json
import re
from collections import defaultdict, Counter
from recovery_v2.data.prepare_candidates import ROOT, normalize, grams, sha
from recovery_v2.preflight import save


def components(rows):
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        question = row['row'].get('question', row['row'].get('text', ''))
        if not question.strip(): raise ValueError('Missing question')
        normalized = normalize(question)
        keys = [('exact', normalized), ('numeric_template', re.sub(r'\d+(?:\s*\.\s*\d+)?', '<n>', normalized))]
        keys.extend(('13gram', gram) for gram in grams(question))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row['id'])
    return sorted((sorted(group) for group in groups.values()), key=lambda group:group[0])


def main():
    source = ROOT/'results/v2/F0/data_audit/candidates.json'
    out = ROOT/'results/v2/F0/duplicate_groups'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists(): raise SystemExit('Existing report preserved.')
    rows = json.loads(source.read_text(encoding='utf-8'))['examples']
    groups = components(rows)
    save(out/'groups.json', {'groups':groups})
    save(out/'report.json', {'status':'partial', 'candidates':len(rows), 'groups':len(groups),
                             'multi_example_groups':sum(len(g)>1 for g in groups),
                             'max_group_size':max(map(len,groups), default=0),
                             'size_histogram':dict(Counter(map(len,groups))),
                             'inputs_sha256':{str(source.relative_to(ROOT)):sha(source)},
                             'outputs_sha256':{'groups.json':sha(out/'groups.json')},
                             'limitations':['Prompt-only grouping: answer/support and semantic overlaps still need review.',
                                            'Groups are conservative constraints, not a validated train/dev/test split.']})


if __name__ == '__main__': main()
