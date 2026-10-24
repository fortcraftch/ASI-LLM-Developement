"""Retrieve historical prompt neighbors for review; never certify independence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def neighbors(queries, historical, k=5):
    if k < 1 or not queries or not historical:
        raise ValueError('Need nonempty inputs and positive k')
    def terms(text):
        words = re.findall(r'[a-z]+', text.lower())
        return Counter(words + [' '.join(pair) for pair in zip(words, words[1:])])
    bags = [terms(r['prompt']) for r in queries + historical]
    frequencies = Counter(term for bag in bags for term in bag)
    vectors = []
    for bag in bags:
        values = {term: (1 + math.log(count)) *
                  (1 + math.log((1 + len(bags)) / (1 + frequencies[term])))
                  for term, count in bag.items()}
        norm = math.sqrt(sum(value ** 2 for value in values.values())) or 1
        vectors.append({term: value / norm for term, value in values.items()})
    result = []
    for i, query in enumerate(queries):
        scored = [(sum(value * other.get(term, 0) for term, value in vectors[i].items()), j)
                  for j, other in enumerate(vectors[len(queries):])]
        ranked = sorted(scored, key=lambda pair: (-pair[0], historical[pair[1]]['id']))[:k]
        result.append({'id': query['id'], 'prompt': query['prompt'], 'candidates': [
            {'id': historical[j]['id'], 'prompt': historical[j]['prompt'], 'cosine': score}
            for score, j in ranked]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--humaneval', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    with gzip.open(args.humaneval, 'rt', encoding='utf8') as stream:
        queries = [dict(id=r['task_id'], prompt=r['prompt']) for r in map(json.loads, stream)]
    historical = [r for r in json.loads(args.historical.read_text(encoding='utf8'))['examples']
                  if r.get('source') == 'mbpp']
    report = {'method': 'word unigram/bigram cosine, sublinear TF, smoothed IDF, no stopword removal',
              'top_k': args.top_k, 'ready_for_independent_test': False,
              'limitations': 'Retrieval is not semantic review; unreturned pairs may overlap. No model outputs used.',
              'sources_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.humaneval, args.historical, Path(__file__))},
              'records': neighbors(queries, historical, args.top_k)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f'Retrieved candidates for {len(queries)} items; no independence decisions made')


if __name__ == '__main__':
    main()
