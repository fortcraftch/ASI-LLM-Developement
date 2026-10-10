"""Frozen five-way generation comparison; one GPU worker at a time, resumable by variant."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

VARIANTS = ('original', 'unadapted32', 'recovered32', 'unadapted16', 'recovered16')
CATEGORIES = ('computer_science_software_engineering', 'mathematics_statistics', 'life_sciences_biology')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def choose_ids(corpus, excluded, per_category=3, seed=20261008):
    """Selection uses IDs/splits only, before observing any new output."""
    if per_category < 1:
        raise ValueError('At least one example per category required')
    rng = random.Random(seed)
    pools = {}
    for category in CATEGORIES:
        candidates = sorted(r['id'] for r in corpus['examples']
                            if r['split'] == 'test' and r['category'] == category and r['id'] not in excluded)
        if len(candidates) < per_category:
            raise ValueError('Insufficient held-out examples')
        pools[category] = rng.sample(candidates, per_category)
    return [pools[c][i] for i in range(per_category) for c in CATEGORIES]


def prepare(output, per_category=3, gpu_gib=8, ram_gib=2, native_capacity=512, prompt_file=None, variants=VARIANTS):
    if not math.isfinite(gpu_gib) or gpu_gib <= 0 or not math.isfinite(ram_gib) or ram_gib < 0 or native_capacity < 8:
        raise ValueError('Invalid memory budget')
    if len(set(variants)) != len(variants) or 'original' not in variants or not set(variants) <= set(VARIANTS):
        raise ValueError('Unique known variants including original required')
    for size in ('32', '16'):
        if ('unadapted'+size in variants) != ('recovered'+size in variants):
            raise ValueError('Both before/after controls required for each size')
    sources = {
        'corpus': 'results/recovery_corpus_v3/corpus.json',
        'study': 'results/recovery_calibration_batched_v3/report.json',
        'subset': 'results/recovery_quick_20261006/ids.json',
        'checkpoint32': 'results/recovery_quick_20261006/router-expert-lora/adaptation_best.pt',
        'checkpoint16': 'results/recovery_quick_20261006/curriculum_direct16/adaptation_best.pt',
    }
    hashes = {k: fingerprint(v) for k, v in sources.items()}
    excluded = set(read(sources['subset']))
    # Exclude all previously generated IDs, including diagnostics outside the training subset.
    for path in Path('results').glob('generation_recovery_*/*/report.json'):
        excluded.update(r['id'] for r in read(path).get('records', []))
    ids = read(prompt_file) if prompt_file else choose_ids(read(sources['corpus']), excluded, per_category)
    rows = {r['id']: r for r in read(sources['corpus'])['examples']}
    if not ids or len(ids) != len(set(ids)) or any(rows[k]['split'] != 'test' for k in ids):
        raise ValueError('Unique held-out IDs required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'prompt_ids.json').write_text(json.dumps(ids, indent=2), encoding='utf8')
    plan = dict(version=1, sources=sources, hashes=hashes, prompt_ids=ids,
                excluded_ids=sorted(excluded), variants=list(variants), new_tokens=256,
                prompt_selection='reused explicit IDs for memory control' if prompt_file else 'seeded held-out selection',
                gpu_allocator_cap_gib=gpu_gib, expert_ram_gib=ram_gib,
                native_cache_experts=native_capacity, process_ram_gib=12,
                limitations=['Exploratory small sample; not a population quality estimate.',
                             'Independent prompts with retained expert cache, not multi-turn chat history.',
                             'Oracle categories; classifier quality and latency excluded.',
                             'Allocator cap excludes driver/other processes; RSS budget trims cache, not hard OS limit.',
                             'EOS stopping produces unequal work; no warmup or timing replicates.'])
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
    return plan


def run(output):
    plan = read(output/'plan.json')
    if read(output/'prompt_ids.json') != plan['prompt_ids']:
        raise ValueError('Frozen prompt IDs changed')
    for name, path in plan['sources'].items():
        if fingerprint(path) != plan['hashes'][name]:
            raise ValueError('Frozen source changed: '+name)
    for name in plan['variants']:
        target = output/name
        if target.exists():
            report = read(target/'report.json') if (target/'report.json').exists() else {}
            if report.get('completed') and report.get('validation_plan_sha256') == fingerprint(output/'plan.json'):
                print('Already completed: '+name, flush=True)
                continue
            raise ValueError('Incomplete/unverified run preserved at '+str(target)+'; inspect before retrying')
        variant = 'original' if name == 'original' else name[-2:]
        command = [sys.executable, '-u', '-m', 'asi.experiments.recovery_generation',
                   '--variant', variant, '--output', str(target), '--prompt-ids', str(output/'prompt_ids.json'),
                   '--new-tokens', str(plan['new_tokens']), '--gpu-memory-gib', str(plan['gpu_allocator_cap_gib']),
                   '--expert-ram-gib', str(plan['expert_ram_gib']), '--process-ram-gib', str(plan['process_ram_gib']),
                   '--native-cache-experts', str(plan['native_cache_experts'])]
        for key in ('corpus', 'study', 'subset'):
            command += ['--'+key, plan['sources'][key]]
        if name.startswith('unadapted'):
            command += ['--unadapted']
        elif name.startswith('recovered'):
            command += ['--checkpoint', plan['sources']['checkpoint'+variant]]
        print('Running '+name, flush=True)
        with (output/(name+'.log')).open('x', encoding='utf8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(target/'report.json')
        report['validation_plan_sha256'] = fingerprint(output/'plan.json')
        temporary = target/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        temporary.replace(target/'report.json')
    from asi.analysis.recovery_validation_report import render
    render(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'report', 'compare-memory'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-category', type=int, default=3)
    p.add_argument('--gpu-memory-gib', type=float, default=8)
    p.add_argument('--expert-ram-gib', type=float, default=2)
    p.add_argument('--native-cache-experts', type=int, default=512)
    p.add_argument('--prompt-ids', type=Path, help='Reuse an earlier frozen selection for a memory control')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--first', type=Path)
    p.add_argument('--second', type=Path)
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.output, a.per_category, a.gpu_memory_gib, a.expert_ram_gib, a.native_cache_experts, a.prompt_ids, a.variants)
    elif a.action == 'run':
        run(a.output)
    elif a.action == 'report':
        from asi.analysis.recovery_validation_report import render
        render(a.output)
    else:
        if not a.first or not a.second:
            p.error('compare-memory requires --first and --second')
        from asi.analysis.recovery_validation_report import compare_memory
        compare_memory(a.first,a.second,a.output)


if __name__ == '__main__':
    main()
