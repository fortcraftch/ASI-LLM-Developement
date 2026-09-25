"""Offline reporting of paired quality and separately repeated swapping observations."""
import json
import math
from pathlib import Path

from asi.experiments.comparison import digest, verify_bundle


def quality(rows):
    tokens = sum(row['tokens'] for row in rows)
    if not tokens:
        raise ValueError('Cannot score an empty run')
    if any(len(r['argmax']) != r['tokens'] for r in rows):
        raise ValueError('Recorded predictions do not cover every scored token')
    nll = sum(row['nll'] * row['tokens'] for row in rows)/tokens
    domains = sorted({r['true_class'] for r in rows})
    by_domain = {}
    for name in domains:
        subset = [r for r in rows if r['true_class'] == name]
        n = sum(r['tokens'] for r in subset)
        by_domain[name] = {'tokens': n, 'nll': sum(r['nll']*r['tokens'] for r in subset)/n}
    return {'tokens': tokens, 'windows': len(rows), 'nll': nll,
            'perplexity': math.exp(nll) if nll < 700 else None,
            'token_accuracy': sum(row['correct_tokens'] for row in rows)/tokens,
            'by_domain': by_domain,
            'macro_domain_nll': sum(r['nll'] for r in by_domain.values())/len(by_domain)}


def paired(left, right, seed):
    import numpy as np
    a, b = ({r['window']: r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b) or not a:
        raise ValueError('Paired comparison needs identical unique window IDs')
    keys = sorted(a)
    if any(a[k]['tokens'] != b[k]['tokens'] for k in keys):
        raise ValueError('Paired comparison token mismatch')
    weights = np.array([a[k]['tokens'] for k in keys])
    diffs = np.array([a[k]['nll']-b[k]['nll'] for k in keys])
    rng = np.random.default_rng(seed)
    # Bound temporary memory for large evaluations.
    boot = []
    for _ in range(1000):
        ids = rng.integers(0, len(keys), len(keys))
        boot.append(float(np.average(diffs[ids], weights=weights[ids])))
    delta = float(np.average(diffs, weights=weights))
    agreement = sum(sum(x == y for x, y in zip(a[k]['argmax'], b[k]['argmax'])) for k in keys)/sum(weights)
    routes_equal = all([e['routes'] for e in a[k]['events']] == [e['routes'] for e in b[k]['events']] for k in keys)
    return {'delta_nll': delta, 'perplexity_ratio': math.exp(delta) if delta < 700 else None,
            'window_bootstrap_95': np.percentile(boot, [2.5,97.5]).tolist(),
            'argmax_agreement': float(agreement), 'routes_equal': routes_equal,
            'note': 'Descriptive window bootstrap; not a document confidence interval.'}


def timing(rows):
    import numpy as np
    result = {'observations': len(rows), 'generated_tokens': sum(len(r.get('generated_tokens', [])) for r in rows),
              'prepare_h2d_bytes': sum(r['prepare_transfers']['host_to_device_bytes'] for r in rows),
              'classification_seconds': sum(r['classification_seconds'] for r in rows),
              'prepare_seconds': sum(r['prepare_seconds'] for r in rows)}
    for phase in ('prefill', 'teacher_forced', 'decode'):
        events = [e for row in rows for e in row['events'] if e['phase'] == phase]
        count = len(events)
        if not count:
            continue
        requests = sum(e['hits']+e['misses'] for e in events)
        result[phase] = {'steps': count, 'seconds': sum(e['seconds'] for e in events),
                         'h2d_bytes': sum(e['host_to_device_bytes'] for e in events),
                         'evictions': sum(e['evictions'] for e in events),
                         'fraction_steps_without_loads': sum(e['misses'] == 0 for e in events)/count,
                         'hit_rate': sum(e['hits'] for e in events)/requests if requests else None,
                         'step_p50_ms': float(np.percentile([e['seconds']*1000 for e in events], 50)),
                         'step_p95_ms': float(np.percentile([e['seconds']*1000 for e in events], 95))}
    if rows:
        rss = [r['memory']['process_rss_bytes'] for r in rows if r['memory']['process_rss_bytes'] is not None]
        result['max_observed_process_rss_bytes'] = max(rss) if rss else None
        result['cuda_peak_allocated_bytes'] = max(r['memory']['cuda_peak_allocated_bytes'] for r in rows)
    return result


def summarize(suite, results, output):
    suite, results, output = Path(suite), Path(results), Path(output)
    config, _ = verify_bundle(suite)
    expected_hash = digest(suite/'manifest.json')
    jobs = {job['id']: job for job in json.loads((suite/'jobs.json').read_text(encoding='utf-8'))}
    windows = json.loads((suite/'windows.json').read_text(encoding='utf-8'))
    sessions = json.loads((suite/'sessions.json').read_text(encoding='utf-8'))
    completed, states, loaded, identities = {}, {}, {}, {}
    runner_hash = None
    for report in sorted(results.glob('*/report.json')):
        header = json.loads(report.read_text(encoding='utf-8'))
        job = header['job']; jid = job['id']
        if header['suite_sha256'] != expected_hash or jobs.get(jid) != job:
            raise ValueError('Result belongs to a different frozen experiment: ' + str(report))
        if jid in states:
            raise ValueError('Duplicate result for job: ' + jid)
        identity = (header['checkpoint_sha256'], header['architecture_sha256'], header.get('mapping_sha256'))
        role = job['role']
        if role in identities and identities[role] != identity:
            raise ValueError('Weights/architecture/mapping changed between jobs for ' + role)
        identities[role] = identity
        if runner_hash is not None and runner_hash != header['runner_sha256']:
            raise ValueError('Cannot mix different versions of the measurement runner')
        runner_hash = header['runner_sha256']
        states[jid] = {'status': header['status'], 'reason': header.get('reason')}
        if header['status'] != 'complete':
            continue
        rows = [json.loads(line) for line in (report.parent/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        if not rows:
            raise ValueError('Completed job contains no observations')
        if job['phase'] == 'quality':
            expected = {w['id'] for w in windows if w['length'] == job['length']+1}
            observed = [r['window'] for r in rows]
        else:
            expected = {(s['id'], i) for s in sessions for i in range(len(s['turns']))}
            observed = [(r['session'], r['turn']) for r in rows]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError('Incomplete or duplicate observations in completed job ' + jid)
        loaded[jid] = (header, rows)
        completed[jid] = {'job': job, 'timing': timing(rows)}
        if job['phase'] == 'quality':
            completed[jid]['quality'] = quality(rows)
        else:
            completed[jid]['by_scenario'] = {scenario: timing([r for r in rows if r['scenario'] == scenario])
                                              for scenario in sorted({r['scenario'] for r in rows})}
    for jid, (header, rows) in loaded.items():
        job = header['job']
        if job['phase'] != 'quality':
            continue
        for baseline_role in dict.fromkeys([job['role'], 'original']):
            matches = [(bid, bheader, brows) for bid, (bheader, brows) in loaded.items()
                       if bheader['job']['role'] == baseline_role and bheader['job']['policy'] == 'resident'
                       and bheader['job']['phase'] == 'quality' and bheader['job']['length'] == job['length']]
            for bid, bheader, brows in matches:
                contrast = paired(rows, brows, config['seed'])
                contrast['same_checkpoint'] = header['checkpoint_sha256'] == bheader['checkpoint_sha256']
                contrast['causal_training_claim'] = False
                completed[jid].setdefault('contrasts', {})[bid] = contrast
                if job['policy'] == 'exact' and baseline_role == job['role']:
                    contrast['exact_control_pass'] = (contrast['same_checkpoint'] and contrast['routes_equal'] and
                        contrast['argmax_agreement'] == 1 and abs(contrast['delta_nll']) <= 1e-4)
    payload = {'state': 'offline_summary', 'completed_jobs': completed, 'reported_states': states,
               'missing_jobs': sorted(set(jobs)-set(states)),
               'limitations': ['Timing includes instrumentation; synchronized serial prefill in every arm.',
                              'Repeats remain separate; repeated tokens are not new quality samples.',
                              'Missing or skipped jobs are never interpreted as zero loss/cost.',
                              'RAM backing only in this suite; SSD tier has its separate existing study.',
                              'Compare each context length separately. No general capability score.',
                              'Inspect training and architecture provenance before causal interpretation.']}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
