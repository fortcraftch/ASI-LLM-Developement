"""Development-only comparison of three mixture scales with one frozen mask."""
import argparse
import json
from pathlib import Path
import time

from asi.experiments.public_moe import sha256, read_lock, input_ids, decode
from asi.taxonomy import digest


ARMS = ('original', 'restricted', 'restricted_unit_mass', 'restricted_native_mass')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_plan(plan):
    if plan.get('purpose') != 'development_scale_pilot' or plan.get('test_frozen') is not False:
        raise ValueError('This runner requires an explicitly developmental plan')
    if plan['arms'] != list(ARMS):
        raise ValueError('Scale comparison must contain all four predefined arms')
    examples = plan['examples']
    ids = [row['id'] for row in examples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate evaluation IDs')
    if any(row['split'] != 'dev' for row in examples):
        raise ValueError('Do not select configurations on test examples')
    if any(not plan['reference_review'].get(i) for i in ids):
        raise ValueError('Reference review missing')
    protocol = plan['protocol']
    if protocol['active'] != 8 or protocol['resident_per_layer'] != 32:
        raise ValueError('This pilot freezes 32 resident and 8 active experts')
    if protocol['new_tokens'] < 1 or protocol['max_length'] < 2:
        raise ValueError('Invalid token budget')
    for path, expected in plan['files_sha256'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Frozen source changed: ' + path)
    required = [plan[k] for k in ('corpus_source', 'audit_source', 'reconciliation_source', 'mapping_source')]
    if any(path not in plan['files_sha256'] for path in required):
        raise ValueError('An input source is not pinned')
    corpus = {r['id']: r for r in read(plan['corpus_source'])['examples']}
    candidates = {r['id'] for r in read(plan['audit_source'])['records'] if r['candidate_only']}
    legacy = {r['id'] for r in read(plan['reconciliation_source'])['files']}
    if not set(ids) <= candidates or set(ids) & legacy:
        raise ValueError('Selected IDs overlap historical exclusions')
    if any(corpus.get(row['id']) != row for row in examples):
        raise ValueError('Evaluation row differs from frozen corpus')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('configs/routing_scale_dev_v1.json'))
    p.add_argument('--model-dir', type=Path, default=Path('public_models/olmoe'))
    p.add_argument('--output', type=Path, default=Path('results/routing_scale_dev_20261011'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    plan = read(args.plan)
    decoding_plan = plan if plan.get('purpose') == 'development_decoding_control' else None
    if decoding_plan:
        parent = Path(decoding_plan['parent_plan'])
        if sha256(parent) != decoding_plan['parent_sha256']:
            raise ValueError('Decoding parent changed')
        plan = read(parent)
        if decoding_plan['test_frozen'] is not False or decoding_plan['arms'] != ['original', 'restricted']:
            raise ValueError('Invalid development decoding arms')
        if decoding_plan['example_ids'] != [r['id'] for r in plan['examples']]:
            raise ValueError('Decoding examples changed')
        expected_policies = [dict(name='greedy', temperature=0, top_p=1, seeds=[1337]),
                             dict(name='sample', temperature=.7, top_p=.9, seeds=[1337,2027,31415])]
        if decoding_plan['policies'] != expected_policies:
            raise ValueError('Unexpected decoding policies')
        if decoding_plan['protocol']['new_tokens'] != plan['protocol']['new_tokens']:
            raise ValueError('Generation length differs')
        required = dict(active=8, restricted_resident_per_layer=32, cache_capacity=512,
                        mixture='native restricted probabilities; no renormalization',
                        stop_at_eos=True, repetition_penalty=1, no_repeat_ngram_size=0,
                        adaptation='none', dtype='BF16 autocast; FP32 gate')
        if any(decoding_plan['protocol'].get(k) != v for k,v in required.items()):
            raise ValueError('Unsupported decoding protocol; refusing silent overrides')
    validate_plan(plan)
    configurations = [(arm, 'greedy', 0., 1., 1337) for arm in ARMS]
    if decoding_plan:
        configurations = [(arm, policy['name'], policy['temperature'], policy['top_p'], seed)
                          for arm in decoding_plan['arms'] for policy in decoding_plan['policies'] for seed in policy['seeds']]
        if len(configurations)*len(plan['examples']) != decoding_plan['expected_records']:
            raise ValueError('Unexpected decoding coverage')
    if args.dry_run:
        print(f"Validated development plan: {len(plan['examples'])} examples x {len(configurations)} configurations; no model loaded.")
        return
    path = args.output / 'report.json'
    if path.exists() and not args.resume:
        p.error('Existing report: use --resume or a fresh output')
    import torch
    import psutil
    import transformers
    from asi.models.olmoe_storage import load_disk_backed
    from asi.models.hf_olmoe import OlmoeExpertCache
    from asi.experiments.recovery import objective, generation_metrics
    from asi.experiments.recovery_protocol import durable_save

    calibration = read(plan['mapping_source'])
    source = digest(read_lock(args.model_dir))
    if calibration['source_identity'] != source:
        raise ValueError('Model/calibration identity mismatch')
    mappings = calibration['mappings']['32']
    for category in {r['category'] for r in plan['examples']}:
        if len(mappings[category]) != 16 or any(len(v['experts']) != 32 for v in mappings[category].values()):
            raise ValueError('Expected 16 layers with 32 candidates each')
    code = [Path(__file__), Path('asi/models/hf_olmoe.py'), Path('asi/models/olmoe_storage.py'),
            Path('asi/experiments/public_moe.py'), Path('asi/experiments/recovery.py')]
    identity = dict(plan_sha256=sha256(args.plan), source=source,
                    code_sha256={str(x.resolve()): sha256(x) for x in code},
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    report = read(path) if path.exists() else dict(identity=identity, completed=False, records=[],
             limitations=plan['limitations'], protocol=plan['protocol'], examples=plan['examples'])
    if decoding_plan:
        report['decoding_plan'] = decoding_plan
        report['limitations'] = decoding_plan['limitations']
        report['protocol'] = dict(plan['protocol'], decoding='Per-record policy in decoding_plan')
    if report['identity'] != identity:
        raise ValueError('Resume identity changed; preserve this report and start a new attempt')
    if report['completed']:
        print('Already completed:', path)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    durable_save(path, report)
    protocol = plan['protocol']
    torch.manual_seed(protocol['seed'])
    model, tokenizer, store = load_disk_backed(args.model_dir, 2 * 2**30, 12 * 2**30)
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    cache = OlmoeExpertCache(model, 'cuda', capacity=512, store=store)
    device = torch.device('cuda')
    def record_key(r):
        return (r['arm'], r['id'], r.get('decoding_policy', 'greedy'), r.get('sampling_seed', 1337))
    done = {record_key(r) for r in report['records']}
    expected = {(arm, row['id'], policy, seed) for row in plan['examples']
                for arm, policy, temperature, top_p, seed in configurations}
    if len(done) != len(report['records']):
        raise ValueError('Duplicate results in resume report')
    if not done <= expected:
        raise ValueError('Unexpected resume records')
    try:
        for row in plan['examples']:
            prompt = input_ids(tokenizer, row['prompt'], limit=protocol['max_length'])
            tokens = prompt[0].tolist() + tokenizer.encode(row['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(tokens) > protocol['max_length']:
                raise ValueError('Full reference exceeds budget: ' + row['id'])
            example = dict(tokens=tokens, prompt_length=prompt.shape[1])
            ordered = list(configurations)
            if decoding_plan:
                import random
                random.Random(str(decoding_plan['protocol']['order_seed']) + ':' + row['id']).shuffle(ordered)
            for arm, policy, temperature, top_p, seed in ordered:
                if (arm, row['id'], policy, seed) in done:
                    continue
                print(arm, row['id'], flush=True)
                torch.cuda.synchronize()
                before = cache.snapshot()
                began = time.perf_counter()
                cache.set_context(None if arm == 'original' else mappings[row['category']],
                                  mixture='restricted' if arm == 'original' else arm, active=8)
                prepare_seconds = time.perf_counter() - began
                ready = cache.snapshot()
                with torch.inference_mode():
                    _, ce, _ = objective(model, example, device, alpha=1.)
                ce_value = ce.item()
                del ce
                if not torch.isfinite(torch.tensor(ce_value)):
                    raise RuntimeError('Non-finite CE')
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = decode(model, prompt, protocol['new_tokens'], device,
                                       stop_token_ids={tokenizer.eos_token_id}, temperature=temperature,
                                       top_p=top_p, seed=seed)
                after = cache.snapshot()
                if arm != 'original' and any(after.get(k, 0) != ready.get(k, 0)
                                             for k in ('loads', 'host_to_device_bytes')):
                    raise RuntimeError('Fixed-mask inference loaded an expert')
                text = tokenizer.decode(generated['token_ids'], skip_special_tokens=True)
                report['records'].append(dict(arm=arm, id=row['id'], category=row['category'],
                    decoding_policy=policy, sampling_seed=seed, temperature=temperature, top_p=top_p,
                    ce=ce_value, text=text, **generated, input_ids=prompt[0].tolist(),
                    eos=generated['token_ids'][-1] == tokenizer.eos_token_id,
                    metrics=generation_metrics(generated['token_ids'], text, row),
                    prepare_seconds=prepare_seconds, before=before, ready=ready, after=after,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    rss_bytes=psutil.Process().memory_info().rss))
                durable_save(path, report)
                print(f"Saved {len(report['records'])}/{len(plan['examples'])*len(configurations)}; {policy} seed={seed}; CE={ce_value:.3f}", flush=True)
        if {record_key(r) for r in report['records']} != expected:
            raise ValueError('Incomplete execution coverage')
        report['completed'] = True
        durable_save(path, report)
    finally:
        cache.close()


if __name__ == '__main__':
    main()
