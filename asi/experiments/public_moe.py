"""Pinned public OLMoE pilot: download, calibrate, classify and compare AirLLM."""
import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import shutil
import time

from asi import ROOT
from asi.taxonomy import Taxonomy, build_profile, digest

MODEL = 'allenai/OLMoE-1B-7B-0924-Instruct'
REVISION = '7f1c97f440f06ce36705e4f2b843edb5925f4498'
AIRLLM_REVISION = 'ae70eefdec83e868f9825613653e34d3c442045c'
PARAMETERS = 6919161856
WEIGHT_BYTES = PARAMETERS * 2
CATEGORIES = ['computer_science_software_engineering', 'mathematics_statistics', 'life_sciences_biology']


def hardware(directory, ram_gib=32, disk_gib=650 * 10**9 / 2**30):
    import psutil
    import torch
    directory = Path(directory)
    ancestor = directory
    while not ancestor.exists():
        ancestor = ancestor.parent
    vm = psutil.virtual_memory()
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {'name': torch.cuda.get_device_name(), 'free_bytes': free, 'total_bytes': total}
    return {'ram_total_bytes': vm.total, 'ram_available_bytes': vm.available,
            'ram_budget_bytes': int(ram_gib * 2**30), 'disk_budget_bytes': int(disk_gib * 2**30),
            'disk_free_bytes': shutil.disk_usage(ancestor).free, 'gpu': gpu,
            'model_weight_bytes': WEIGHT_BYTES, 'estimated_disk_original_plus_airllm_bytes': 3 * WEIGHT_BYTES,
            'backing': 'disk with adaptive bounded RAM cache; full checkpoint RAM is not required',
            'disk_fits_estimate': 3 * WEIGHT_BYTES <= min(shutil.disk_usage(ancestor).free, disk_gib * 2**30),
            'limits': 'Preflight estimates and per-phase RSS checks; not OS-enforced hard quotas.'}


def read_lock(directory):
    lock = json.loads((directory / 'asi_source.json').read_text())
    if lock.get('model') != MODEL or lock.get('revision') != REVISION:
        raise ValueError('Unexpected public checkpoint identity')
    # Verify weights too: file size alone would miss same-size checkpoint edits.
    for filename, size in lock['files'].items():
        if not (directory / filename).is_file() or (directory / filename).stat().st_size != size:
            raise ValueError('Snapshot files changed/missing: ' + filename)
        if sha256(directory / filename) != lock['sha256'][filename]:
            raise ValueError('Snapshot content changed: ' + filename)
    return lock


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def download(directory, report):
    from huggingface_hub import snapshot_download
    if not report['disk_fits_estimate']:
        raise ValueError('Insufficient disk budget for checkpoint and AirLLM copies')
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=directory,
                      allow_patterns=['*.safetensors', '*.json'], max_workers=2)
    files = {p.name: p.stat().st_size for p in directory.iterdir() if p.suffix in ('.json', '.safetensors') and p.name != 'asi_source.json'}
    lock = {'schema': 1, 'model': MODEL, 'revision': REVISION, 'files': files,
            'sha256': {name: sha256(directory / name) for name in files},
            'weight_format': 'BF16 original safetensors', 'airllm_revision': AIRLLM_REVISION}
    (directory / 'asi_source.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    return lock


def input_ids(tokenizer, text, limit=128):
    import torch
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
    if len(ids) > limit:
        raise ValueError('Prompt exceeds frozen benchmark length; do not truncate silently')
    return torch.tensor([ids], dtype=torch.long)


def sample_examples(path, taxonomy, split, per_label):
    groups = {label: [] for label in CATEGORIES}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] != split:
            continue
        labels = {ancestor for label in row['labels'] for ancestor in taxonomy.ancestors(label)}
        relevant = [label for label in CATEGORIES if label in labels]
        # Single-domain pilot: keep evidence unambiguous.
        if len(relevant) == 1 and len(groups[relevant[0]]) < per_label:
            groups[relevant[0]].append({'text': row['text'], 'category': relevant[0], 'id': row['id']})
    if any(len(rows) < per_label for rows in groups.values()):
        raise ValueError('Insufficient independent examples per category')
    return [row for rows in groups.values() for row in rows]


def choose_capacity(model, k, free_bytes, reserve_bytes):
    blocks = [layer.mlp for layer in model.model.layers]
    expert_ids = {id(p) for block in blocks for p in block.experts.parameters()}
    base = sum(p.numel() * p.element_size() for p in model.parameters() if id(p) not in expert_ids)
    expert = blocks[0].experts
    one = (expert.gate_up_proj[0].numel() + expert.down_proj[0].numel()) * expert.gate_up_proj.element_size()
    capacity = min(len(blocks) * model.config.num_experts, max(0, int((free_bytes - base - reserve_bytes) // one)))
    if capacity < k * len(blocks):
        raise ValueError('Backbone plus requested fixed context does not fit estimated VRAM')
    return {'backbone_weight_bytes': base, 'expert_bytes': one, 'capacity': capacity,
            'requested_fixed_experts': k * len(blocks), 'reserve_bytes': reserve_bytes}


def check_rss(budget):
    import psutil
    rss = psutil.Process().memory_info().rss
    if rss > budget:
        raise MemoryError('Process RSS exceeded configured RAM budget')
    return rss


def calibrate(model, tokenizer, cache, examples, k, ram_budget=None):
    import torch
    from asi.models.hf_olmoe import CategoryCalibration
    from asi.runtime.classifier import IntegratedHead, features
    collector = CategoryCalibration(model.config.num_hidden_layers, model.config.num_experts)
    cache.callbacks.append(collector.observe)
    vectors, targets = [], []
    try:
        with torch.inference_mode():
            for row in examples:
                print('Calibrating: ' + row['category'], flush=True)
                collector.labels = [row['category']]
                ids = input_ids(tokenizer, row['text']).to(cache.device)
                model(input_ids=ids, use_cache=False)
                vectors.append(features(model.model.embed_tokens, [ids[0].tolist()]).cpu())
                targets.append([float(row['category'] == label) for label in CATEGORIES])
                if ram_budget:
                    check_rss(ram_budget)
    finally:
        cache.callbacks.remove(collector.observe)
    torch.manual_seed(1337)
    # Clone out of inference-mode tensors before training only the small head.
    x = torch.cat(vectors).clone()
    y = torch.tensor(targets)
    head = IntegratedHead(model.config.hidden_size, len(CATEGORIES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    for _ in range(200):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return collector.mapping(CATEGORIES, k), head.eval()


def decode(model, ids, steps, device, after_step=None, stop_token_ids=None):
    import torch
    timings, generated, past = [], [], None
    current = ids.to(device)
    with torch.inference_mode():
        for step in range(steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            past = output.past_key_values
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            generated.append(int(token.item()))
            current = token
            if after_step:
                after_step()
            if stop_token_ids and generated[-1] in stop_token_ids:
                break
    return {'token_ids': generated, 'ttft_seconds': timings[0], 'decode_seconds': sum(timings[1:]),
            'generation_seconds': sum(timings), 'decode_tokens_per_second': (len(generated) - 1) / sum(timings[1:]) if len(generated) > 1 else None,
            'fixed_length': not bool(stop_token_ids), 'eos_policy': 'stop at EOS or token budget' if stop_token_ids else 'ignore EOS equally for timing; not normal chat stopping'}


def compare(paths):
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    native = [r for r in reports if r.get('backend') == 'native']
    if len(native) != 1 or any(not r.get('executed') or not r.get('records') for r in reports):
        raise ValueError('Need exactly one native baseline and executed benchmark reports')
    baseline = native[0]
    base_rows = {(r['repeat'], r['prompt_id']): r for r in baseline['records']}
    output = {'reference': 'native', 'runs': [], 'limitations': [
        'Token agreement measures output changes, not correctness or task accuracy.',
        'Three synthetic held-out prompts are an integration pilot, not a quality benchmark.',
        'Native ASI uses a Python expert loop; differences from AirLLM include kernel implementation.',
        'First pass and later passes are reported separately; OS disk cache is not flushed.']}
    for report in reports:
        for key in ['source_identity', 'bundle_identity', 'versions', 'dtype', 'attention', 'prompt_protocol']:
            if report.get(key) != baseline.get(key):
                raise ValueError('Incomparable reports: ' + key)
        if report['hardware']['gpu'] != baseline['hardware']['gpu']:
            # Free memory naturally varies between processes; physical GPU must match.
            for key in ['name', 'total_bytes']:
                if report['hardware']['gpu'][key] != baseline['hardware']['gpu'][key]:
                    raise ValueError('Compare reports from the same GPU')
        rows = {(r['repeat'], r['prompt_id']): r for r in report['records']}
        if len(rows) != len(report['records']) or rows.keys() != base_rows.keys():
            raise ValueError('Prompt/repeat sets differ')
        for key, row in rows.items():
            if row['input_ids'] != base_rows[key]['input_ids'] or len(row['token_ids']) != len(base_rows[key]['token_ids']):
                raise ValueError('Prompt tokens or generation lengths differ')
        matched = sum(sum(a == b for a, b in zip(row['token_ids'], base_rows[key]['token_ids'])) for key, row in rows.items())
        tokens = sum(len(r['token_ids']) for r in rows.values())
        summaries = {}
        for phase, selected in [('first_pass', [r for r in rows.values() if r['repeat'] == 0]),
                                ('later_passes', [r for r in rows.values() if r['repeat'] > 0])]:
            if not selected:
                continue
            summaries[phase] = {'prompts': len(selected),
                'mean_ttft_seconds': sum(r['ttft_seconds'] for r in selected) / len(selected),
                'mean_response_seconds_including_preparation': sum(r['generation_seconds'] + r['prepare_seconds'] + r['classifier_seconds'] for r in selected) / len(selected),
                'peak_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in selected),
                'sampled_max_rss_bytes': max(r['rss_bytes'] for r in selected)}
        output['runs'].append({'backend': report['backend'], 'routing': report['routing'],
            'token_position_agreement_with_native': matched / tokens,
            'exact_response_agreement_with_native': sum(row['token_ids'] == base_rows[key]['token_ids'] for key, row in rows.items()) / len(rows),
            'load_seconds': report['load_seconds'], 'phases': summaries})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'download', 'calibrate', 'benchmark', 'compare'])
    parser.add_argument('--reports', nargs='+', type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'public_models/olmoe')
    parser.add_argument('--taxonomy', type=Path, default=ROOT / 'configs/knowledge_v1.json')
    parser.add_argument('--examples', type=Path, default=ROOT / 'data/classifier_seed_v1/train_val.jsonl')
    parser.add_argument('--bundle', type=Path, default=ROOT / 'results/public_olmoe_calibration')
    parser.add_argument('--backend', choices=['native', 'fixed', 'airllm'], default='native')
    parser.add_argument('--routing', choices=['oracle', 'classifier'], default='oracle')
    parser.add_argument('--ram-gib', type=float, default=32)
    parser.add_argument('--ram-cache-gib', type=float, default=None, help='Expert RAM cache ceiling; default uses remaining process/available RAM dynamically')
    parser.add_argument('--disk-gib', type=float, default=650 * 10**9 / 2**30, help='Disk budget in GiB; default is 650 decimal GB')
    parser.add_argument('--reserve-gib', type=float, default=2)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--calibration-per-category', type=int, default=8)
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.ram_gib, args.disk_gib, args.experts, args.calibration_per_category, args.new_tokens, args.repeats) <= 0 or args.reserve_gib < 0:
        parser.error('Use a new report path and positive budgets')
    if args.action == 'compare':
        if not args.reports:
            parser.error('compare requires --reports with native and other executed reports')
        result = compare(args.reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    report = hardware(args.model_dir, args.ram_gib, args.disk_gib)
    result = {'hardware': report, 'action': args.action, 'model': MODEL, 'revision': REVISION, 'executed': False}
    if args.action == 'download':
        result['source'] = download(args.model_dir, report)
        result['download_complete'] = True
    elif args.action != 'inspect':
        import torch
        from asi.models.hf_olmoe import OlmoeExpertCache
        from asi.runtime.classifier import IntegratedHead, features
        if not report['gpu']:
            result['blocked'] = 'CUDA unavailable'
        elif not report['disk_fits_estimate']:
            result['blocked'] = 'Disk budget insufficient'
        else:
            lock = read_lock(args.model_dir)
            identity = digest(lock)
            taxonomy = Taxonomy.read(args.taxonomy)
            cache = None
            start = time.perf_counter()
            if args.action == 'calibrate' or args.backend != 'airllm':
                from asi.models.olmoe_storage import load_disk_backed
                ram_limit = int((args.ram_cache_gib if args.ram_cache_gib is not None else args.ram_gib) * 2**30)
                if ram_limit < 0:
                    parser.error('RAM cache cannot be negative')
                model, tokenizer, store = load_disk_backed(args.model_dir, ram_limit, report['ram_budget_bytes'])
                capacity = choose_capacity(model, args.experts, report['gpu']['free_bytes'], int(args.reserve_gib * 2**30))
                result['capacity_plan'] = capacity
                # Equal strict budget across native and fixed, even if more VRAM is free.
                cache = OlmoeExpertCache(model, capacity=capacity['requested_fixed_experts'], store=store)
            else:
                from asi.models.airllm_olmoe import load_airllm
                model = load_airllm(args.model_dir, device='cuda:0', dtype=torch.bfloat16,
                                       max_seq_len=512, layer_shards_saving_path=str(args.model_dir / 'airllm_layers'),
                                       prefetching=False, delete_original=False)
                model.model.set_attn_implementation('eager')
                tokenizer = model.tokenizer
            check_rss(report['ram_budget_bytes'])
            result['load_seconds'] = time.perf_counter() - start
            result['versions'] = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'huggingface-hub']}
            result['source_identity'] = identity
            try:
                if args.action == 'calibrate':
                    rows = sample_examples(args.examples, taxonomy, 'train', args.calibration_per_category)
                    mapping, head = calibrate(model, tokenizer, cache, rows, args.experts, report['ram_budget_bytes'])
                    args.bundle.mkdir(parents=True, exist_ok=False)
                    manifest = {'pools': {label: {'broad_domains': [label]} for label in CATEGORIES}}
                    profile = build_profile(taxonomy, manifest, MODEL)
                    profile.update(status='public_model_calibrated', source_identity=identity)
                    for binding in profile['bindings'].values():
                        binding['evidence'] = 'observed_native_router_on_synthetic_train_prompts_not_proven_expertise'
                    bundle = {'schema': 1, 'source_identity': identity, 'taxonomy': taxonomy.identity, 'categories': CATEGORIES,
                              'mapping': mapping, 'profile': profile, 'calibration_examples': rows, 'experts_per_category': args.experts,
                              'evaluation_examples': sample_examples(args.examples, taxonomy, 'val', 1)}
                    (args.bundle / 'bundle.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
                    torch.save({'state': head.state_dict(), 'hidden': head.hidden}, args.bundle / 'input_head.pt')
                    result['calibrated_categories'] = CATEGORIES
                else:
                    bundle = json.loads((args.bundle / 'bundle.json').read_text())
                    result['bundle_identity'] = digest(bundle)
                    if bundle['source_identity'] != identity or bundle['taxonomy'] != taxonomy.identity:
                        raise ValueError('Calibration identity mismatch')
                    if args.backend == 'fixed' and args.experts != bundle['experts_per_category']:
                        raise ValueError('Use the same K as the calibrated bundle')
                    head = None
                    if args.backend == 'fixed' and args.routing == 'classifier':
                        saved = torch.load(args.bundle / 'input_head.pt', map_location='cpu', weights_only=True)
                        head = IntegratedHead(model.config.hidden_size, len(CATEGORIES), saved['hidden'])
                        head.load_state_dict(saved['state']); head.eval()
                    result.update(backend=args.backend, routing=args.routing, records=[], dtype='bfloat16', attention='eager',
                                  classifier_parameters=sum(p.numel() for p in head.parameters()) if head else 0,
                                  classifier_weight_bytes=sum(p.numel() * p.element_size() for p in head.parameters()) if head else 0,
                                  prompt_protocol='single user chat template, batch=1, greedy, fixed generation length',
                                  airllm_prefetch=False if args.backend == 'airllm' else None)
                    if args.backend == 'airllm':
                        result['airllm_version'] = importlib.metadata.version('airllm')
                        direct = importlib.metadata.distribution('airllm').read_text('direct_url.json')
                        result['airllm_source'] = json.loads(direct) if direct else None
                        if not direct or json.loads(direct).get('vcs_info', {}).get('commit_id') != AIRLLM_REVISION:
                            raise ValueError('Install the pinned AirLLM commit from requirements-public-moe.txt')
                    for repeat in range(args.repeats):
                        for row in bundle['evaluation_examples']:
                            print(f"Benchmark {args.backend}, repeat {repeat}: {row['id']}", flush=True)
                            ids = input_ids(tokenizer, row['text'])
                            label = row['category']
                            started = time.perf_counter()
                            if head is not None:
                                with torch.no_grad():
                                    scores = head(features(model.model.embed_tokens, [ids[0].tolist()]).cpu()).sigmoid()[0]
                                label = CATEGORIES[int(scores.argmax())]
                            classifier_seconds = time.perf_counter() - started if head else 0.
                            started = time.perf_counter()
                            if cache:
                                cache.set_context(bundle['mapping'][label] if args.backend == 'fixed' else None)
                            torch.cuda.synchronize()
                            preparation = time.perf_counter() - started
                            before = cache.snapshot() if cache else None
                            torch.cuda.reset_peak_memory_stats()
                            measured = decode(model, ids, args.new_tokens, 'cuda', lambda: check_rss(report['ram_budget_bytes']))
                            after = cache.snapshot() if cache else None
                            if cache and args.backend == 'fixed' and after.get('loads', 0) != before.get('loads', 0):
                                raise RuntimeError('Fixed inference loaded experts inside the response')
                            result['records'].append({'repeat': repeat, 'prompt_id': row['id'], 'expected_category': row['category'],
                                'selected_category': label, 'input_ids': ids[0].tolist(), 'classifier_seconds': classifier_seconds,
                                'prepare_seconds': preparation, **measured, 'text': tokenizer.decode(measured['token_ids']),
                                'cache_before': before, 'cache_after': after, 'rss_bytes': check_rss(report['ram_budget_bytes']),
                                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_reserved_bytes': torch.cuda.memory_reserved()})
                if cache:
                    result['cache_final'] = cache.snapshot()
                result['executed'] = True
            finally:
                if cache:
                    cache.close()
    result['limitations'] = ['Pilot synthetic calibration, three categories, not comprehensive specialization.',
        'Input head is a closed-set three-category pilot; argmax, not general multilabel task decomposition.',
        'Fixed category changes routing; speed must be interpreted together with quality degradation.',
        'AirLLM uses layer streaming, no compression/prefetch; first repeat is cold relative to later repeats.',
        'AirLLM includes our OLMoE layout conversion of derived shards once during setup; counted in load_seconds.',
        'RAM checks are per phase/token, not continuous RSS peaks or a hard operating-system limit.',
        'AirLLM transfer bytes and disk I/O are not instrumented in this pilot; null is not zero.',
        'No quality conclusion or speedup can be inferred from configuration or unexecuted reports.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'executed': result['executed'], 'blocked': result.get('blocked'), 'output': str(args.output)}, indent=2))
