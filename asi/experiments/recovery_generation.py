"""Matched, generation-only OLMoE recovery measurements in fresh processes."""
import argparse
import json
import math
import platform
import threading
import time
from pathlib import Path

import psutil
import torch

from asi.experiments.public_moe import decode, input_ids, read_lock, sha256
from asi.experiments.recovery import enable_routers, generation_metrics, read_examples, restore_adaptation
from asi.models.hf_olmoe import OlmoeExpertCache
from asi.models.olmoe_storage import load_disk_backed
from asi.models.recovery_lora import ExpertLoRA
from asi.taxonomy import digest

PROMPTS = ('mbpp_test_16', 'gsm8k_train_4607', 'sciq_test_772')


def select_prompts(corpus, prompt_ids, subset, external=False):
    rows = {r['id']: r for r in corpus['examples']}
    if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError('Prompt IDs must be nonempty and unique')
    selected = [rows[k] for k in prompt_ids]
    if any(r['split'] != 'test' or (not external and r['id'] not in subset) for r in selected):
        raise ValueError('Generation requires held-out test examples')
    return selected


def checkpoint_identity(source, corpus, study, state, subset):
    sizes = state['recovery_settings']['mask_sizes']
    bank = {str(k): study['mappings'][str(k)] for k in sizes}
    expected = dict(source=source, corpus=digest(corpus), mapping=digest(bank['32']),
                    experts=32, active=8, policy='restricted',
                    objective='assistant_ce_native_kl_eos_v3', mask_bank=digest(bank),
                    evaluation_subset=digest(sorted(subset)))
    if state['identity'] != expected:
        raise ValueError('Adaptation provenance differs from the frozen recovery pilot')
    return bank


def counter_delta(before, after):
    result = {k: after.get(k, 0)-before.get(k, 0)
              for k in ('loads', 'host_to_device_bytes', 'evictions', 'hits', 'misses')}
    for k in ('logical_read_bytes', 'ram_misses', 'ram_evictions'):
        result[k] = after['disk_ram'].get(k, 0)-before['disk_ram'].get(k, 0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('original', '32', '16'), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--unadapted', action='store_true', help='Restricted original weights: control for the same resident mask')
    p.add_argument('--prompt-ids', type=Path, help='JSON list of held-out corpus IDs, in execution order')
    p.add_argument('--gpu-memory-gib', type=float, help='Hard PyTorch allocator cap; excludes driver and other processes')
    p.add_argument('--native-cache-experts', type=int, default=512, help='Global layer/expert pairs for original routing')
    p.add_argument('--model-dir', type=Path, default=Path('public_models/olmoe'))
    p.add_argument('--corpus', type=Path, default=Path('results/recovery_corpus_v3/corpus.json'))
    p.add_argument('--study', type=Path, default=Path('results/recovery_calibration_batched_v3/report.json'))
    p.add_argument('--subset', type=Path, default=Path('results/recovery_quick_20261006/ids.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--new-tokens', type=int, default=256)
    p.add_argument('--expert-ram-gib', type=float, default=2)
    p.add_argument('--process-ram-gib', type=float, default=12)
    args = p.parse_args()
    if args.new_tokens < 1 or args.expert_ram_gib < 0 or args.process_ram_gib <= 0:
        p.error('Invalid token or RAM limit')
    if args.variant == 'original' and (args.checkpoint or args.unadapted):
        p.error('Original uses neither checkpoint nor --unadapted')
    if args.variant != 'original' and (bool(args.checkpoint) == args.unadapted):
        p.error('Restricted variants require exactly one of --checkpoint or --unadapted')
    if args.native_cache_experts < 8:
        p.error('Native cache requires at least eight expert slots')
    if args.gpu_memory_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        if not math.isfinite(args.gpu_memory_gib) or not 0 < args.gpu_memory_gib * 2**30 <= total:
            p.error('GPU cap must be positive and no larger than physical VRAM')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gib * 2**30 / total, 0)
    if args.output.exists():
        p.error('Output already exists; preserve earlier measurements')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    print('Verifying source and recovery identities', flush=True)
    source = digest(read_lock(args.model_dir))
    corpus = read_examples(args.corpus)
    study = json.loads(args.study.read_text(encoding='utf8'))
    subset = json.loads(args.subset.read_text(encoding='utf8'))
    if not study['completed'] or study['source_identity'] != source or study['corpus_identity'] != digest(corpus):
        raise ValueError('Calibration provenance mismatch')
    if not set(study['train_ids']) <= {r['id'] for r in corpus['examples'] if r['split'] == 'train'}:
        raise ValueError('Calibration leakage')
    prompt_ids = json.loads(args.prompt_ids.read_text(encoding='utf8')) if args.prompt_ids else list(PROMPTS)
    selected = select_prompts(corpus, prompt_ids, subset, external=bool(args.prompt_ids))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    bank = checkpoint_identity(source, corpus, study, state, subset) if state else None
    if args.unadapted:
        bank = {args.variant: study['mappings'][args.variant]}
    if bank is not None and args.variant not in bank:
        raise ValueError('Requested restriction was not trained')
    torch.manual_seed(1337)
    model, tokenizer, store = load_disk_backed(args.model_dir, int(args.expert_ram_gib*2**30),
                                               int(args.process_ram_gib*2**30))
    enable_routers(model)  # Same FP32 router precision in all three variants.
    cache = OlmoeExpertCache(model, capacity=args.native_cache_experts if args.variant == 'original' else int(args.variant)*model.config.num_hidden_layers, store=store)
    if state:
        adapter = ExpertLoRA(model, bank, rank=state['recovery_settings']['rank'])
        restore_adaptation(model, adapter, state)
        cache.expert_adapter = adapter.requires_grad_(False)
    model.requires_grad_(False).eval()
    report = {'completed': False, 'variant': args.variant, 'source_identity': source,
              'adapted': state is not None, 'mapping_identity': digest(study['mappings']),
              'mask_identity': digest(bank[args.variant]) if bank else None,
              'cache_capacity_pairs': cache.capacity,
              'corpus_identity': digest(corpus), 'checkpoint': str(args.checkpoint) if state else None,
              'checkpoint_sha256': sha256(args.checkpoint) if state else None,
              'selected_step': state['selected_step'] if state else None,
              'protocol': {'prompts': prompt_ids, 'new_tokens': args.new_tokens, 'greedy': True,
                           'gpu_allocator_cap_gib': args.gpu_memory_gib,
                           'native_cache_experts': args.native_cache_experts,
                           'stop_at_eos': True, 'active_experts': model.config.num_experts_per_tok,
                           'expert_ram_gib': args.expert_ram_gib, 'process_ram_gib': args.process_ram_gib,
                           'warmup': 'none; first prompt includes cold kernels; OS cache not flushed',
                           'category_source': 'held-out dataset label (oracle), no classifier latency'},
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'platform': platform.platform(),
              'model_disk_bytes': sum(f.stat().st_size for f in args.model_dir.rglob('*') if f.is_file()),
              'adapter_disk_bytes': args.checkpoint.stat().st_size if state else 0,
              'startup_seconds': time.perf_counter()-started, 'records': []}
    del state
    def save():
        target = args.output/'report.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(target)
    save()
    try:
        for index, row in enumerate(selected):
            print('Generating '+args.variant+' '+row['id'], flush=True)
            peak_rss = [psutil.Process().memory_info().rss]
            stop = threading.Event()
            def sample():
                process = psutil.Process()
                while not stop.wait(.05):
                    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            torch.cuda.reset_peak_memory_stats()
            before_prepare = cache.snapshot()
            start = time.perf_counter()
            try:
                if bank:
                    cache.set_context(bank[args.variant][row['category']], mixture='restricted', active=8)
                torch.cuda.synchronize()
                preparation = time.perf_counter()-start
                before = cache.snapshot()
                ids = input_ids(tokenizer, row['prompt'], limit=512)
                start = time.perf_counter()
                def progress():
                    progress.tokens += 1
                    if progress.tokens % 32 == 0:
                        print(f"  {progress.tokens} tokens, {time.perf_counter()-start:.1f}s", flush=True)
                progress.tokens = 0
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = decode(model, ids, args.new_tokens, 'cuda', after_step=progress,
                                       stop_token_ids=[tokenizer.eos_token_id])
                elapsed = time.perf_counter()-start
                after = cache.snapshot()
                if bank and after.get('loads', 0) != before.get('loads', 0):
                    raise RuntimeError('Restricted experts swapped during generation')
                text = tokenizer.decode(generated['token_ids'], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            finally:
                stop.set()
                monitor.join()
            report['records'].append({'id': row['id'], 'category': row['category'], 'prompt': row['prompt'],
                'previous_category': selected[index-1]['category'] if index else None,
                'category_changed': index > 0 and row['category'] != selected[index-1]['category'],
                'reference_answer': row['answer'], 'input_ids': ids[0].tolist(), **generated, 'text': text,
                'stop_reason': 'EOS' if generated['token_ids'][-1] == tokenizer.eos_token_id else 'token_budget',
                'prepare_seconds': preparation, 'wall_generation_seconds': elapsed,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'sampled_peak_rss_bytes': peak_rss[0],
                'prepare_transfers': counter_delta(before_prepare, before),
                'generation_transfers': counter_delta(before, after), 'cache_after': after,
                'metrics': generation_metrics(generated['token_ids'], text, row)})
            save()
            print(f"Completed {row['id']}: {len(generated['token_ids'])} tokens in {elapsed:.2f}s", flush=True)
        report['completed'] = True
        save()
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise
    finally:
        cache.close()


if __name__ == '__main__':
    main()
