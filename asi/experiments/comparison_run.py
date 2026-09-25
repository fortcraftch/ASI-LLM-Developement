"""Explicit execution backend for comparison bundles. Never called by prepare/check."""
import hashlib
import json
from pathlib import Path
import time

from asi.experiments.comparison import digest, resolve, verify_bundle, readiness


def process_rss_bytes():
    """Current resident memory, without requiring an optional dependency on Windows."""
    import sys
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if sys.platform != 'win32':
            return None  # Never replace unavailable RSS with zero or tensor bytes.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.WorkingSetSize


def choose_mapping(model, spec, role, job, names, checkpoint_hash, manifest_hash):
    from asi.models.original import native_moes
    moes = native_moes(model)
    if spec['adapter'] == 'domain':
        layers = {str(lid): {name: [i for i, pool in enumerate(moe.expert_pool_ids.tolist()) if pool == p]
                             for p, name in enumerate(names)} for lid, moe in moes.items()}
        mapping = {'layers': layers}
    elif role == 'classified':
        mapping = json.loads(resolve(spec['mapping']).read_text(encoding='utf-8'))
        provenance = mapping.get('provenance', {})
        if provenance.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('Mapping checkpoint hash mismatch')
        if provenance.get('pool_manifest_sha256') != manifest_hash:
            raise ValueError('Mapping pool manifest hash mismatch')
        if provenance.get('architecture_sha256') != digest(resolve(spec['architecture'])):
            raise ValueError('Mapping architecture hash mismatch')
        # Explicit provenance is mandatory for future calibrations.
        if provenance.get('calibration_split') != 'train':
            raise ValueError('Mapping must declare provenance.calibration_split=train; regenerate from train data')
    else:
        mapping = {'layers': {str(lid): {} for lid in moes}}
    if job['policy'] in ('fixed', 'restrict'):
        count = job['candidates']
        if job['label_source'] == 'global':
            # Global ranking must be calibrated explicitly, never inferred from validation.
            layers = mapping.get('global_layers')
            if layers is None:
                return None, 'Mapping lacks train-calibrated global_layers'
        else:
            layers = mapping['layers']
        required = ['global'] if job['label_source'] == 'global' else names
        for lid, moe in moes.items():
            for label in required:
                ids = layers.get(str(lid), {}).get(label, [])
                if len(ids) < count:
                    return None, f'Layer {lid}/{label} has fewer than {count} classified experts'
                if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(moe.experts) for i in ids):
                    raise ValueError('Invalid or duplicate expert identities')
        mapping = {**mapping, 'layers': {str(lid): {label: layers[str(lid)][label][:count] for label in required}
                                        for lid in moes}}
    return mapping, None


def run_job(directory, job_id, output):
    # Import runtime dependencies only following explicit run-job invocation.
    import numpy as np
    import tiktoken
    import torch
    import torch.nn.functional as F
    from asi.models.original import load_original_model, native_moes, IncrementalDecoder, sync, autocast
    from asi.runtime.generation import load_model, validate_pool_identity
    from asi.runtime.routing import DomainSessionRouter
    from asi.runtime.cache import NativeExpertSessionCache, move_backbone, module_memory
    from asi.experiments.cache_study import COUNTERS

    directory, output = Path(directory), Path(output)
    config, lock = verify_bundle(directory)
    ready = readiness(config)
    if not ready['ready_paths']:
        raise ValueError('All three models must be available before execution: ' + '; '.join(ready['blockers']))
    jobs = json.loads((directory/'jobs.json').read_text(encoding='utf-8'))
    matches = [j for j in jobs if j['id'] == job_id]
    if len(matches) != 1:
        raise ValueError('Unknown job ID')
    job = matches[0]
    if output.exists():
        raise FileExistsError('Use a fresh job output directory; existing results are never overwritten')
    spec = config['models'][job['role']]
    manifest_path = resolve(config['pool_manifest'])
    names = list(json.loads(manifest_path.read_text(encoding='utf-8'))['pools'])
    device = config['device']
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('GPU/RAM measurements require CUDA; CPU simulation is not a measured experiment')
    torch.manual_seed(config['seed'] + job['repeat'])
    torch.set_grad_enabled(False)
    checkpoint_hash = digest(resolve(spec['checkpoint']))
    if spec['adapter'] == 'native':
        model, metadata = load_original_model(resolve(spec['checkpoint']), resolve(spec['architecture']))
    else:
        model, metadata = load_model(resolve(spec['checkpoint']))
        validate_pool_identity(model, metadata, names)
        model.set_active_pools(list(range(len(names))))
    moes = native_moes(model)
    sizes = {moe.gate.topk for moe in moes.values()}
    if sizes != {2}:
        raise ValueError('This registered suite targets native top-2 models; prepare a revised protocol for other N')
    reason = None
    if job['capacity'] is not None and any(job['capacity'] > len(moe.experts) for moe in moes.values()):
        reason = 'Capacity exceeds actual number of experts per layer'
    if job['phase'] == 'quality' and job['length'] > model.config.block_size:
        reason = 'Quality context exceeds model block size'
    if job['phase'] == 'quality' and not any(w['length'] == job['length']+1 for w in
            json.loads((directory/'windows.json').read_text(encoding='utf-8'))):
        reason = 'No held-out windows at this length'
    if job['phase'] == 'generation' and config['generation']['max_prompt_tokens'] + job['length'] > model.config.block_size:
        reason = 'Generation context exceeds model block size; no silent truncation of requested budget'
    mapping, mapping_reason = choose_mapping(model, spec, job['role'], job, names,
                                             checkpoint_hash, lock['pool_manifest_sha256'])
    reason = reason or mapping_reason
    output.mkdir(parents=True)
    header = {'job': job, 'checkpoint_sha256': checkpoint_hash, 'metadata': metadata,
              'runner_sha256': digest(__file__),
              'training': spec.get('training'), 'suite_sha256': digest(directory/'manifest.json'),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'architecture_sha256': digest(resolve(spec['architecture'])) if spec['adapter'] == 'native' else digest(Path(__file__).parents[1]/'models/domain.py'),
              'mapping_sha256': digest(resolve(spec['mapping'])) if spec.get('mapping') else None,
              'dtype': 'bfloat16_autocast_float32_weights', 'decoding': 'greedy',
              'kv_protocol': 'serial prefill; reset and re-prefill shared user-only history each turn',
              'status': 'skipped' if reason else 'running', 'reason': reason}
    def save_header():
        (output/'report.json').write_text(json.dumps(header, indent=2), encoding='utf-8')
    save_header()
    if reason:
        return
    encoder = tiktoken.get_encoding('gpt2')
    classifier = DomainSessionRouter(names, device='cpu', max_pools=1, session_inertia=0,
                                     classifier_model=config['classifier']) if job['label_source'] == 'predicted' else None
    cache = None
    decoder = None
    position = 0
    captures = {}
    # Register after cache hooks so these are final, executed expert IDs.
    route_handles = []

    def attach():
        nonlocal cache
        for handle in route_handles:
            handle.remove()
        route_handles.clear()
        if job['policy'] == 'resident':
            model.to(device)
        else:
            move_backbone(model, device)
            cache = NativeExpertSessionCache(model, mapping, device,
                max_hot_experts=job['capacity'] * len(moes), max_experts_per_layer=job['capacity'],
                policy='prefetch' if job['policy'] == 'exact' else job['policy'],
                pin_memory=config['pin_memory'], max_pinned_bytes=config['max_pinned_mib'] * 1024**2)
        for lid, moe in moes.items():
            route_handles.append(moe.gate.register_forward_hook(
                lambda gate, args, out, layer=lid: captures.__setitem__(str(layer), out[1].detach())))

    def snapshot():
        return cache.snapshot() if cache else {}

    def delta(a, b):
        return {k: b.get(k, 0)-a.get(k, 0) for k in COUNTERS}

    def select(label, text):
        start = time.perf_counter()
        if classifier:
            classifier.previous_pools = []
            label = classifier.route(text).ranked_pools[0]
        elif job['label_source'] == 'global':
            label = 'global'
        seconds = time.perf_counter()-start if classifier else 0.
        before = snapshot()
        sync(device); start = time.perf_counter()
        if cache and job['policy'] in ('fixed', 'restrict'):
            cache.set_context_labels([label])
        sync(device)
        return label, seconds, time.perf_counter()-start, delta(before, snapshot())

    def reset():
        nonlocal position, decoder
        position = 0
        decoder = IncrementalDecoder(model) if spec['adapter'] == 'native' else None

    def step(token, phase):
        nonlocal position
        before = snapshot()
        sync(device); start = time.perf_counter()
        with autocast(device):
            if decoder:
                logits = decoder.step(int(token))[0, -1].float()
            else:
                logits = model(torch.tensor([[int(token)]], device=device), start_pos=position)[0][0, -1].float()
        sync(device)
        elapsed = time.perf_counter()-start
        transfers = delta(before, snapshot())
        if job['policy'] == 'fixed' and (transfers['misses'] or transfers['host_to_device_bytes']):
            raise RuntimeError('Fixed N/N performed an unexpected within-turn load')
        event = {'phase': phase, 'position': position, 'seconds': elapsed, **transfers,
                 'routes': {lid: ids.cpu().tolist() for lid, ids in captures.items()}}
        position += 1
        return logits, event

    def memory():
        return {'inventory': cache.inventory() if cache else module_memory(model),
                'process_rss_bytes': process_rss_bytes(),
                'cuda_allocated_bytes': torch.cuda.memory_allocated(device),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(device),
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}

    def emit(record):
        with (output/'records.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')

    attach()
    try:
        # Kernel warmup excluded. Logical KV reset and expert cache rebuilt afterwards.
        select(names[0], PROMPT_WARMUP)
        reset(); step(encoder.eot_token, 'warmup')
        if cache:
            cache.close(); cache = None
            attach()
        torch.cuda.reset_peak_memory_stats(device)
        if job['phase'] == 'quality':
            windows = json.loads((directory/'windows.json').read_text(encoding='utf-8'))
            for window in (w for w in windows if w['length'] == job['length']+1):
                if cache:
                    cache.close(); cache = None; attach()
                arr = np.load(window['shard'], mmap_mode='r')
                raw = arr[window['start']:window['start']+window['length']].copy()
                arr._mmap.close()
                if hashlib.sha256(raw.tobytes()).hexdigest() != window['tokens_sha256']:
                    raise ValueError('Frozen evaluation tokens changed')
                # The classifier sees only the prompt prefix, not future scored tokens.
                # First half is prefill, second half held out identically for every model.
                boundary = max(1, job['length']//2)
                label, classify_s, prepare_s, transfers = select(window['label'], encoder.decode(raw[:boundary].tolist()))
                reset(); losses, correct, predictions, events = [], 0, [], []
                for i, token in enumerate(raw[:-1]):
                    logits, event = step(token, 'prefill' if i < boundary-1 else 'teacher_forced')
                    events.append(event)
                    if i >= boundary-1:
                        target = torch.tensor([int(raw[i+1])], device=device)
                        losses.append(float(F.cross_entropy(logits[None], target)))
                        predictions.append(int(logits.argmax()))
                        correct += predictions[-1] == int(raw[i+1])
                emit({'window': window['id'], 'true_class': window['label'], 'class': label,
                      'tokens': len(losses), 'nll': sum(losses)/len(losses), 'correct_tokens': correct,
                      'argmax': predictions, 'classification_seconds': classify_s, 'prepare_seconds': prepare_s,
                      'prepare_transfers': transfers, 'events': events, 'memory': memory()})
        else:
            sessions = json.loads((directory/'sessions.json').read_text(encoding='utf-8'))
            for session in sessions:
                # Independent session cache. Warm means one unmeasured replay of this
                # same workload, recorded separately; never hidden advance knowledge.
                if cache:
                    cache.close(); cache = None; attach()
                replays = range(2) if job['cache_start'] == 'warm' else range(1)
                for replay in replays:
                    history = ''
                    for turn_id, turn in enumerate(session['turns']):
                        history = (history+'\n'+turn['prompt']).strip()
                        tokens = encoder.encode(history)[-config['generation']['max_prompt_tokens']:]
                        label, classify_s, prepare_s, transfers = select(turn['label'], turn['prompt'])
                        reset(); events, generated = [], []
                        for token in tokens:
                            logits, event = step(token, 'prefill'); events.append(event)
                        for i in range(job['length']):
                            token = int(logits[:encoder.n_vocab].argmax())
                            generated.append(token)
                            if token == encoder.eot_token or i == job['length']-1:
                                break
                            logits, event = step(token, 'decode'); events.append(event)
                        if replay == max(replays):
                            emit({'session': session['id'], 'scenario': session['scenario'], 'turn': turn_id,
                                  'prompt': turn['prompt'], 'class': label, 'true_class': turn['label'],
                                  'input_tokens': tokens, 'generated_tokens': generated, 'text': encoder.decode(generated),
                                  'classification_seconds': classify_s, 'prepare_seconds': prepare_s,
                                  'prepare_transfers': transfers, 'events': events, 'memory': memory()})
        header.update(status='complete', memory=memory())
    except Exception as exc:
        header.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_header()
        for handle in route_handles:
            handle.remove()
        if cache:
            cache.close()


PROMPT_WARMUP = 'Explain a simple example.'
