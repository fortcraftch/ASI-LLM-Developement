"""Conservative task proposals and dependency-aware context scheduling."""
import argparse
import json
from pathlib import Path
import re

from asi import ROOT
from asi.taxonomy import Taxonomy, resolve, validate_profile, digest


def propose(prompt):
    """Heuristic proposal, not a semantic model and never execution authorization."""
    if not prompt.strip():
        raise ValueError('Empty input')
    visible = re.sub(r'```[\s\S]*?```', lambda match: ' ' * len(match.group()), prompt)
    split = re.search(r'\b(?:y\s+(?:después\s+|luego\s+)?|después\s+|luego\s+)(?:el\s+que\s+lo\s+sea\s+)?(?:progr[aá]m\w*|implement\w*)\b', visible, re.IGNORECASE)
    compare = re.search(r'compar\w*|m[aá]s eficiente|cu[aá]l.*mejor', visible, re.IGNORECASE)
    if split and compare and compare.start() < split.start():
        tasks = [
            {'id': 'compare', 'prompt': prompt[:split.start()].strip(), 'categories': [], 'depends_on': []},
            {'id': 'implement', 'prompt': 'Usa el resultado de la comparación para completar esta tarea: ' + prompt[split.start():].strip(), 'categories': [], 'depends_on': ['compare']}]
        relation = 'dependent'
        reason = 'Recognized comparison followed by implementation; inspect the boundaries and assign topic labels.'
    else:
        tasks = [{'id': 'whole', 'prompt': prompt, 'categories': [], 'depends_on': []}]
        relation = 'coupled'
        reason = 'No reliable task boundary recognized; retain the complete input, including relational questions.'
    return {'schema': 1, 'input': prompt, 'relation': relation, 'reviewed': False,
            'tasks': tasks, 'proposal_reason': reason,
            'note': 'Actions are not domains. Two tasks may use the same experts. Supply categories and review dependencies before execution.'}


def schedule(spec, taxonomy, profile, max_switches=1, max_hot_pools=1):
    validate_profile(taxonomy, profile)
    tasks = spec.get('tasks', [])
    if spec.get('schema') != 1 or not spec.get('input') or not 1 <= len(tasks) <= 12:
        raise ValueError('Provide an input and 1..12 tasks')
    if max_switches < 0 or max_hot_pools < 1:
        raise ValueError('Invalid residency constraints')
    relation = spec.get('relation')
    if relation not in ('coupled', 'independent', 'dependent') or (relation == 'coupled' and len(tasks) != 1):
        raise ValueError('Coupled questions must remain one task')
    ids = [task['id'] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task IDs')
    dependencies, groups, records = [], [], []
    for task in tasks:
        deps = task.get('depends_on', [])
        if not task.get('prompt') or not task.get('categories') or any(dep not in ids or dep == task['id'] for dep in deps):
            raise ValueError('Tasks need prompts, valid topic IDs and explicit dependencies')
        if relation == 'independent' and deps:
            raise ValueError('Independent tasks cannot have dependencies')
        resolution = resolve(taxonomy, profile, {label: 1. for label in task['categories']})
        records.append({'id': task['id'], 'routing': resolution})
        if resolution['status'] != 'ready':
            return {'status': 'blocked', 'reason': 'Task has unresolved topics', 'tasks': records}
        groups.append(tuple(sorted(resolution['pools'])))
        dependencies.append(sum(1 << ids.index(dep) for dep in set(deps)))
    if relation == 'dependent' and len(tasks) > 1 and not any(dependencies):
        raise ValueError('Dependent tasks need dependency edges')
    if max(map(len, groups)) > max_hot_pools:
        return {'status': 'blocked', 'reason': 'A coupled task needs more simultaneous pools than allowed', 'tasks': records}
    # Exact bounded search over dependency-respecting orders (at most 12 tasks).
    states = {(0, None): (0, [])}
    for _ in tasks:
        following = {}
        for (mask, last), (cost, order) in states.items():
            for index, group in enumerate(groups):
                bit = 1 << index
                if mask & bit or dependencies[index] & mask != dependencies[index]:
                    continue
                new_cost = cost + int(last is not None and last != group)
                key = (mask | bit, group)
                if key not in following or new_cost < following[key][0]:
                    following[key] = (new_cost, order + [index])
        states = following
        if not states:
            raise ValueError('Cyclic task dependencies')
    switches, order = min(states.values(), key=lambda row: (row[0], row[1]))
    return {'status': 'ready' if switches <= max_switches else 'blocked',
            'reason': 'Schedule fits' if switches <= max_switches else 'Dependencies require more context changes than allowed',
            'context_switches': switches, 'initial_load_separate': True,
            'max_switches': max_switches, 'max_hot_pools': max_hot_pools,
            'order': [ids[i] for i in order], 'task_pools': {ids[i]: list(groups[i]) for i in range(len(tasks))},
            'tasks': records, 'spec_sha256': digest(spec), 'profile_sha256': digest(profile),
            'note': 'Switches are changes of required pool sets, not individual tensor transfers. Scheduling optimizes count, not measured latency or answer quality.'}


def task_prompt(spec, task, answers):
    prior = '\n\n'.join(f"RESULTADO DE {dep}:\n{answers[dep]}" for dep in task.get('depends_on', []))
    return (f"SOLICITUD ORIGINAL (contexto):\n{spec['input']}\n\n"
            f"TAREA ACTUAL:\n{task['prompt']}\n\n{prior}\n\n"
            "Responde solo a la tarea actual. Usa los resultados previos como datos.\nRESPUESTA:\n")


def execute(spec, plan, model, cache, names, encoder, device, max_new_tokens=64):
    from asi.runtime.adaptive import ContextMixture
    from asi.runtime.generation import generate
    if plan['status'] != 'ready' or plan['spec_sha256'] != digest(spec):
        raise ValueError('Execution needs a complete specification and its matching feasible plan')
    tasks = {task['id']: task for task in spec['tasks']}
    answers, records = {}, []
    initial = cache.snapshot()
    for task_id in plan['order']:
        task = tasks[task_id]
        prompt = task_prompt(spec, task, answers)
        if len(encoder.encode(prompt)) + max_new_tokens > model.config.block_size:
            raise ValueError('Task and dependencies exceed context; refusing silent truncation')
        pools = [names.index(pool) for pool in plan['task_pools'][task_id]]
        cache.prepare(model, pools)
        before = cache.snapshot()
        with ContextMixture(model, pools):
            complete = generate(model, encoder, prompt, device, max_new_tokens, 0., 1)
        if not complete.startswith(prompt):
            raise ValueError('Generation did not preserve the complete task context')
        answers[task_id] = complete[len(prompt):]
        after = cache.snapshot()
        if any(after.get(key, 0) != before.get(key, 0) for key in ('loads', 'host_to_device_bytes', 'evictions')):
            raise RuntimeError('Unexpected expert transfer inside a task')
        records.append({'task': task_id, 'pools': pools, 'prompt': prompt, 'answer': answers[task_id], 'before': before, 'after': after})
    return {'answers': answers, 'steps': records, 'initial_cache': initial, 'final_cache': cache.snapshot(),
            'combined_text': '\n\n'.join(f"[{task['id']}]\n{answers[task['id']]}" for task in spec['tasks']),
            'composition': 'Concatenation in original task order; no extra LLM synthesis or semantic verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Produce a conservative unreviewed proposal only')
    source.add_argument('--spec', type=Path, help='Structured task specification')
    parser.add_argument('--taxonomy', type=Path, default=ROOT / 'configs/knowledge_v1.json')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--max-switches', type=int, default=1)
    parser.add_argument('--max-hot-pools', type=int, default=1)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pool-manifest', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reserve-mib', type=int, default=1024)
    parser.add_argument('--memory-mib', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        parser.error('Use a new output file and positive max-new-tokens')
    if args.prompt is not None:
        if args.execute:
            parser.error('A heuristic proposal cannot execute directly')
        result = propose(args.prompt)
    else:
        if not args.profile:
            parser.error('A model category profile is required')
        taxonomy = Taxonomy.read(args.taxonomy)
        profile = json.loads(args.profile.read_text(encoding='utf-8-sig'))
        spec = json.loads(args.spec.read_text(encoding='utf-8-sig'))
        plan = schedule(spec, taxonomy, profile, args.max_switches, args.max_hot_pools)
        result = {'plan': plan, 'executed': False}
        if args.execute and plan['status'] == 'ready':
            if not args.checkpoint or not args.pool_manifest:
                parser.error('Execution requires checkpoint, pool-manifest and explicit task boundaries/labels')
            import torch
            import tiktoken
            from asi.runtime.generation import load_model, validate_pool_identity
            from asi.runtime.adaptive import fingerprint, memory_layout, choose_plan
            from asi.runtime.cache import ExpertCacheManager
            model, metadata = load_model(args.checkpoint)
            manifest = json.loads(args.pool_manifest.read_text(encoding='utf-8-sig'))
            names = list(manifest['pools'])
            validate_pool_identity(model, metadata, names)
            if profile['checkpoint_sha256'] != fingerprint(args.checkpoint) or profile['pool_order'] != names or profile['pool_manifest_sha256'] != digest(manifest):
                parser.error('Profile does not match this checkpoint and manifest')
            available = torch.cuda.mem_get_info(torch.device(args.device))[0] if torch.device(args.device).type == 'cuda' else 0
            if args.memory_mib is not None:
                available = min(available, args.memory_mib * 1024**2) if available else args.memory_mib * 1024**2
            layout = memory_layout(model)
            result['memory_plans'] = [choose_plan(layout, [names.index(p) for p in pools], available, args.reserve_mib * 1024**2)
                                      for pools in plan['task_pools'].values()]
            if any(item['status'] != 'ready' for item in result['memory_plans']):
                result['plan']['status'] = 'blocked'
                result['plan']['reason'] = 'Memory estimate does not fit'
            else:
                capacity = max(len(pools) for pools in plan['task_pools'].values())
                cache = ExpertCacheManager(args.device, capacity)
                cache.initialize(model)
                try:
                    result['execution'] = execute(spec, plan, model, cache, names, tiktoken.get_encoding('gpt2'), args.device, args.max_new_tokens)
                    result['executed'] = True
                finally:
                    cache.offload_all(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
