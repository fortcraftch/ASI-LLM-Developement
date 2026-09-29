"""Shared topic vocabulary and explicit model-specific availability contracts."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Taxonomy:
    def __init__(self, payload):
        if payload.get('schema') != 1 or not payload.get('id') or not payload.get('version'):
            raise ValueError('Taxonomy needs schema, stable id and version')
        self.payload = payload
        self.nodes = {}
        for node in payload['nodes']:
            if node['id'] in self.nodes or not node.get('name'):
                raise ValueError('Duplicate category ID or missing name')
            self.nodes[node['id']] = node
        if 'knowledge' not in self.nodes or self.nodes['knowledge']['parent'] is not None:
            raise ValueError('Expected knowledge root')
        for key, node in self.nodes.items():
            if key != 'knowledge' and node.get('parent') not in self.nodes:
                raise ValueError('Unknown parent: ' + key)
            seen = set()
            current = key
            while current is not None:
                if current in seen:
                    raise ValueError('Taxonomy cycle')
                seen.add(current)
                current = self.nodes[current]['parent']
        if 'unknown' not in self.nodes:
            raise ValueError('An explicit unknown category is required')
        self.identity = {'id': payload['id'], 'version': payload['version'], 'sha256': digest(payload)}

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def ancestors(self, category):
        if category not in self.nodes:
            raise ValueError('Unknown category ID: ' + category)
        result = []
        while category is not None:
            result.append(category)
            category = self.nodes[category]['parent']
        return result

    def at_depth(self, category, depth):
        chain = list(reversed(self.ancestors(category)))
        return chain[min(depth, len(chain) - 1)]

    @property
    def labels(self):
        return [key for key in self.nodes if key != 'knowledge']


def build_profile(taxonomy, manifest, model_id, checkpoint_sha256=None):
    bindings = {}
    for pool, info in manifest['pools'].items():
        domains = info.get('broad_domains')
        if not domains:
            raise ValueError('Explicit broad_domains required; do not infer arbitrary expert capabilities')
        for domain in domains:
            if domain not in taxonomy.nodes or domain in ('knowledge', 'unknown'):
                raise ValueError('Unknown or non-routable declared domain: ' + domain)
            if domain in bindings:
                raise ValueError('Overlapping model domains require an explicit resolution policy')
            bindings[domain] = {'pools': [pool], 'evidence': 'training_assignment_not_quality_validation'}
        for category in info.get('categories', []):
            if category not in taxonomy.nodes or not any(d in taxonomy.ancestors(category) for d in domains):
                raise ValueError('Dataset category is absent or outside the declared domain: ' + category)
    return {'schema': 1, 'model_id': model_id, 'taxonomy': taxonomy.identity,
            'checkpoint_sha256': checkpoint_sha256, 'pool_manifest_sha256': digest(manifest),
            'pool_order': list(manifest['pools']), 'bindings': bindings,
            'status': 'checkpoint_bound' if checkpoint_sha256 else 'planned_unbound',
            'note': 'Assignment describes routing availability, not demonstrated knowledge. Hardware residency is a separate decision.'}


def validate_profile(taxonomy, profile):
    if profile.get('schema') != 1 or profile.get('taxonomy') != taxonomy.identity:
        raise ValueError('Taxonomy version/content differs from model profile')
    pools = profile['pool_order']
    if not pools or len(set(pools)) != len(pools):
        raise ValueError('Model pools must be unique and nonempty')
    for node, binding in profile['bindings'].items():
        if node not in taxonomy.nodes or node in ('knowledge', 'unknown'):
            raise ValueError('Invalid model category binding')
        if not binding.get('pools') or len(set(binding['pools'])) != len(binding['pools']) or any(p not in pools for p in binding['pools']):
            raise ValueError('Invalid pools in binding')
        if not binding.get('evidence'):
            raise ValueError('Model bindings must state their evidence')


def resolve(taxonomy, profile, scores, threshold=.5, max_depth=None):
    """Resolve semantic categories first; do not silently replace missing topics."""
    validate_profile(taxonomy, profile)
    if not 0 <= threshold <= 1 or (max_depth is not None and (type(max_depth) is not int or max_depth < 1)):
        raise ValueError('Invalid threshold or depth')
    for key, score in scores.items():
        if key not in taxonomy.nodes or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Scores must use known IDs and finite values in [0,1]')
    selected = {key: value for key, value in scores.items() if value >= threshold and key != 'knowledge'}
    # Prefer a detected child over its redundant ancestor; unrelated topics remain.
    for node in list(selected):
        if any(node in taxonomy.ancestors(child)[1:] for child in selected):
            selected.pop(node, None)
    if max_depth is not None:
        collapsed = {}
        for node, score in selected.items():
            parent = taxonomy.at_depth(node, max_depth)
            collapsed[parent] = max(collapsed.get(parent, 0), score)
        selected = collapsed
    resolved, unsupported, refine = [], [], []
    pools = []
    for node, confidence in sorted(selected.items(), key=lambda row: (-row[1], row[0])):
        binding_id = next((ancestor for ancestor in taxonomy.ancestors(node) if ancestor in profile['bindings']), None)
        if binding_id is not None:
            binding = profile['bindings'][binding_id]
            resolved.append({'category': node, 'score': confidence, 'model_category': binding_id, **binding})
            pools.extend(binding['pools'])
        else:
            descendants = [key for key in profile['bindings'] if node in taxonomy.ancestors(key)[1:]]
            if descendants:
                refine.append({'category': node, 'available_subcategories': descendants})
            else:
                unsupported.append(node)
    status = 'ready' if resolved else 'unsupported'
    if resolved and (unsupported or refine):
        status = 'partial'
    elif refine:
        status = 'needs_refinement'
    if not selected or 'unknown' in unsupported:
        status = 'partial' if resolved else 'uncertain'
    return {'status': status, 'taxonomy': taxonomy.identity, 'model_id': profile['model_id'],
            'detected_categories': selected, 'resolved': resolved, 'unsupported': unsupported,
            'needs_refinement': refine, 'pools': list(dict.fromkeys(pools)),
            'dominant_category': max(selected, key=selected.get) if selected else None,
            'note': 'Scores are classifier outputs, not calibrated probabilities or evidence of expert competence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy', type=Path, default=Path(__file__).resolve().parent.parent / 'configs/knowledge_v1.json')
    parser.add_argument('--pool-manifest', type=Path, help='Create a profile from explicit training assignments')
    parser.add_argument('--model-id', default='domain_124m')
    parser.add_argument('--checkpoint', type=Path, help='Bind profile to this exact local-model checkpoint')
    parser.add_argument('--profile', type=Path, help='Resolve category scores using this profile')
    parser.add_argument('--scores', type=Path, help='JSON object of category ID to score')
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    taxonomy = Taxonomy.read(args.taxonomy)
    if args.pool_manifest:
        manifest = json.loads(args.pool_manifest.read_text(encoding='utf-8-sig'))
        checkpoint_hash = None
        if args.checkpoint:
            from asi.runtime.adaptive import fingerprint
            from asi.runtime.generation import load_model, validate_pool_identity
            model, metadata = load_model(args.checkpoint)
            validate_pool_identity(model, metadata, list(manifest['pools']))
            checkpoint_hash = fingerprint(args.checkpoint)
        result = build_profile(taxonomy, manifest, args.model_id, checkpoint_hash)
    elif args.profile and args.scores:
        result = resolve(taxonomy, json.loads(args.profile.read_text(encoding='utf-8-sig')),
                         json.loads(args.scores.read_text(encoding='utf-8-sig')), args.threshold, args.max_depth)
    elif args.profile or args.scores or args.checkpoint:
        parser.error('Provide both profile and scores, or pool-manifest with optional checkpoint')
    else:
        result = {'taxonomy': taxonomy.identity, 'categories': len(taxonomy.labels),
                  'domains': [n for n in taxonomy.payload['nodes'] if n['parent'] == 'knowledge'],
                  'scope': taxonomy.payload.get('scope')}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            output.write(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
