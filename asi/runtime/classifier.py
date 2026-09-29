"""Train a separate multilabel head on frozen, shared LLM input embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from asi.data.identity import manifest_matches


class IntegratedHead(nn.Module):
    def __init__(self, dimension, labels, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.network = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden),
                                     nn.GELU(), nn.Linear(hidden, labels))

    def forward(self, features):
        return self.network(features)


def embedding_hash(embedding):
    return hashlib.sha256(embedding.weight.detach().cpu().float().numpy().tobytes()).hexdigest()


@torch.no_grad()
def features(embedding, sequences, max_tokens=512):
    if not sequences or any(not row for row in sequences):
        raise ValueError('Nonempty token sequences required')
    result = []
    for row in sequences:
        ids = torch.tensor(row[-max_tokens:], device=embedding.weight.device, dtype=torch.long)
        if ids.min() < 0 or ids.max() >= embedding.weight.shape[0]:
            raise ValueError('Classifier token outside embedding vocabulary')
        result.append(embedding(ids).float().mean(0))
    return torch.stack(result)


@torch.no_grad()
def predict(embedding, head, tokens, names, threshold=0.5):
    scores = head(features(embedding, [tokens]).to(next(head.parameters()).device)).sigmoid()[0]
    chosen = (scores >= threshold).nonzero().flatten().tolist()
    if not chosen:
        chosen = [int(scores.argmax())]
    return [names[i] for i in sorted(chosen, key=lambda i: float(scores[i]), reverse=True)]


def load_head(path, model, names, manifest_hash=None, taxonomy_identity=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != 1 or payload['pool_names'] != names or payload['embedding_sha256'] != embedding_hash(model.embed):
        raise ValueError('Head belongs to a different embedding checkpoint or pool order')
    if manifest_hash is not None and payload.get('pool_manifest_sha256') != manifest_hash:
        raise ValueError('Classifier taxonomy differs from the runtime manifest')
    if payload.get('taxonomy') != taxonomy_identity:
        raise ValueError('Classifier uses a different semantic taxonomy')
    head = IntegratedHead(model.config.n_embd, len(names), payload['hidden'])
    head.load_state_dict(payload['head'])
    head.trained_labels = payload.get('trained_labels', names)
    return head.eval()


def read_examples(path, names, taxonomy=None):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    splits = {'train': [], 'val': []}
    seen = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row['split']
        if split not in splits or not row['labels'] or any(name not in names for name in row['labels']):
            raise ValueError('Examples need train/val split and known nonempty labels')
        tokens = row.get('tokens')
        if tokens is None:
            tokens = enc.encode(row['text'])
        if not tokens:
            raise ValueError('Empty example')
        key = tuple(tokens[-512:])
        if key in seen and seen[key] != split:
            raise ValueError('Duplicate effective input across train and validation')
        seen[key] = split
        labels = set(row['labels'])
        if taxonomy:
            labels = {ancestor for label in labels for ancestor in taxonomy.ancestors(label)}
        target = [float(name in labels) for name in names]
        splits[split].append((tokens, target))
    if any(not rows for rows in splits.values()):
        raise ValueError('Both train and val examples are required')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool-manifest', type=Path, required=True)
    parser.add_argument('--taxonomy', type=Path, help='Train shared semantic labels instead of model pool labels; requires examples')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--examples', type=Path, help='JSONL: text or tokens, labels, split=train|val; supports real multilabel targets')
    source.add_argument('--data-root', type=Path, help='Classified token shards; single-positive labels only')
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--val-per-pool', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='cpu', help='Head device; frozen LLM embeddings stay on CPU')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len, args.val_per_pool, args.hidden) < 1 or args.seq_len > 512:
        parser.error('Positive sizes required; seq-len <= 512')
    if args.output.exists():
        parser.error('Output directory already exists')
    from asi.runtime.generation import load_model, validate_pool_identity
    from asi.experiments.train import PoolData, PoolManifest
    from asi.runtime.adaptive import fingerprint
    model, metadata = load_model(args.checkpoint)
    manifest = PoolManifest(args.pool_manifest)
    names = manifest.names
    validate_pool_identity(model, metadata, names)
    taxonomy = None
    if args.taxonomy:
        from asi.taxonomy import Taxonomy
        if not args.examples:
            parser.error('Semantic taxonomy requires explicitly labeled --examples')
        taxonomy = Taxonomy.read(args.taxonomy)
        names = taxonomy.labels
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    head = IntegratedHead(model.config.n_embd, len(names), args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    if args.examples:
        examples = read_examples(args.examples, names, taxonomy)
        positive_counts = {name: sum(int(row[1][i]) for row in examples['train']) for i, name in enumerate(names)}
        validation = examples['val']
        def batch():
            return [rng.choice(examples['train']) for _ in range(args.batch_size)]
        provenance = {'examples_sha256': fingerprint(args.examples), 'labels': 'provided multilabel targets'}
    else:
        recipe = metadata.get('training_recipe') or {}
        if recipe.get('source_manifest_sha256') and not manifest_matches(args.data_root / 'manifest.json', recipe['source_manifest_sha256']):
            parser.error('Dataset manifest differs from checkpoint training recipe')
        if recipe.get('pool_manifest_sha256') and not manifest_matches(args.pool_manifest, recipe['pool_manifest_sha256']):
            parser.error('Pool assignments differ from checkpoint training recipe')
        train = PoolData(args.data_root, manifest, 'train', 1, args.seq_len, 'random', args.seed)
        val = PoolData(args.data_root, manifest, 'val', 1, args.seq_len, 'random', args.seed + 1)
        if set(train.names()) != set(names) or set(val.names()) != set(names):
            parser.error('All pools need train and val shards')
        def sample(data, name):
            return data.next_batch(name)[0][0].tolist(), [float(n == name) for n in names]
        def batch():
            return [sample(train, rng.choice(names)) for _ in range(args.batch_size)]
        validation = [sample(val, name) for name in names for _ in range(args.val_per_pool)]
        provenance = {'dataset_manifest_sha256': fingerprint(args.data_root / 'manifest.json'),
                      'labels': 'single pool per window; simultaneous labels not trained'}
        positive_counts = None
    head.train()
    for step in range(args.steps):
        rows = batch()
        x = features(model.embed, [r[0] for r in rows]).to(args.device)
        y = torch.tensor([r[1] for r in rows], device=args.device)
        loss = nn.functional.binary_cross_entropy_with_logits(head(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'step={step} classifier_bce={float(loss.detach()):.5f}', flush=True)
    head.eval()
    scores, targets = [], []
    with torch.no_grad():
        for start in range(0, len(validation), args.batch_size):
            rows = validation[start:start + args.batch_size]
            scores.append(head(features(model.embed, [r[0] for r in rows]).to(args.device)).cpu())
            targets.extend(r[1] for r in rows)
    scores = torch.cat(scores)
    targets = torch.tensor(targets)
    predicted = scores.sigmoid() >= .5
    empty = ~predicted.any(1)
    if taxonomy is None:
        predicted[empty, scores[empty].argmax(1)] = True
    truth = targets.bool()
    tp = int((predicted & truth).sum()); fp = int((predicted & ~truth).sum()); fn = int((~predicted & truth).sum())
    report = {'schema': 1, 'pool_names': names, 'validation_examples': len(targets),
              'label_space': 'semantic_taxonomy' if taxonomy else 'model_pools',
              'taxonomy': taxonomy.identity if taxonomy else None,
              'training_positive_examples_by_label': positive_counts,
              'metrics_label_scope': 'includes ancestor labels' if taxonomy else 'model pools',
              'bce': float(nn.functional.binary_cross_entropy_with_logits(scores, targets)),
              'micro_f1': 2 * tp / max(1, 2 * tp + fp + fn),
              'exact_label_match': float((predicted == truth).all(1).float().mean()),
              'head_parameters': sum(p.numel() for p in head.parameters()),
              'head_weight_bytes': sum(p.numel() * p.element_size() for p in head.parameters()),
              'reused_embedding_parameters': model.embed.weight.numel(), 'additional_embedding_parameters': 0,
              'checkpoint_sha256': fingerprint(args.checkpoint), 'provenance': provenance,
              'settings': vars(args), 'note': 'Frozen bag-of-embeddings baseline; no experts executed. Threshold 0.5; argmax fallback only for legacy pool labels. Semantic mode permits abstention. Validation is not a final test.'}
    args.output.mkdir(parents=True)
    torch.save({'schema': 1, 'head': head.cpu().state_dict(), 'hidden': args.hidden, 'pool_names': names,
                'taxonomy': taxonomy.identity if taxonomy else None,
                'trained_labels': [name for name in names if positive_counts is None or positive_counts[name] > 0],
                'pool_manifest_sha256': fingerprint(args.pool_manifest),
                'embedding_sha256': embedding_hash(model.embed), 'report': report}, args.output / 'head.pt')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
