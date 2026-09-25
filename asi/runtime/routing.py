#!/usr/bin/env python3
"""Session-level multi-label domain router for the V3 specialization prototype.

It reuses the same 17-label topic classifier used by the FineWeb-Edu pipeline,
then adds transparent pool-level keyword evidence so fine distinctions such as
Python vs Java can be made even when the broad classifier only says 'computer
science'.
"""
from __future__ import annotations

import re
import math
from collections import Counter
from types import MethodType
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

LABELS = {
    0: "mathematics_statistics",
    1: "computer_science_software_engineering",
    2: "machine_learning_ai",
    3: "physical_sciences",
    4: "life_sciences_biology",
    5: "medicine_health",
    6: "engineering_technology",
    7: "business_economics",
    8: "law_government",
    9: "social_sciences",
    10: "history_geography",
    11: "philosophy_ethics",
    12: "education_pedagogy",
    13: "language_writing",
    14: "arts_humanities",
    15: "environmental_science_energy",
    16: "personal_finance_practical_life",
}

BROAD_TO_POOLS = {
    "mathematics_statistics": ["expert_03_math"],
    "computer_science_software_engineering": ["expert_00_programming", "expert_01_systems"],
    "machine_learning_ai": ["expert_02_ai"],
    "physical_sciences": ["expert_04_physical_sciences"],
    "life_sciences_biology": ["expert_05_life_sciences"],
    "engineering_technology": ["expert_06_engineering"],
    "social_sciences": ["expert_07_humanities_social"],
    "history_geography": ["expert_07_humanities_social"],
    "philosophy_ethics": ["expert_07_humanities_social"],
    "arts_humanities": ["expert_07_humanities_social"],
}

POOL_HINTS = {
    "expert_00_programming": [
        "python", "java", "kotlin", "scala", "c++", "cpp", "c/c++", "javascript",
        "typescript", "node.js", "rust", "golang", "php", "ruby", "lua", "swift",
        "coding", "programming", "source code", "compiler", "interpreter",
    ],
    "expert_01_systems": [
        "linux", "kernel", "unix", "bash", "systemd", "tcp", "udp", "dns", "networking",
        "distributed system", "kubernetes", "docker", "terraform", "database", "sql",
        "postgresql", "mysql", "redis", "cybersecurity", "firewall", "devops",
    ],
    "expert_02_ai": [
        "machine learning", "deep learning", "neural network", "pytorch", "tensorflow",
        "transformer", "llm", "large language model", "gpt", "rag", "retrieval augmented",
        "mixture of experts", "moe", "computer vision", "yolo", "nlp", "reinforcement learning",
    ],
    "expert_03_math": [
        "mathematics", "math", "algebra", "calculus", "derivative", "integral", "matrix",
        "eigenvalue", "probability", "statistics", "bayesian", "optimization", "equation",
        "geometry", "number theory", "gradient descent",
    ],
    "expert_04_physical_sciences": [
        "physics", "chemistry", "quantum", "astronomy", "astrophysics", "galaxy", "planet",
        "mechanics", "electromagnetism", "thermodynamics", "molecule", "semiconductor",
    ],
    "expert_05_life_sciences": [
        "biology", "gene", "genome", "dna", "rna", "protein", "neuroscience", "neuron",
        "ecology", "evolution", "bacteria", "virus", "biotechnology",
    ],
    "expert_06_engineering": [
        "engineering", "electronics", "circuit", "microcontroller", "robotics", "control system",
        "pid controller", "mechanical engineering", "cad", "manufacturing", "3d printing",
        "telecommunications", "solar power", "battery technology", "civil engineering",
    ],
    "expert_07_humanities_social": [
        "history", "historical", "french revolution", "revolution", "geography", "philosophy", "ethics",
        "psychology", "sociology", "literature", "poetry", "painting", "music", "film", "theatre",
        "politics", "election", "education", "linguistics", "translation",
    ],
}


@dataclass
class RouteResult:
    ranked_pools: List[str]
    scores: Dict[str, float]
    broad_scores: Dict[str, float]


class DomainSessionRouter:
    def __init__(
        self,
        pool_names: List[str],
        classifier_model: str = "mdonigian/fineweb-edu-topic-classifier",
        device: str = "cpu",
        max_pools: int = 3,
        threshold: float = 0.20,
        session_inertia: float = 0.15,
        load_classifier: bool = True,
    ):
        if not pool_names or len(set(pool_names)) != len(pool_names):
            raise ValueError("pool_names must be nonempty and unique")
        if not 1 <= max_pools <= len(pool_names):
            raise ValueError("max_pools must be between 1 and the number of pools")
        self.pool_names = pool_names
        self.max_pools = max_pools
        self.threshold = threshold
        self.session_inertia = session_inertia
        self.previous_pools: List[str] = []
        self.device = torch.device(device)
        self.classifier = None
        self.tokenizer = None
        if load_classifier:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print(f"Loading domain classifier: {classifier_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(classifier_model)
            self.tokenizer.truncation_side = "left"  # Retain the latest session context.
            self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model).to(self.device)
            self.classifier.eval()

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @torch.inference_mode()
    def _broad_scores(self, text: str) -> Dict[str, float]:
        if self.classifier is None:
            return {}
        enc = self.tokenizer([text], truncation=True, max_length=512, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.classifier(**enc).logits
        probs = torch.sigmoid(logits)[0].float().cpu().tolist()
        if len(probs) != len(LABELS):
            raise ValueError("Classifier must use the pipeline's 17-label ordering")
        return {LABELS[i]: float(probs[i]) for i in range(len(probs))}

    def route(self, text: str) -> RouteResult:
        t = self._norm(text)
        broad = self._broad_scores(t)
        scores = {pool: 0.0 for pool in self.pool_names}

        for label, value in broad.items():
            for pool in BROAD_TO_POOLS.get(label, []):
                if pool in scores:
                    scores[pool] += 0.70 * value

        for pool in self.pool_names:
            hints = POOL_HINTS.get(pool, [])
            hits = 0.0
            for hint in hints:
                pattern = r"(?<!\w)" + re.escape(hint.lower()) + r"(?!\w)"
                if re.search(pattern, t):
                    hits += 1.0 + 0.15 * hint.count(" ")
            if hints:
                scores[pool] += min(hits / max(1.0, len(hints) * 0.12), 1.0) * 0.80

        for pool in self.previous_pools:
            if pool in scores:
                scores[pool] += self.session_inertia

        ranked = sorted(self.pool_names, key=lambda p: scores[p], reverse=True)
        selected = [p for p in ranked if scores[p] >= self.threshold][: self.max_pools]
        if not selected:
            selected = ranked[:1]

        self.previous_pools = selected
        return RouteResult(selected, {k: scores[k] for k in ranked}, broad)


class ExpertUsagePredictor:
    """Smoothed empirical demand rates, learned only after observing a turn.

    Current labels predict this turn's demand; previous labels predict demand in
    the next turn. Each table counts presence per turn, not raw token frequency.
    Ranking never updates tables, so evaluation can remain frozen.
    """
    def __init__(self, experts, smoothing=2.0):
        self.experts = sorted(set(tuple(key) for key in experts))
        if not self.experts or not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError('Nonempty expert universe and positive smoothing required')
        if any(len(key) != 2 or any(type(value) is not int or value < 0 for value in key) for key in self.experts):
            raise ValueError('Experts must be nonnegative integer (layer, expert) pairs')
        self.smoothing = float(smoothing)
        self.tables = {}
        self.provenance = {}
        self.training_sessions = []

    def observe(self, previous_labels, current_labels, demanded):
        demanded = set(tuple(key) for key in demanded)
        if not demanded.issubset(self.experts):
            raise ValueError('Observed expert outside predictor universe')
        contexts = ['global']
        contexts += ['current:' + label for label in sorted(set(current_labels))]
        contexts += ['previous:' + label for label in sorted(set(previous_labels))]
        for context in contexts:
            count, usage = self.tables.setdefault(context, [0, Counter()])
            self.tables[context][0] = count + 1
            usage.update(demanded)

    def rank(self, current_labels=(), previous_labels=(), mode='learned'):
        if mode not in ('learned', 'popularity'):
            raise ValueError('Unknown predictor mode')
        turns, global_usage = self.tables.get('global', (0, {}))
        if not turns:
            raise ValueError('Predictor has no training observations')
        contexts = []
        if mode == 'learned':
            # Two equally weighted sources when both are available. Unseen labels
            # back off to the global rate through additive smoothing.
            for prefix, labels in [('current:', current_labels), ('previous:', previous_labels)]:
                labels = sorted(set(labels))
                if labels:
                    contexts.append([self.tables.get(prefix + label, (0, {})) for label in labels])
        def score(key):
            prior = global_usage.get(key, 0) / turns
            if not contexts:
                return prior
            return sum(sum((usage.get(key, 0) + self.smoothing * prior) /
                           (count + self.smoothing) for count, usage in group) / len(group)
                       for group in contexts) / len(contexts)
        return sorted(self.experts, key=lambda key: (-score(key), key))

    def to_dict(self):
        return {'schema': 1, 'experts': self.experts, 'smoothing': self.smoothing,
                'provenance': self.provenance, 'training_sessions': self.training_sessions,
                'tables': {name: {'turns': count, 'usage': [[*key, value] for key, value in sorted(usage.items())]}
                           for name, (count, usage) in self.tables.items()}}

    @classmethod
    def from_dict(cls, payload):
        if payload.get('schema') != 1:
            raise ValueError('Unsupported predictor schema')
        predictor = cls(payload['experts'], payload['smoothing'])
        predictor.provenance = payload.get('provenance', {})
        predictor.training_sessions = payload.get('training_sessions', [])
        for name, table in payload['tables'].items():
            count = table['turns']
            usage = Counter({(layer, expert): value for layer, expert, value in table['usage']})
            if not isinstance(count, int) or count < 0 or not set(usage).issubset(predictor.experts):
                raise ValueError('Invalid predictor table')
            if any(not isinstance(value, int) or not 0 <= value <= count for value in usage.values()):
                raise ValueError('Invalid demand frequency')
            predictor.tables[name] = [count, usage]
        return predictor


class FixedExpertRouting:
    """Execute N fixed experts for one class, bypassing E entirely at inference.

    Equal weights sum to route_scale in uniform mode. The optional calibrated
    mode uses a fixed training-only mean total routing mass per layer/class.
    Neither mode computes E(x), top-k, or token-dependent mixture weights.
    """
    def __init__(self, moes, mapping, weight_mode='uniform'):
        if weight_mode not in ('uniform', 'calibrated'):
            raise ValueError('Unknown fixed mixture weight mode')
        self.moes, self.mapping, self.weight_mode = moes, mapping, weight_mode
        self.label = None
        self.originals = []
        self.calls = Counter()
        self.labels = None
        for layer, moe in moes.items():
            classes = mapping['layers'].get(str(layer), {})
            if not classes or (self.labels is not None and set(classes) != self.labels):
                raise ValueError('Fixed classes must exist consistently in every MoE layer')
            self.labels = set(classes)
            for label, ids in classes.items():
                if len(ids) != moe.gate.topk or len(set(ids)) != len(ids):
                    raise ValueError('Every class must contain exactly native N distinct experts')
                if any(type(i) is not int or not 0 <= i < len(moe.experts) for i in ids):
                    raise ValueError('Invalid fixed expert ID')
                if weight_mode == 'calibrated':
                    mass = mapping.get('fixed_mass', {}).get(str(layer), {}).get(label)
                    if not isinstance(mass, (int,float)) or not math.isfinite(mass) or mass <= 0:
                        raise ValueError('Calibrated routing needs a positive training-only mass for every class')

    def select(self, labels):
        labels = list(dict.fromkeys(labels))
        if len(labels) != 1 or labels[0] not in self.labels:
            raise ValueError('N-of-N routing requires exactly one known class; no multilabel union')
        self.label = labels[0]

    def forward(self, layer, gate, x):
        if self.label is None:
            raise ValueError('Select a fixed class before inference')
        ids = self.mapping['layers'][str(layer)][self.label]
        mass = gate.route_scale if self.weight_mode == 'uniform' else self.mapping['fixed_mass'][str(layer)][self.label]
        indices = torch.tensor(ids, device=x.device, dtype=torch.long).expand(x.shape[0], -1)
        weights = torch.full(indices.shape, mass/len(ids), device=x.device, dtype=x.dtype)
        self.calls[layer] += 1
        return weights, indices

    def attach(self):
        if self.originals:
            raise ValueError('Fixed router already attached')
        for layer, moe in self.moes.items():
            gate = moe.gate
            self.originals.append((gate, 'forward' in gate.__dict__, gate.__dict__.get('forward')))
            gate.forward = MethodType(lambda gate, x, *args, lid=layer, **kwargs: self.forward(lid, gate, x), gate)

    def close(self):
        for gate, had_override, original in self.originals:
            if had_override:
                gate.forward = original
            else:
                del gate.forward
        self.originals.clear()
