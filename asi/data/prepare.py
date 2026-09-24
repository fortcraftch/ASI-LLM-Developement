#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, hashlib, json, os, re, time
from collections import Counter, defaultdict
from pathlib import Path
from asi import DATA_ROOT
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 17-label space published by mdonigian/fineweb-edu-topic-classifier.
LABELS = {
    0: "mathematics_statistics", 1: "computer_science_software_engineering",
    2: "machine_learning_ai", 3: "physical_sciences", 4: "life_sciences_biology",
    5: "medicine_health", 6: "engineering_technology", 7: "business_economics",
    8: "law_government", 9: "social_sciences", 10: "history_geography",
    11: "philosophy_ethics", 12: "education_pedagogy", 13: "language_writing",
    14: "arts_humanities", 15: "environmental_science_energy",
    16: "personal_finance_practical_life",
}

# Transparent fine-grained rules. These are deliberately inspectable and easy to replace later.
FINE_RULES = {
    "computer_science_software_engineering": {
        "programming_python": ["python", "cpython", "pypy", "django", "flask", "fastapi", "pandas", "numpy", "scipy", "pytest", "pip install", "pyproject.toml"],
        "programming_java_jvm": ["java", "jdk", "jre", "jvm", "spring boot", "spring framework", "maven", "gradle", "kotlin", "scala", ".jar", "pom.xml"],
        "programming_c_cpp": ["c++", "c/c++", "cpp", "gcc", "g++", "clang", "cmake", "makefile", "valgrind", "cppreference"],
        "programming_javascript_typescript": ["javascript", "typescript", "node.js", "nodejs", "npm", "yarn", "pnpm", "react", "vue", "angular", "next.js", "express.js", ".jsx", ".tsx"],
        "programming_rust_go": ["rust", "cargo", "golang", "go language", "goroutine", "rustc", "go.mod"],
        "programming_php_ruby": ["php", "laravel", "composer", "ruby", "rails", "rubygems"],
        "programming_other": ["lua", "swift", "objective-c", "perl", "haskell", "erlang", "elixir", "fortran", "matlab", "assembly"],
        "software_engineering": ["software engineering", "design pattern", "refactoring", "unit test", "integration test", "code review", "solid principles", "clean architecture", "dependency injection", "git", "github", "gitlab"],
        "algorithms_data_structures": ["algorithm", "data structure", "binary tree", "linked list", "hash table", "graph traversal", "dijkstra", "dynamic programming", "sorting algorithm", "time complexity", "big o notation"],
        "web_development": ["html", "css", "http", "rest api", "graphql", "web development", "frontend", "backend", "full stack", "web server", "browser"],
        "systems_linux_os": ["linux", "kernel", "unix", "system call", "process scheduling", "memory management", "filesystem", "bash", "shell scripting", "systemd"],
        "networking_distributed": ["tcp", "udp", "dns", "networking", "distributed system", "load balancer", "microservices", "grpc", "kafka", "message queue"],
        "databases_data_engineering": ["sql", "postgresql", "mysql", "sqlite", "database", "mongodb", "redis", "database schema", "etl", "data warehouse", "spark", "data pipeline"],
        "devops_cloud": ["docker", "kubernetes", "k8s", "terraform", "ansible", "aws", "azure", "google cloud", "gcp", "devops", "ci/cd", "jenkins"],
        "cybersecurity": ["cybersecurity", "cyber security", "vulnerability", "exploit", "penetration testing", "malware", "ransomware", "firewall", "authentication", "authorization", "tls", "ssl", "cryptography"],
        "graphics_games": ["opengl", "vulkan", "directx", "shader", "gpu programming", "unity", "unreal engine", "game development", "3d rendering", "computer graphics"],
    },
    "machine_learning_ai": {
        "machine_learning": ["machine learning", "classification", "regression", "clustering", "feature engineering", "scikit-learn", "decision tree", "random forest", "gradient boosting"],
        "deep_learning": ["deep learning", "neural network", "pytorch", "tensorflow", "backpropagation", "transformer", "cnn", "convolutional neural", "lstm", "gru"],
        "large_language_models": ["large language model", "llm", "gpt", "chatgpt", "attention mechanism", "tokenizer", "fine-tuning", "rag", "retrieval augmented generation", "mixture of experts", "moe"],
        "computer_vision": ["computer vision", "image classification", "object detection", "yolo", "opencv", "segmentation", "image recognition", "image processing"],
        "reinforcement_learning": ["reinforcement learning", "q-learning", "policy gradient", "reward function", "markov decision process", "actor critic", "ppo"],
        "nlp": ["natural language processing", "nlp", "named entity recognition", "sentiment analysis", "word embedding", "word2vec", "sequence tagging", "text classification"],
        "data_science": ["data science", "data analysis", "jupyter", "dataframe", "exploratory data analysis", "visualization", "data mining"],
        "robotics_ai": ["robotics", "robot", "slam", "path planning", "autonomous", "robot arm"],
    },
    "mathematics_statistics": {
        "algebra": ["linear algebra", "matrix", "matrices", "eigenvalue", "eigenvector", "group theory", "ring theory", "field theory", "polynomial"],
        "calculus": ["calculus", "derivative", "derivatives", "integral", "integrals", "differential equation", "limit", "gradient", "jacobian"],
        "geometry": ["geometry", "geometric", "euclidean", "triangle", "polygon", "topology", "manifold"],
        "number_theory": ["number theory", "prime number", "integer", "modular arithmetic", "diophantine", "congruence"],
        "discrete_math_logic": ["discrete mathematics", "combinatorics", "graph theory", "boolean algebra", "propositional logic", "predicate logic", "set theory"],
        "probability_statistics": ["probability", "statistics", "statistical", "bayesian", "random variable", "distribution", "hypothesis test", "confidence interval", "regression analysis"],
        "optimization": ["optimization", "optimisation", "linear programming", "convex optimization", "gradient descent", "lagrange multiplier"],
        "numerical_methods": ["numerical method", "numerical analysis", "finite difference", "finite element", "numerical integration", "approximation"],
        "applied_mathematics": ["applied mathematics", "mathematical model", "mathematical modeling", "operations research"],
    },
    "physical_sciences": {
        "physics": ["physics", "mechanics", "electromagnetism", "thermodynamics", "classical mechanics", "fluid mechanics", "optics"],
        "chemistry": ["chemistry", "chemical reaction", "molecule", "organic chemistry", "inorganic chemistry", "analytical chemistry", "spectroscopy"],
        "astronomy_space": ["astronomy", "astrophysics", "galaxy", "planet", "cosmology", "black hole", "stellar", "space telescope"],
        "materials_physics": ["materials science", "solid state", "crystal structure", "semiconductor", "nanomaterial", "material properties"],
        "quantum": ["quantum mechanics", "quantum physics", "quantum computing", "quantum state", "wave function", "qubit"],
    },
    "life_sciences_biology": {
        "biology": ["biology", "organism", "cell", "species", "biological"],
        "genetics": ["genetics", "gene", "genome", "dna", "rna", "mutation", "genomic"],
        "molecular_biology": ["molecular biology", "protein", "enzyme", "transcription", "translation"],
        "microbiology": ["microbiology", "bacteria", "bacterial", "virus", "viral", "microorganism"],
        "neuroscience": ["neuroscience", "neuron", "brain", "synapse", "cognitive neuroscience"],
        "ecology": ["ecology", "ecosystem", "biodiversity", "population ecology"],
        "evolution": ["evolution", "natural selection", "phylogenetic", "evolutionary biology"],
        "biotechnology": ["biotechnology", "biotech", "genetic engineering", "crispr", "bioprocess"],
    },
    "medicine_health": {
        "medicine": ["medicine", "medical", "disease", "diagnosis", "patient"],
        "clinical": ["clinical trial", "clinical practice", "treatment", "hospital", "surgery"],
        "pharmacology": ["pharmacology", "drug", "medication", "dosage", "pharmaceutical"],
        "anatomy_physiology": ["anatomy", "physiology", "organ system", "muscle", "hormone"],
        "public_health_epidemiology": ["public health", "epidemiology", "outbreak", "incidence", "prevalence"],
        "nutrition": ["nutrition", "diet", "nutrient", "calorie", "vitamin"],
        "mental_health": ["mental health", "psychiatry", "psychological disorder", "depression", "anxiety"],
    },
    "engineering_technology": {
        "electrical_electronics": ["electrical engineering", "electronics", "circuit", "resistor", "capacitor", "microcontroller"],
        "telecommunications": ["telecommunication", "5g", "4g", "radio frequency", "fiber optic", "wireless communication"],
        "mechanical": ["mechanical engineering", "machine design", "cad"],
        "civil": ["civil engineering", "structural engineering", "concrete", "bridge", "construction"],
        "chemical": ["chemical engineering", "process engineering", "reaction engineering", "chemical process"],
        "aerospace": ["aerospace", "aerodynamics", "aircraft", "rocket", "aviation"],
        "robotics_control": ["robotics", "control system", "pid controller", "automation", "robot arm"],
        "manufacturing": ["manufacturing", "machining", "3d printing", "additive manufacturing", "production line"],
        "materials": ["materials engineering", "composite material", "metallurgy", "ceramic material"],
        "energy": ["energy engineering", "solar power", "photovoltaic", "battery technology", "power system"],
    },
    "business_economics": {
        "economics": ["economics", "macroeconomics", "microeconomics", "inflation", "gdp"],
        "finance": ["finance", "investment", "stock market", "bond", "portfolio", "asset"],
        "accounting": ["accounting", "balance sheet", "income statement", "audit", "bookkeeping"],
        "business_management": ["business management", "management", "strategy", "organization", "leadership"],
        "marketing": ["marketing", "advertising", "brand", "customer acquisition", "seo"],
        "entrepreneurship": ["entrepreneurship", "startup", "founder", "venture capital", "business model"],
        "ecommerce": ["e-commerce", "ecommerce", "online store", "shopify", "woocommerce"],
    },
    "law_government": {
        "law": ["law", "legal", "court", "lawsuit", "contract law"],
        "constitutional_law": ["constitution", "constitutional law", "supreme court"],
        "international_law": ["international law", "treaty", "diplomatic law"],
        "regulation_policy": ["regulation", "regulatory", "policy", "legislation"],
        "government": ["government", "parliament", "congress", "public administration"],
    },
    "social_sciences": {
        "psychology": ["psychology", "psychological", "behavioral", "cognition"],
        "sociology": ["sociology", "social class", "social structure", "society"],
        "anthropology": ["anthropology", "ethnography", "culture", "tribe"],
        "political_science": ["political science", "election", "politics", "political party"],
        "communication": ["communication studies", "journalism", "media studies", "mass communication"],
        "demography": ["demography", "population growth", "census", "migration"],
    },
    "history_geography": {
        "history": ["history", "historical", "ancient", "medieval", "empire", "war history"],
        "archaeology": ["archaeology", "archaeological", "artifact", "excavation"],
        "geography": ["geography", "geographic", "cartography", "region", "territory"],
        "cultural_history": ["cultural history", "historical culture", "heritage"],
    },
    "philosophy_ethics": {
        "philosophy": ["philosophy", "philosophical", "metaphysics", "ontology"],
        "ethics": ["ethics", "ethical", "morality", "moral philosophy"],
        "epistemology_logic": ["epistemology", "knowledge theory", "logic", "rationalism", "empiricism"],
        "religion_theology": ["religion", "theology", "theological", "christianity", "islam", "buddhism"],
    },
    "education_pedagogy": {
        "education": ["education", "educational", "school system", "student learning"],
        "pedagogy": ["pedagogy", "learning theory", "instructional"],
        "teaching": ["teaching", "teacher", "lesson plan", "classroom"],
        "curriculum": ["curriculum", "syllabus", "assessment", "educational standards"],
        "academic_reference": ["textbook", "lecture notes", "study guide", "academic reference"],
    },
    "language_writing": {
        "linguistics": ["linguistics", "linguistic", "phonology", "morphology", "syntax", "semantics"],
        "english_language": ["english grammar", "english language", "grammar rules", "vocabulary"],
        "language_learning": ["language learning", "learn english", "second language", "foreign language"],
        "writing_communication": ["writing", "essay writing", "technical writing", "copywriting"],
        "translation": ["translation", "translate", "localization", "translator"],
    },
    "arts_humanities": {
        "literature": ["literature", "novel", "poetry", "poem", "literary"],
        "visual_arts": ["painting", "drawing", "sculpture", "visual art", "art history"],
        "music": ["music", "musical", "songwriting", "composition", "harmony"],
        "film": ["film", "cinema", "movie", "filmmaking", "screenplay"],
        "theatre": ["theatre", "theater", "stage production", "drama"],
        "design": ["design", "graphic design", "user experience", "ux design", "industrial design"],
        "architecture": ["architecture", "architectural design", "building design"],
    },
    "environmental_science_energy": {
        "environmental_science": ["environmental science", "pollution", "ecosystem services"],
        "climate": ["climate change", "climate science", "global warming", "greenhouse gas"],
        "ecology_environment": ["conservation", "habitat", "biodiversity", "environmental impact"],
        "energy": ["renewable energy", "solar energy", "wind energy", "energy transition"],
        "sustainability": ["sustainability", "sustainable development", "circular economy"],
        "geoscience": ["geology", "soil science", "earth science", "hydrology"],
    },
    "personal_finance_practical_life": {
        "personal_finance": ["personal finance", "budgeting", "mortgage", "credit card", "retirement planning"],
        "consumer": ["consumer", "product review", "shopping", "price comparison"],
        "cooking_food": ["recipe", "cooking", "food", "baking", "ingredient"],
        "health_fitness": ["fitness", "workout", "exercise", "bodybuilding"],
        "home_garden": ["home improvement", "gardening", "garden", "plumbing", "house repair"],
        "automotive": ["car", "automotive", "vehicle", "engine repair", "motorcycle"],
        "travel": ["travel", "hotel", "tourism", "vacation", "itinerary"],
        "sports_hobbies": ["sports", "football", "basketball", "cycling", "hobby", "photography hobby"],
    },
}

URL_BOOSTS = {
    "programming_python": ["python.org", "pypi.org", "pytorch.org", "numpy.org", "pandas.pydata.org"],
    "programming_java_jvm": ["spring.io", "kotlinlang.org"],
    "programming_javascript_typescript": ["developer.mozilla.org", "nodejs.org", "npmjs.com", "react.dev"],
    "programming_c_cpp": ["cppreference.com", "isocpp.org"],
    "large_language_models": ["huggingface.co", "openai.com", "anthropic.com"],
    "machine_learning": ["scikit-learn.org"],
}


def slug(x: str) -> str:
    x = re.sub(r"[^a-z0-9_]+", "_", x.lower().strip())
    return re.sub(r"_+", "_", x).strip("_") or "unknown"


def stable_split(doc_id: str, val_pct: float) -> str:
    h = hashlib.sha1(doc_id.encode()).hexdigest()
    return "val" if int(h[:8], 16) / 0xFFFFFFFF < val_pct else "train"


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_fine(text: str, url: str, broad: str) -> Tuple[str, float, Dict[str, float]]:
    rules = FINE_RULES.get(broad)
    if not rules:
        return "general", 0.0, {}
    t = normalized(text)
    u = (url or "").lower()
    scores = defaultdict(float)
    for sub, phrases in rules.items():
        for p in phrases:
            c = min(t.count(p), 5)
            if c:
                scores[sub] += c * (1.0 + 0.25 * min(p.count(" "), 3))
    for sub, hints in URL_BOOSTS.items():
        if sub in scores:
            for hint in hints:
                if hint in u:
                    scores[sub] += 4.0
    if broad == "computer_science_software_engineering":
        if any(x in u for x in ["python", "pypi.org"]): scores["programming_python"] += 5
        if any(x in u for x in ["java", "maven", "spring"]): scores["programming_java_jvm"] += 5
        if any(x in u for x in ["javascript", "node", "npm", "typescript"]): scores["programming_javascript_typescript"] += 5
        if any(x in u for x in ["cplusplus", "cppreference"]): scores["programming_c_cpp"] += 5
    if not scores:
        return "general", 0.0, {}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ordered[0]
    total = sum(v for _, v in ordered)
    conf = best_score / total if total else 0.0
    if best_score < 1.5:
        return "general", conf, dict(ordered)
    return best, conf, dict(ordered)


class TopicClassifier:
    def __init__(self, model_name: str, device: str, threshold: float):
        self.device = torch.device(device)
        self.threshold = threshold
        print(f"Loading classifier: {model_name}")
        print(f"Classifier device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.model.config.num_labels != 17:
            raise RuntimeError(f"Expected 17 labels, got {self.model.config.num_labels}")
        self.bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    @torch.inference_mode()
    def classify(self, texts: List[str]) -> List[dict]:
        enc = self.tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if self.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(**enc).logits
        else:
            logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).float().cpu().numpy()
        out = []
        for row in probs:
            scores = {LABELS[i]: float(row[i]) for i in range(17)}
            selected = [k for k, v in scores.items() if v >= self.threshold]
            if not selected:
                selected = [max(scores, key=scores.get)]
            selected.sort(key=lambda k: scores[k], reverse=True)
            out.append({
                "primary": selected[0],
                "secondary": selected[1:],
                "scores": scores,
                "confidence": scores[selected[0]],
            })
        return out


class TokenShardWriter:
    """Append-only, resumable uint16 shard writer.

    Why this does NOT use ``open_memmap(..., shape=(shard_tokens,))`` for the
    active shard: NumPy must create a file with the complete requested shape.
    With many domain/category writers, a 100M-token shard would therefore
    reserve about 191 MiB *per active writer* before those tokens exist.

    Active data is instead appended to ``*.part.bin``. The temporary file grows
    only with the tokens actually written. Once it reaches ``shard_tokens`` (or
    when the full run finishes), it is converted chunk-by-chunk to a standard
    ``.npy`` file, so downstream code can continue using ``np.load`` unchanged.
    """

    RAW_DTYPE = np.dtype("<u2")  # stable two-byte little-endian uint16

    def __init__(self, directory: Path, split: str, shard_tokens: int, state: dict | None = None):
        self.directory = directory
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.directory.mkdir(parents=True, exist_ok=True)

        state = state or {}
        self.index = int(state.get("index", 0))
        self.pos = int(state.get("pos", 0))
        self.tokens = int(state.get("tokens", 0))
        self.docs = int(state.get("docs", 0))
        self.shards = int(state.get("shards", 0))
        self.fh = None

        if self.pos:
            self._validate_partial()

    @property
    def part_path(self):
        return self.directory / f"{self.split}_{self.index:06d}.part.bin"

    @property
    def legacy_part_path(self):
        return self.directory / f"{self.split}_{self.index:06d}.part.npy"

    @property
    def final_path(self):
        return self.directory / f"{self.split}_{self.index:06d}.npy"

    def _validate_partial(self):
        """Validate that a resumable raw partial contains exactly ``pos`` tokens."""
        if not self.part_path.exists():
            raise RuntimeError(
                f"Resume state expects {self.part_path}, but it does not exist. "
                "If this is output from the previous memmap version, run with "
                "--resume: legacy .part.npy files are migrated before writers open."
            )
        expected = self.pos * self.RAW_DTYPE.itemsize
        actual = self.part_path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"Corrupt/inconsistent partial shard {self.part_path}: "
                f"expected {expected:,} bytes for {self.pos:,} tokens, got {actual:,}."
            )

    def _ensure_open(self):
        if self.fh is None:
            # Buffered append. There is no preallocation: file size == written payload.
            self.fh = open(self.part_path, "ab", buffering=1024 * 1024)

    def _sync(self):
        if self.fh is not None:
            self.fh.flush()
            os.fsync(self.fh.fileno())

    def _close_handle(self):
        if self.fh is not None:
            self._sync()
            self.fh.close()
            self.fh = None

    def add(self, ids: List[int]):
        arr = np.asarray(ids, dtype=self.RAW_DTYPE)
        offset = 0

        while offset < len(arr):
            remaining = self.shard_tokens - self.pos
            take = min(remaining, len(arr) - offset)
            if take <= 0:
                self._finalize_current()
                continue

            self._ensure_open()
            # tobytes creates only a document/chunk-sized temporary, never a shard-sized one.
            self.fh.write(arr[offset:offset + take].tobytes(order="C"))
            self.pos += take
            self.tokens += take
            offset += take

            if self.pos == self.shard_tokens:
                self._finalize_current()

        self.docs += 1

    def flush(self):
        self._sync()

    def _raw_to_npy(self, count: int):
        """Convert current raw partial to .npy in bounded memory."""
        if count <= 0:
            return

        expected_bytes = count * self.RAW_DTYPE.itemsize
        actual_bytes = self.part_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Cannot finalize {self.part_path}: expected {expected_bytes:,} bytes, "
                f"found {actual_bytes:,}."
            )

        tmp = self.directory / f".{self.split}_{self.index:06d}.npy.tmp"
        tmp.unlink(missing_ok=True)

        src = np.memmap(self.part_path, mode="r", dtype=self.RAW_DTYPE, shape=(count,))
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(count,))

        # ~2 MiB source slices. Conversion never loads the complete shard into RAM.
        chunk_tokens = 1_000_000
        for begin in range(0, count, chunk_tokens):
            end = min(begin + chunk_tokens, count)
            dst[begin:end] = src[begin:end]

        # Windows does not allow replace/unlink while a NumPy memmap still owns
        # an open file mapping. Close both mappings explicitly before touching
        # either path. Relying on ``del`` alone is not deterministic enough.
        dst.flush()
        dst_mm = getattr(dst, "_mmap", None)
        src_mm = getattr(src, "_mmap", None)
        if dst_mm is not None:
            dst_mm.close()
        if src_mm is not None:
            src_mm.close()
        del dst, src
        gc.collect()

        # Replace only after a successful complete conversion.
        os.replace(tmp, self.final_path)
        unlink_with_retry(self.part_path)

    def _finalize_current(self):
        if self.pos == 0:
            return

        count = self.pos
        self._close_handle()
        self._raw_to_npy(count)

        self.index += 1
        self.shards += 1
        self.pos = 0

    def close(self):
        # End-of-run partial shards become normal compact .npy files.
        self._finalize_current()

    def state_dict(self):
        # Checkpoint only after bytes are durably flushed; then progress.json can
        # safely claim that these ``pos`` tokens exist on disk.
        self._sync()
        return {
            "index": self.index,
            "pos": self.pos,
            "tokens": self.tokens,
            "docs": self.docs,
            "shards": self.shards,
            "format": "append_bin_v3",
        }


def unlink_with_retry(path: Path, attempts: int = 12, delay: float = 0.15):
    """Delete a path robustly, including transient Windows file locks.

    Windows raises WinError 32 if a file is still mapped/open. NumPy mappings
    are explicitly closed before this helper is called, but antivirus/indexing
    software can also hold a file very briefly, so retry a few times.
    """
    path = Path(path)
    if not path.exists():
        return

    last_error = None
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise last_error


def close_memmap(mapping, flush: bool = False):
    """Explicitly close a NumPy memmap and release its Windows file handle."""
    if mapping is None:
        return
    if flush:
        mapping.flush()
    mm = getattr(mapping, "_mmap", None)
    if mm is not None:
        mm.close()


def migrate_legacy_memmap_partials(output: Path, writer_states: dict):
    """One-time migration from the previous preallocated *.part.npy format.

    Only the valid prefix indicated by progress.json is copied. Therefore a
    legacy 191 MiB preallocated file that contains, say, 20k real tokens becomes
    a ~40 KiB ``*.part.bin`` file. The old file is deleted only after the new
    raw partial has been fully written, flushed and atomically renamed.
    """
    migrated_files = 0
    migrated_tokens = 0
    reclaimed_logical_bytes = 0

    for key, state in writer_states.items():
        if "|" not in key:
            continue
        split, category = key.split("|", 1)
        pos = int(state.get("pos", 0))
        index = int(state.get("index", 0))
        if pos <= 0:
            continue

        directory = output / category
        legacy = directory / f"{split}_{index:06d}.part.npy"
        raw = directory / f"{split}_{index:06d}.part.bin"
        expected_raw_bytes = pos * TokenShardWriter.RAW_DTYPE.itemsize

        if raw.exists():
            if raw.stat().st_size != expected_raw_bytes:
                raise RuntimeError(
                    f"Existing migrated partial {raw} has {raw.stat().st_size:,} bytes; "
                    f"expected {expected_raw_bytes:,}."
                )
            # A previous migration may have completed the new file but crashed
            # before deleting the legacy one. In that case the raw file wins.
            if legacy.exists():
                reclaimed_logical_bytes += legacy.stat().st_size
                unlink_with_retry(legacy)
            continue

        if not legacy.exists():
            raise RuntimeError(
                f"Resume state for {key} expects {pos:,} partial tokens, but neither "
                f"{raw.name} nor legacy {legacy.name} exists."
            )

        old_size = legacy.stat().st_size
        src = np.load(legacy, mmap_mode="r")
        if src.dtype != np.uint16 or src.ndim != 1 or len(src) < pos:
            close_memmap(src)
            del src
            raise RuntimeError(f"Incompatible legacy partial shard: {legacy}")

        tmp = raw.with_name(raw.name + ".migrating")
        tmp.unlink(missing_ok=True)
        try:
            with open(tmp, "wb", buffering=1024 * 1024) as fh:
                chunk_tokens = 1_000_000
                for begin in range(0, pos, chunk_tokens):
                    end = min(begin + chunk_tokens, pos)
                    # Force a real copy. np.asarray() can return a view that keeps
                    # the source memmap alive, which prevents unlink() on Windows.
                    block = np.array(
                        src[begin:end],
                        dtype=TokenShardWriter.RAW_DTYPE,
                        copy=True,
                    )
                    fh.write(block.tobytes(order="C"))
                    del block
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            # Critical on Windows: release the mmap handle before deleting the
            # legacy .part.npy. ``del src`` by itself was the source of WinError 32.
            close_memmap(src)
            del src
            gc.collect()

        if tmp.stat().st_size != expected_raw_bytes:
            raise RuntimeError(f"Migration size mismatch for {legacy}")
        os.replace(tmp, raw)
        unlink_with_retry(legacy)

        migrated_files += 1
        migrated_tokens += pos
        reclaimed_logical_bytes += max(0, old_size - expected_raw_bytes)

    if migrated_files:
        print(
            f"Migrated {migrated_files} legacy preallocated partial shards "
            f"({migrated_tokens:,} valid tokens) to append-only storage; "
            f"released about {reclaimed_logical_bytes / (1024**3):.2f} GiB of logical file size."
        )


def _jsonable_samples(samples):
    return {k: list(v) for k, v in samples.items()}


def _atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--output", type=Path, default=DATA_ROOT)
    p.add_argument("--classifier-model", default="mdonigian/fineweb-edu-topic-classifier")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-docs", type=int, default=10000, help="-1 = complete dataset")
    p.add_argument("--val-pct", type=float, default=0.01)
    p.add_argument("--topic-threshold", type=float, default=0.30)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--min-int-score", type=int, default=None)
    p.add_argument("--sample-limit", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="Resume safely from progress.json")
    p.add_argument("--reset", action="store_true", help="Delete previous output shards/progress before starting")
    p.add_argument("--checkpoint-every", type=int, default=1000, help="Save progress every N source documents")
    p.add_argument("--stats-every", type=int, default=5000, help="Print extended statistics every N source documents")
    args = p.parse_args()
    if args.max_docs == 0:
        raise ValueError("--max-docs cannot be 0")
    if args.resume and args.reset:
        raise ValueError("Use either --resume or --reset, not both")

    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "progress.json"

    if args.reset:
        for pattern in ("*.npy", "*.part.bin", "*.part.npy", "*.migrating", "*.npy.tmp"):
            for q in args.output.rglob(pattern):
                q.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        (args.output / "manifest.json").unlink(missing_ok=True)

    existing_data = any(args.output.rglob("*.npy")) or any(args.output.rglob("*.part.bin")) or progress_path.exists()
    if existing_data and not args.resume and not args.reset:
        raise RuntimeError(
            f"{args.output} already contains data. Use --resume to continue safely or --reset to start over."
        )

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    enc = tiktoken.get_encoding("gpt2")
    clf = TopicClassifier(args.classifier_model, device, args.topic_threshold)

    resume = {}
    if args.resume:
        if not progress_path.exists():
            raise RuntimeError(f"Cannot resume: {progress_path} does not exist")
        resume = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = resume.get("run", {})
        current = {"dataset": args.dataset, "config": args.config, "split": args.split,
                   "shard_tokens": args.shard_tokens, "val_pct": args.val_pct,
                   "topic_threshold": args.topic_threshold, "classifier_model": args.classifier_model}
        for k, v in current.items():
            if k in expected and expected[k] != v:
                raise RuntimeError(f"Resume mismatch for {k}: checkpoint={expected[k]!r}, current={v!r}")

    source_rows_seen = int(resume.get("source_rows_seen", 0))
    stats = resume.get("stats") or {
        "total_documents": 0, "total_tokens": 0, "skipped_documents": 0,
        "filtered_documents": 0, "low_confidence_documents": 0,
        "splits": {"train": {"documents": 0, "tokens": 0}, "val": {"documents": 0, "tokens": 0}},
        "samples": {},
    }
    stats["samples"] = defaultdict(list, stats.get("samples", {}))
    category_docs = Counter(resume.get("category_docs", {})); category_tokens = Counter(resume.get("category_tokens", {}))
    broad_docs = Counter(resume.get("broad_docs", {})); broad_tokens = Counter(resume.get("broad_tokens", {}))
    broad_secondary = Counter(resume.get("broad_secondary", {}))
    writer_states = resume.get("writers", {})
    if args.resume and writer_states:
        migrate_legacy_memmap_partials(args.output, writer_states)
    writers: Dict[Tuple[str, str], TokenShardWriter] = {}

    def writer_key(split, category): return f"{split}|{category}"
    def writer_for(split: str, category: str):
        key = (split, category)
        if key not in writers:
            writers[key] = TokenShardWriter(
                args.output / category, split, args.shard_tokens,
                writer_states.get(writer_key(split, category)),
            )
        return writers[key]

    start = time.time()
    last_checkpoint_seen = source_rows_seen
    last_stats_seen = source_rows_seen

    def save_progress():
        all_writer_states = dict(writer_states)
        for (sp, cat), w in writers.items():
            all_writer_states[writer_key(sp, cat)] = w.state_dict()
        obj = {
            "version": 3,
            "run": {"dataset": args.dataset, "config": args.config, "split": args.split,
                    "shard_tokens": args.shard_tokens, "val_pct": args.val_pct,
                    "topic_threshold": args.topic_threshold, "classifier_model": args.classifier_model},
            "source_rows_seen": source_rows_seen,
            "stats": {**stats, "samples": _jsonable_samples(stats["samples"])},
            "category_docs": dict(category_docs), "category_tokens": dict(category_tokens),
            "broad_docs": dict(broad_docs), "broad_tokens": dict(broad_tokens),
            "broad_secondary": dict(broad_secondary), "writers": all_writer_states,
            "saved_at_unix": time.time(),
        }
        _atomic_json(progress_path, obj)

    def print_stats():
        elapsed = max(time.time() - start, 1e-9)
        session_rows = max(source_rows_seen - int(resume.get("source_rows_seen", 0)), 0)
        rate = session_rows / elapsed
        disk_bytes = _dir_size_bytes(args.output)
        gib = disk_bytes / (1024 ** 3)
        token_payload_bytes = stats["total_tokens"] * 2
        payload_mib = token_payload_bytes / (1024 ** 2)
        print(
            f"\n[progress] source={source_rows_seen:,} accepted={stats['total_documents']:,} "
            f"tokens={stats['total_tokens']:,} categories={len(category_docs)} "
            f"session_rate={rate:.2f} source-doc/s disk={gib:.3f} GiB "
            f"token_payload≈{payload_mib:.1f} MiB "
            f"skipped={stats['skipped_documents']:,} filtered={stats.get('filtered_documents',0):,}"
        )

    def process(rows):
        if not rows:
            return
        texts = [str(r.get("text", "")) for r in rows]
        preds = clf.classify(texts)
        for row, pred in zip(rows, preds):
            text = str(row.get("text", "")).strip()
            if not text:
                stats["skipped_documents"] += 1
                continue
            if args.min_score is not None and row.get("score") is not None and float(row["score"]) < args.min_score:
                stats["filtered_documents"] += 1; continue
            if args.min_int_score is not None and row.get("int_score") is not None and int(row["int_score"]) < args.min_int_score:
                stats["filtered_documents"] += 1; continue
            doc_id = str(row.get("id") or f"{row.get('url','')}:{source_rows_seen}")
            split = stable_split(doc_id, args.val_pct)
            broad = pred["primary"]
            sub, sub_conf, fine_scores = score_fine(text, str(row.get("url", "")), broad)
            category = f"{slug(broad)}__{slug(sub)}"
            ids = enc.encode_ordinary(text); ids.append(enc.eot_token)
            writer_for(split, category).add(ids)
            n = len(ids)
            stats["total_documents"] += 1; stats["total_tokens"] += n
            stats["splits"][split]["documents"] += 1; stats["splits"][split]["tokens"] += n
            category_docs[category] += 1; category_tokens[category] += n
            broad_docs[broad] += 1; broad_tokens[broad] += n
            for sec in pred["secondary"]: broad_secondary[sec] += 1
            if pred["confidence"] < 0.35: stats["low_confidence_documents"] += 1
            if len(stats["samples"][category]) < args.sample_limit:
                stats["samples"][category].append({
                    "id": doc_id, "url": row.get("url", ""), "broad": broad,
                    "secondary_broad": pred["secondary"], "broad_confidence": pred["confidence"],
                    "subdomain": sub, "subdomain_confidence": sub_conf,
                    "fine_scores": fine_scores, "text_preview": text[:500],
                })

    print(f"Streaming {args.dataset} / {args.config} | device={device}")
    if source_rows_seen:
        print(f"Resuming after {source_rows_seen:,} source documents. Streaming source will skip them first.")
    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)

    batch = []
    try:
        with tqdm(total=None if args.max_docs < 0 else args.max_docs, initial=source_rows_seen,
                  desc="FineWeb-Edu", unit="doc") as bar:
            for i, row in enumerate(ds):
                if i < source_rows_seen:
                    continue
                if args.max_docs > 0 and source_rows_seen >= args.max_docs:
                    break
                batch.append(row)
                if len(batch) >= args.batch_size:
                    process(batch)
                    source_rows_seen += len(batch)
                    bar.update(len(batch))
                    batch.clear()
                    if source_rows_seen - last_checkpoint_seen >= args.checkpoint_every:
                        save_progress(); last_checkpoint_seen = source_rows_seen
                    if source_rows_seen - last_stats_seen >= args.stats_every:
                        print_stats(); last_stats_seen = source_rows_seen
            if batch and (args.max_docs < 0 or source_rows_seen < args.max_docs):
                if args.max_docs > 0:
                    batch = batch[:args.max_docs - source_rows_seen]
                process(batch); source_rows_seen += len(batch); bar.update(len(batch)); batch.clear()
        save_progress()
    except (KeyboardInterrupt, Exception):
        # Safe checkpoint boundary is the last completely processed batch.
        save_progress()
        print(f"\nSaved resumable state to {progress_path}")
        raise

    for w in writers.values():
        w.close()

    stats["categories"] = {k: {"documents": category_docs[k], "tokens": category_tokens[k]} for k in sorted(category_docs)}
    stats["broad_domains"] = {k: {"documents": broad_docs[k], "tokens": broad_tokens[k]} for k in sorted(broad_docs)}
    stats["secondary_domain_mentions"] = dict(broad_secondary)
    stats["samples"] = _jsonable_samples(stats["samples"])
    stats.update({
        "dataset": args.dataset, "config": args.config, "source_split": args.split,
        "classifier_model": args.classifier_model, "topic_threshold": args.topic_threshold,
        "tokenizer": "gpt2", "tokenizer_vocab": 50257, "eot_token": enc.eot_token,
        "dtype": "uint16", "bytes_per_token": 2, "shard_tokens": args.shard_tokens,
        "validation_fraction": args.val_pct, "source_rows_seen": source_rows_seen,
        "elapsed_seconds_this_session": time.time() - start,
        "output_bytes": _dir_size_bytes(args.output),
    })
    _atomic_json(args.output / "manifest.json", stats)
    progress_path.unlink(missing_ok=True)
    print_stats()
    print(json.dumps({
        "source_documents": source_rows_seen, "accepted_documents": stats["total_documents"],
        "tokens": stats["total_tokens"], "categories": len(stats["categories"]),
        "train": stats["splits"]["train"], "val": stats["splits"]["val"],
        "manifest": str(args.output / "manifest.json")
    }, indent=2))

if __name__ == "__main__":
    main()
