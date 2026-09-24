#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from asi import DATA_ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=(DATA_ROOT / "manifest.json"))
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--output", type=Path, default=(DATA_ROOT / "expert_pools.json"))
    a = p.parse_args()
    base = json.loads(a.manifest.read_text(encoding="utf-8"))
    pools = json.loads(a.pools.read_text(encoding="utf-8"))
    categories = set(base.get("categories", {}))
    assigned = set(); result = {"pools": {}, "unassigned_categories": {}}
    for pool, cats in pools.items():
        pool_tokens = pool_docs = 0
        for cat in cats:
            if cat not in categories:
                raise ValueError(f"Unknown category: {cat}")
            if cat in assigned:
                raise ValueError(f"Category assigned twice: {cat}")
            assigned.add(cat)
            pool_tokens += int(base["categories"][cat]["tokens"])
            pool_docs += int(base["categories"][cat]["documents"])
        result["pools"][pool] = {"categories": cats, "tokens": pool_tokens, "documents": pool_docs}
    for cat in sorted(categories - assigned):
        result["unassigned_categories"][cat] = base["categories"][cat]
    result["summary"] = {
        "pool_count": len(result["pools"]),
        "assigned_categories": len(assigned),
        "unassigned_categories": len(result["unassigned_categories"]),
        "assigned_tokens": sum(x["tokens"] for x in result["pools"].values()),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {a.output}")

if __name__ == "__main__":
    main()
