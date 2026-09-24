#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
from asi import DATA_ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=(DATA_ROOT / "manifest.json"))
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--min-tokens", type=int, default=0)
    a = p.parse_args()
    
    data = json.loads(a.manifest.read_text(encoding="utf-8"))
    total = max(int(data.get("total_tokens", 0)), 1)
    rows = []
    for category, info in data.get("categories", {}).items():
        tokens, docs = int(info["tokens"]), int(info["documents"])
        if tokens >= a.min_tokens:
            rows.append({"category": category, "documents": docs, "tokens": tokens, "percent_tokens": 100*tokens/total})
    rows.sort(key=lambda x: x["tokens"], reverse=True)
    print(f"{'CATEGORY':60s} {'DOCS':>12s} {'TOKENS':>15s} {'%':>8s}")
    print("-"*100)
    for r in rows:
        print(f"{r['category'][:60]:60s} {r['documents']:12,d} {r['tokens']:15,d} {r['percent_tokens']:7.2f}%")
    print("-"*100)
    print(f"{'TOTAL':60s} {sum(x['documents'] for x in rows):12,d} {sum(x['tokens'] for x in rows):15,d}")
    if a.csv:
        with a.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["category","documents","tokens","percent_tokens"])
            w.writeheader(); w.writerows(rows)
        print(f"CSV: {a.csv}")

if __name__ == "__main__":
    main()
