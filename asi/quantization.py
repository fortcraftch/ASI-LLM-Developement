#!/usr/bin/env python3
"""Offline INT8 weight-only compression for routed V3 experts.

This is intentionally post-training and independent of the router. It provides
an actual compressed representation for the future W8 warm tier while keeping
the first specialization experiment free from quantization effects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def quantize_tensor_int8(t: torch.Tensor):
    t = t.detach().float().cpu()
    max_abs = float(t.abs().max().item())
    scale = max(max_abs / 127.0, 1e-8)
    q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
    return q.numpy(), np.asarray(scale, dtype=np.float32), t.shape


def export_experts(checkpoint_path: Path, output_dir: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    experts = {}
    for key, tensor in state.items():
        if ".ffn.experts." not in key or not key.endswith("weight"):
            continue
        parts = key.split(".")
        try:
            layer = int(parts[1])
            expert = int(parts[4])
            proj = parts[5]
        except (ValueError, IndexError):
            continue
        experts.setdefault((layer, expert), {})[proj] = tensor

    manifest = {
        "source_checkpoint": str(checkpoint_path),
        "quantization": "int8_symmetric_per_tensor",
        "experts": {},
    }
    dense_bytes = 0
    compressed_bytes = 0

    for (layer, expert), projections in sorted(experts.items()):
        out_path = output_dir / f"layer_{layer:02d}_expert_{expert:02d}.npz"
        arrays = {}
        entry = {"file": out_path.name, "projections": {}}
        for proj in ("w1", "w2", "w3"):
            if proj not in projections:
                raise ValueError(f"Missing {proj} for layer={layer}, expert={expert}")
            tensor = projections[proj]
            q, scale, shape = quantize_tensor_int8(tensor)
            arrays[f"{proj}_q"] = q
            arrays[f"{proj}_scale"] = scale
            dense = int(np.prod(shape)) * 4
            compressed = q.nbytes + np.asarray(scale).nbytes
            dense_bytes += dense
            compressed_bytes += compressed
            entry["projections"][proj] = {
                "shape": list(shape),
                "dense_bytes": dense,
                "compressed_bytes": compressed,
            }
        np.savez_compressed(out_path, **arrays)
        manifest["experts"][f"layer_{layer:02d}_expert_{expert:02d}"] = entry

    manifest["summary"] = {
        "expert_count": len(experts),
        "dense_bytes": dense_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": dense_bytes / compressed_bytes if compressed_bytes else None,
        "bytes_reduction_fraction": 1.0 - compressed_bytes / dense_bytes if dense_bytes else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Wrote {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("expert_int8"))
    args = p.parse_args()
    export_experts(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
