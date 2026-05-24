"""Stage-0 driver: quantify how much headroom CacheBlend's HKVD selection leaves.

For each dataset x selection rule x recompute ratio, this measures the first-token
logit deviation from a true full prefill (see :mod:`cacheblend.analysis`). The
point is to answer two questions *before* committing to a full Stage-1 sweep:

  1. Headroom: how much better is the ``oracle`` selection than the released
     ``raw`` KV-deviation selection at the same ``r``?
  2. Capture: how much of that headroom does the cheap online ``attn_weighted``
     candidate recover?

Run cheaply (a handful of examples) -- the numbers are only meaningful on a real
trained model, so this is a smoke-sized GPU job, not a full grid:

    python -m scripts.selection_analysis --config configs/default.yaml \
        --datasets wikimqa --n 5 --ratios 0.05 0.1 0.15

It reuses the eval harness's model load, dataset loaders, and prompt builders, so
the construction matches the accuracy/TTFT runs exactly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from cacheblend.analysis import selection_quality
from cacheblend.kv_cache import ChunkKVStore
from eval.run_eval import _build_prompts, _load_dataset, _load_model, _suffix_of


def _aggregate(per_example: List[Dict], rules, ratios) -> List[Dict]:
    """Mean logit deviation + argmax-match rate per (rule, ratio) across examples."""
    out: List[Dict] = []
    for rule in rules:
        for r in ratios:
            l2, mx, match, nsel = [], [], [], []
            for ex in per_example:
                for row in ex["rows"]:
                    if row["rule"] == rule and abs(row["ratio"] - r) < 1e-9:
                        l2.append(row["logit_l2"])
                        mx.append(row["logit_max"])
                        match.append(1.0 if row["argmax_match"] else 0.0)
                        nsel.append(row["n_selected"])
            if not l2:
                continue
            out.append({
                "rule": rule, "ratio": r, "n_examples": len(l2),
                "logit_l2_mean": float(np.mean(l2)),
                "logit_max_mean": float(np.mean(mx)),
                "argmax_match_rate": float(np.mean(match)),
                "n_selected_mean": float(np.mean(nsel)),
            })
    return out


def run(config_path: str, datasets, n: int, ratios, rules, deviation_mode: str,
        check_layer: int, mass_source: str) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model, tokenizer, device, dtype = _load_model(cfg)
    n_layers = model.config.num_hidden_layers

    selected = datasets or list(cfg["datasets"].keys())
    summary_ds: Dict[str, dict] = {}
    for ds_name in selected:
        ds_cfg = cfg["datasets"].get(ds_name)
        if ds_cfg is None:
            print(f"[stage0] unknown dataset {ds_name}, skipping")
            continue
        import os
        if not os.path.exists(ds_cfg["path"]):
            print(f"[stage0] {ds_name}: {ds_cfg['path']} not found, skipping")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], n)
        print(f"[stage0] dataset={ds_name} n={len(examples)} rules={list(rules)} "
              f"ratios={list(ratios)} dev={deviation_mode}")
        store = ChunkKVStore(n_layers, dtype, device="cpu")
        per_example: List[Dict] = []
        for i, ex in enumerate(examples):
            full_prompt, chunk_strs = _build_prompts(ds_name, ex)
            suffix_text = _suffix_of(full_prompt, chunk_strs)
            per_example.append(selection_quality(
                model, tokenizer, chunk_strs, suffix_text, store,
                ratios=ratios, rules=rules, deviation_mode=deviation_mode,
                check_layer=check_layer, mass_source=mass_source))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(examples)}")
        agg = _aggregate(per_example, rules, ratios)
        summary_ds[ds_name] = {"aggregate": agg, "per_example": per_example}
        # Console: headroom (oracle vs raw) and capture (attn_weighted vs raw).
        for r in ratios:
            byrule = {row["rule"]: row for row in agg if abs(row["ratio"] - r) < 1e-9}
            if "raw" in byrule:
                line = f"  r={r:.2f}  " + "  ".join(
                    f"{rule}: l2={byrule[rule]['logit_l2_mean']:.3f} "
                    f"match={byrule[rule]['argmax_match_rate']:.2f}"
                    for rule in rules if rule in byrule)
                print(line)

    summary = {
        "config": config_path,
        "deviation_mode": deviation_mode,
        "check_layer": check_layer,
        "mass_source": mass_source,
        "rules": list(rules),
        "ratios": list(ratios),
        "datasets": summary_ds,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (time.strftime("%Y%m%d_%H%M%S") + "_stage0_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage0] wrote {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="subset of dataset keys; default = all in config")
    p.add_argument("--n", type=int, default=5, help="examples per dataset (keep small)")
    p.add_argument("--ratios", type=float, nargs="*", default=[0.05, 0.1, 0.15, 0.18])
    p.add_argument("--rules", nargs="*",
                   default=["raw", "attn_weighted", "oracle", "random"])
    p.add_argument("--deviation-mode", choices=("v", "k", "kv"), default="v")
    p.add_argument("--check-layer", type=int, default=1)
    p.add_argument("--mass-source", choices=("suffix", "all"), default="suffix")
    args = p.parse_args()
    run(args.config, args.datasets, args.n, args.ratios, args.rules,
        args.deviation_mode, args.check_layer, args.mass_source)


if __name__ == "__main__":
    main()
