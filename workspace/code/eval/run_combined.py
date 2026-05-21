"""Combined accuracy + TTFT harness, with the two phases isolated.

This runs BOTH measurements that ``eval/run_eval.py`` (quality) and
``eval/run_ttft.py`` (latency) produce, in a single process that loads the
model **once** (per ``CLAUDE.md``'s "load the model once per call" cost rule),
and merges them into one ``<run_id>_combined.json``.

Why a dedicated driver instead of just calling both back-to-back
----------------------------------------------------------------
The requirement is that *measuring accuracy must not affect the TTFT numbers*.
Accuracy measurement does full-length generation (32-200 new tokens), which
grows the KV cache, fragments the CUDA caching allocator, and pushes the GPU
into a higher/throttled clock state. If that work ran before or interleaved
with the timed first-token region, it would bias TTFT. So this driver enforces:

1. **TTFT phase runs FIRST**, on a freshly-loaded (clean) device. Because the
   accuracy phase runs strictly *after* the timed region, it cannot perturb it
   -- this temporal ordering is the load-bearing guarantee, not a heuristic.
2. **Isolation barrier between phases**: ``cuda.synchronize`` + ``empty_cache``
   + ``reset_peak_memory_stats`` so the accuracy phase also starts from a clean
   allocator (and nothing it allocates can reach back into the timed region).
3. **Re-seed before accuracy** so its metrics are bit-identical to a standalone
   ``run_eval`` (greedy decoding is deterministic, but this keeps RNG state
   independent of phase 1).

The TTFT and accuracy phases reuse ``measure_ttft`` / ``measure_accuracy``
verbatim, so the combined numbers match the standalone harnesses exactly.

cacheblend TTFT caveat (inherited from run_ttft)
------------------------------------------------
This reproduction's ``cacheblend_generate`` is two-pass, so its measured TTFT
is an upper bound, NOT the paper's single-pass selective-recompute latency. The
``full_recompute`` vs ``full_reuse`` TTFT comparison is faithful; the accuracy
numbers are faithful for all three strategies.

Run as::

    python -m eval.run_combined --config configs/default.yaml
    python -m eval.run_combined --config configs/default.yaml --repeats 5 --warmup 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from eval.run_eval import _load_model, _set_seed, measure_accuracy
from eval.run_ttft import measure_ttft


def _merge(
    ttft_rows: List[Dict[str, Any]], acc_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Join the two phases' rows on (dataset, strategy, ratio).

    Accuracy rows contribute ``mean``; TTFT rows contribute ``ttft_ms_*`` and
    ``speedup_vs_recompute``. Order follows the accuracy rows (then any
    TTFT-only rows), so the merged list reads in the familiar run_eval order.
    """
    TTFT_FIELDS = ("ttft_ms_median", "ttft_ms_mean", "ttft_ms_p90", "speedup_vs_recompute")

    def key(r: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
        return (r["dataset"], r["strategy"], r.get("ratio"))

    merged: "Dict[Tuple[str, str, Optional[float]], Dict[str, Any]]" = {}
    order: List[Tuple[str, str, Optional[float]]] = []

    def slot(r: Dict[str, Any]) -> Dict[str, Any]:
        k = key(r)
        if k not in merged:
            merged[k] = {"dataset": r["dataset"], "strategy": r["strategy"],
                         "ratio": r.get("ratio"), "n": r.get("n")}
            order.append(k)
        if "deviation_mode" in r and "deviation_mode" not in merged[k]:
            merged[k]["deviation_mode"] = r["deviation_mode"]
        return merged[k]

    for r in acc_rows:
        slot(r)["mean"] = r["mean"]
    for r in ttft_rows:
        row = slot(r)
        for f in TTFT_FIELDS:
            if f in r:
                row[f] = r[f]

    return [merged[k] for k in order]


def run_combined(config_path: str, repeats: int = 3, warmup: int = 1) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed = cfg.get("seed", 42)
    _set_seed(seed)
    print(f"[run_combined] seed={seed} config={config_path} "
          f"repeats={repeats} warmup={warmup}")

    model, tokenizer, device, dtype = _load_model(cfg)

    # PHASE 1/2 -- TTFT FIRST, on a freshly-loaded (clean) device. Running TTFT
    # before any accuracy work is the guarantee that the accuracy phase cannot
    # perturb the TTFT numbers: long full-length generations, KV-cache growth,
    # allocator fragmentation and GPU clock/thermal drift all happen strictly
    # *after* the timed region.
    print("[run_combined] phase 1/2: TTFT (timed, isolated)")
    ttft_rows = measure_ttft(model, tokenizer, dtype, cfg, repeats=repeats, warmup=warmup)

    # Isolation barrier: drain the CUDA queue and release the caching-allocator
    # blocks the timed phase used, so phase 2 starts from a clean allocator.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Re-seed so accuracy is bit-identical to a standalone run_eval.
    _set_seed(seed)

    # PHASE 2/2 -- accuracy (full-length generation + scoring). Never timed.
    print("[run_combined] phase 2/2: accuracy (untimed)")
    acc_rows = measure_accuracy(model, tokenizer, dtype, cfg)

    results = _merge(ttft_rows, acc_rows)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_combined"
    out_path = out_dir / f"{run_id}.json"
    summary = {
        "config": cfg,
        "ttft": {"repeats": repeats, "warmup": warmup, "max_new_tokens": 1,
                 "note": "cacheblend is two-pass; its TTFT is an upper bound, "
                         "not the paper's single-pass selective-recompute TTFT"},
        "phase_order": ["ttft", "accuracy"],
        "isolation": ("TTFT measured first on a clean device; CUDA cache emptied "
                      "and RNG re-seeded before the accuracy phase; accuracy does "
                      "full-length generation but is never inside a timed region, "
                      "so it cannot affect the TTFT numbers."),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_combined] wrote {out_path}")

    print("\n=== accuracy + TTFT ===")
    for row in results:
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        mean = row.get("mean")
        mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else "n/a"
        ttft = row.get("ttft_ms_median")
        ttft_s = f"{ttft:.1f}ms" if isinstance(ttft, (int, float)) else "n/a"
        spd = row.get("speedup_vs_recompute")
        spd_s = f"{spd:.2f}x" if isinstance(spd, (int, float)) else "n/a"
        print(f"  {row['dataset']:<14} {row['strategy']:<16} {ratio_s:<8} "
              f"acc={mean_s} ttft={ttft_s} ({spd_s}) n={row.get('n')}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=3, help="timed TTFT iterations per example")
    p.add_argument("--warmup", type=int, default=1, help="untimed TTFT warmup iterations")
    p.add_argument("--deviation-mode", choices=("v", "k", "kv"), default=None,
                   help="override strategy.deviation_mode from the YAML")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.deviation_mode is not None:
        cfg.setdefault("strategy", {})["deviation_mode"] = args.deviation_mode
        tmp_path = Path(args.config).with_suffix(".combined_override.yaml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        run_combined(str(tmp_path), repeats=args.repeats, warmup=args.warmup)
    else:
        run_combined(args.config, repeats=args.repeats, warmup=args.warmup)


if __name__ == "__main__":
    main()
