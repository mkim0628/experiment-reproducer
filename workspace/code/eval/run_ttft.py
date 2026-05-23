"""Time-to-first-token (TTFT) measurement harness.

This is the latency analogue of ``eval/run_eval.py``. Where ``run_eval``
iterates ``(dataset x strategy x recompute_ratio)`` and records a *quality*
score (F1 / Rouge-L), this harness records **TTFT** -- the wall-clock latency
to produce the first generated token -- for the same three strategies:

* ``full_recompute`` -- full prefill over the whole prompt, then 1 token.
* ``full_reuse``     -- chunk KV loaded from the (pre-warmed) store, only the
  query/suffix is prefilled, then 1 token.
* ``cacheblend``     -- selective recompute at one ratio, then 1 token.

It reuses ``run_eval``'s model/dataset/prompt helpers verbatim so the two
harnesses stay in lock-step.

IMPORTANT caveat about the ``cacheblend`` number
------------------------------------------------
This reproduction's ``cacheblend_generate`` is a **two-pass** implementation
(see ``cacheblend/baselines.py`` docstring): it runs one full forward to
capture K_new/V_new and a second pass to decode from the blended cache. That
matches the paper's *quality* exactly but makes its measured TTFT an **upper
bound**, NOT the paper's single-pass selective-recompute TTFT. So:

* ``full_recompute`` vs ``full_reuse`` TTFT here is a faithful comparison.
* ``cacheblend`` TTFT here is dominated by the extra capture pass and will
  often be >= full_recompute. Do not read it as the paper's TTFT speedup.

Chunk KV is **pre-warmed** before timing (``precompute_chunk_kv`` is
cache-aware, see ``cacheblend/precompute.py:49``), so the timed region excludes
offline chunk precompute -- matching the paper's assumption that chunk KV is
already cached.

Run as::

    python -m eval.run_ttft --config configs/default.yaml
    python -m eval.run_ttft --config configs/default.yaml --repeats 5 --warmup 2
    python -m eval.run_ttft --config configs/default.yaml --deviation-mode v
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import yaml

from cacheblend.baselines import (
    cacheblend_generate,
    full_recompute_generate,
    full_reuse_generate,
)
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.selective_recompute import BlendConfig
from cacheblend.single_pass import cacheblend_selective_generate
from eval.datasets import Example
from eval.run_eval import _build_prompts, _load_dataset, _load_model, _set_seed


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[], Any], repeats: int, warmup: int) -> List[float]:
    """Return per-iteration wall-clock latencies in milliseconds.

    ``warmup`` untimed iterations run first (CUDA autotune + populate the chunk
    KV store); the next ``repeats`` iterations are timed with a CUDA sync on
    both sides so the number reflects device time, not just kernel launch.
    """
    for _ in range(max(0, warmup)):
        fn()
    _sync()
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _agg(per_example_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(per_example_ms, dtype=float)
    return {
        "ttft_ms_median": float(np.median(arr)),
        "ttft_ms_mean": float(np.mean(arr)),
        "ttft_ms_p90": float(np.percentile(arr, 90)),
    }


def measure_ttft(
    model, tokenizer, dtype, cfg: dict, repeats: int = 3, warmup: int = 1
) -> List[Dict[str, Any]]:
    """TTFT phase: time first-token latency per (dataset, strategy, ratio).

    Takes an already-loaded model/tokenizer so a combined driver can share a
    single model load across the TTFT and accuracy phases (see
    ``eval/run_combined.py``) instead of paying for a second 14 GB weight load.
    Returns the per-row result dicts; the caller owns serialization.
    """
    n_layers = model.config.num_hidden_layers
    ratios = cfg["strategy"]["recompute_ratios"]
    check_layer = cfg["strategy"]["check_layer"]
    deviation_mode = cfg["strategy"].get("deviation_mode", "v")

    print(
        "[run_ttft] NOTE: cacheblend here is a two-pass impl; its TTFT is an "
        "upper bound, not the paper's single-pass selective-recompute latency."
    )

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        import os

        if not os.path.exists(ds_cfg["path"]):
            print(f"[run_ttft] dataset={ds_name}: {ds_cfg['path']} not found, skipping")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        print(f"[run_ttft] dataset={ds_name} n={len(examples)}")

        # TTFT = latency to the first token, so generate exactly one token.
        max_new = 1
        store = ChunkKVStore(n_layers, dtype, device="cpu")

        # Per-example median latency for each strategy.
        recompute_ms: List[float] = []
        reuse_ms: List[float] = []
        cb_ms: Dict[float, List[float]] = {r: [] for r in ratios}
        cb_sel_ms: Dict[float, List[float]] = {r: [] for r in ratios}

        for ex in examples:
            full_prompt, chunk_strs = _build_prompts(ds_name, ex)
            suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]

            recompute_ms.append(np.median(_time_call(
                lambda: full_recompute_generate(model, tokenizer, full_prompt, max_new),
                repeats, warmup,
            )))
            reuse_ms.append(np.median(_time_call(
                lambda: full_reuse_generate(
                    model, tokenizer, chunk_strs, query="", store=store,
                    max_new_tokens=max_new, suffix=suffix_text,
                ),
                repeats, warmup,
            )))
            for r in ratios:
                blend_cfg = BlendConfig(
                    recompute_ratio=r, check_layer=check_layer,
                    deviation_mode=deviation_mode,
                )
                # Two-pass cacheblend: faithful quality, but TTFT is an upper bound.
                cb_ms[r].append(np.median(_time_call(
                    lambda bc=blend_cfg: cacheblend_generate(
                        model, tokenizer, chunk_strs, query="", store=store,
                        cfg=bc, max_new_tokens=max_new, suffix=suffix_text,
                    ),
                    repeats, warmup,
                )))
                # True single-pass selective recompute: the paper's TTFT path.
                cb_sel_ms[r].append(np.median(_time_call(
                    lambda bc=blend_cfg: cacheblend_selective_generate(
                        model, tokenizer, chunk_strs, query="", store=store,
                        cfg=bc, max_new_tokens=max_new, suffix=suffix_text,
                    ),
                    repeats, warmup,
                )))

        rec_agg = _agg(recompute_ms)
        rec_median = rec_agg["ttft_ms_median"]
        results.append({"dataset": ds_name, "strategy": "full_recompute",
                        "ratio": None, "n": len(recompute_ms),
                        "speedup_vs_recompute": 1.0, **rec_agg})
        reuse_agg = _agg(reuse_ms)
        results.append({"dataset": ds_name, "strategy": "full_reuse",
                        "ratio": None, "n": len(reuse_ms),
                        "speedup_vs_recompute": rec_median / reuse_agg["ttft_ms_median"],
                        **reuse_agg})
        for r in ratios:
            cb_agg = _agg(cb_ms[r])
            results.append({"dataset": ds_name, "strategy": "cacheblend", "ratio": r,
                            "deviation_mode": deviation_mode, "n": len(cb_ms[r]),
                            "speedup_vs_recompute": rec_median / cb_agg["ttft_ms_median"],
                            **cb_agg})
        for r in ratios:
            sel_agg = _agg(cb_sel_ms[r])
            results.append({"dataset": ds_name, "strategy": "cacheblend_selective", "ratio": r,
                            "deviation_mode": deviation_mode, "n": len(cb_sel_ms[r]),
                            "speedup_vs_recompute": rec_median / sel_agg["ttft_ms_median"],
                            **sel_agg})

        print(f"  full_recompute median={rec_median:.1f}ms  "
              f"full_reuse median={reuse_agg['ttft_ms_median']:.1f}ms "
              f"({reuse_agg['ttft_ms_median'] and rec_median / reuse_agg['ttft_ms_median']:.2f}x)")
        for r in ratios:
            cb_row = next(x for x in results if x["dataset"] == ds_name
                          and x["strategy"] == "cacheblend" and x["ratio"] == r)
            sel_row = next(x for x in results if x["dataset"] == ds_name
                           and x["strategy"] == "cacheblend_selective" and x["ratio"] == r)
            print(f"  cacheblend(2pass) r={r} median={cb_row['ttft_ms_median']:.1f}ms "
                  f"({cb_row['speedup_vs_recompute']:.2f}x)  "
                  f"cacheblend_selective(1pass) median={sel_row['ttft_ms_median']:.1f}ms "
                  f"({sel_row['speedup_vs_recompute']:.2f}x)")
    return results


def run_ttft(config_path: str, repeats: int = 3, warmup: int = 1) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    print(f"[run_ttft] seed={cfg.get('seed', 42)} config={config_path} "
          f"repeats={repeats} warmup={warmup}")

    model, tokenizer, device, dtype = _load_model(cfg)
    results = measure_ttft(model, tokenizer, dtype, cfg, repeats=repeats, warmup=warmup)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_ttft"
    out_path = out_dir / f"{run_id}.json"
    summary = {
        "config": cfg,
        "ttft": {"repeats": repeats, "warmup": warmup, "max_new_tokens": 1,
                 "note": "strategy 'cacheblend' is the two-pass impl (TTFT is an "
                         "upper bound, not faithful); 'cacheblend_selective' is the "
                         "true single-pass selective recompute (the paper's TTFT)"},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_ttft] wrote {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=3, help="timed iterations per example")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup iterations")
    p.add_argument("--deviation-mode", choices=("v", "k", "kv"), default=None,
                   help="override strategy.deviation_mode from the YAML")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.deviation_mode is not None:
        cfg.setdefault("strategy", {})["deviation_mode"] = args.deviation_mode
        tmp_path = Path(args.config).with_suffix(".ttft_override.yaml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        run_ttft(str(tmp_path), repeats=args.repeats, warmup=args.warmup)
    else:
        run_ttft(args.config, repeats=args.repeats, warmup=args.warmup)


if __name__ == "__main__":
    main()
