"""Measure prefill (Time-To-First-Token) for full_recompute / full_reuse /
cacheblend(r) on each dataset listed in a config YAML.

Why this is its own script (not part of ``run_eval``):

* This is the Modal-path *latency-model approximation* (driven by
  ``run_latency_modal.py``). The Cerebrium eval path
  (``eval/run_ttft.py`` / ``eval/run_combined.py``) instead times the real
  single-pass ``cacheblend.single_pass.cacheblend_selective_generate``.
* The approximation mirrors the paper's algorithmic cost (single forward,
  selective recompute):

    full_recompute(prompt)        = forward( full_prompt_ids )
    full_reuse(chunks, suffix)    = forward( suffix_ids,                  past_key_values = fused_chunk_cache )
    cacheblend(chunks, suffix, r) = forward( recompute_ids ++ suffix_ids, past_key_values = partial_chunk_cache )

  where ``recompute_ids`` is the last ``ceil(r * total_chunk_tokens)`` tokens
  of the concatenated chunks and ``partial_chunk_cache`` keeps only the
  prefix of the fused cache that is NOT being recomputed. This isolates the
  prefill arithmetic that the paper's two-thread Fusor optimizes; cache
  loading is excluded from the timer (paper's idealized prefix-caching
  baseline does the same).

All times are measured with ``torch.cuda.Event`` after ``cuda.synchronize``,
with ``warmup`` runs first to stabilize SDPA/cuBLAS launch costs.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

_CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from cacheblend.baselines import _build_fused_cache, _tok_ids
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.precompute import precompute_chunk_kv
from eval.datasets import (
    build_claim_verification_prompt,
    build_qa_prompt,
    build_summarization_prompt,
    load_hotpotqa,
    load_hover,
    load_multihop_rag,
    load_musique,
    load_nq_dpr,
    load_samsum,
    load_wikimqa,
)

try:
    from eval.datasets import load_multinews  # added in a follow-on commit
except ImportError:  # pragma: no cover
    load_multinews = None


# ---------------------------------------------------------------- dispatchers
def _load_dataset(name: str, path: str, n: int):
    table = {
        "wikimqa": load_wikimqa,
        "musique": load_musique,
        "samsum": load_samsum,
        "hotpotqa": load_hotpotqa,
        "multihop_rag": load_multihop_rag,
        "hover": load_hover,
        "nq_dpr": load_nq_dpr,
    }
    if name == "multinews" and load_multinews is not None:
        return load_multinews(path, n)
    if name not in table:
        raise ValueError(f"unknown dataset {name}")
    return table[name](path, n)


def _build_prompt(name: str, ex) -> Tuple[str, List[str]]:
    if name in ("wikimqa", "musique", "hotpotqa", "multihop_rag", "nq_dpr"):
        return build_qa_prompt(ex.question, ex.contexts)
    if name == "samsum":
        dialogue = ex.metadata.get("input", ex.question)
        return build_summarization_prompt(dialogue, ex.contexts)
    if name == "hover":
        return build_claim_verification_prompt(ex.question, ex.contexts)
    if name == "multinews":
        # multinews has a build_multinews_prompt in eval/datasets.py; resolve dynamically.
        from eval.datasets import build_multinews_prompt
        return build_multinews_prompt(ex.contexts)
    raise ValueError(f"unknown dataset {name}")


# ---------------------------------------------------------------- helpers
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trim_fused_cache(fused_cache, keep_len: int):
    """Return a new DynamicCache containing only the first ``keep_len`` tokens
    per layer of the input fused_cache. Used by cacheblend(r) to simulate the
    state where ``r * total_chunk_tokens`` tail tokens are about to be
    selectively recomputed."""
    from transformers import DynamicCache

    new_cache = DynamicCache()
    for li in range(len(fused_cache.layers)):
        k = fused_cache.layers[li].keys[..., :keep_len, :].contiguous()
        v = fused_cache.layers[li].values[..., :keep_len, :].contiguous()
        new_cache.update(k, v, li)
    return new_cache


@torch.no_grad()
def _timed_prefill(model, input_ids, past_key_values, warmup: int, repeats: int) -> List[float]:
    times_ms: List[float] = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        _ = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=False,
        )
        ev_end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times_ms.append(float(ev_start.elapsed_time(ev_end)))
    return times_ms


# ---------------------------------------------------------------- per-example
@torch.no_grad()
def measure_example(
    model, tokenizer, ds_name: str, ex, ratios: List[float],
    warmup: int, repeats: int, device, dtype,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    full_prompt, chunks = _build_prompt(ds_name, ex)
    suffix_text = full_prompt[sum(len(c) for c in chunks):]

    # Token ids that the model will actually see.
    full_ids = _tok_ids(tokenizer, full_prompt, add_special_tokens=True).to(device)
    suffix_ids = _tok_ids(tokenizer, suffix_text, add_special_tokens=False).to(device)
    chunk_ids_list = [
        _tok_ids(tokenizer, c, add_special_tokens=(i == 0)).to(device)
        for i, c in enumerate(chunks)
    ]
    chunk_lens = [c.shape[1] for c in chunk_ids_list]

    # ---- Pre-cache chunk KV (OFF the timer; "prefix caching" baseline)
    store = ChunkKVStore(model.config.num_hidden_layers, dtype, device="cpu")
    chunk_hashes: List[bytes] = []
    for i, c in enumerate(chunks):
        chunk_hashes.append(
            precompute_chunk_kv(model, tokenizer, c, store, add_special_tokens=(i == 0))
        )
    offsets = [0]
    for L in chunk_lens[:-1]:
        offsets.append(offsets[-1] + L)
    fused_cache, total_chunk_len = _build_fused_cache(
        model, chunk_hashes, store, offsets, chunk_lens, device=device, dtype=dtype
    )

    # ---- 1) full_recompute: forward over the full prompt
    full_times = _timed_prefill(model, full_ids, None, warmup, repeats)

    # ---- 2) full_reuse: suffix prefill against the fully prefilled cache
    fused_cache_clone = _trim_fused_cache(fused_cache, int(total_chunk_len))
    reuse_times = _timed_prefill(model, suffix_ids, fused_cache_clone, warmup, repeats)

    # ---- 3) cacheblend(r): partial cache + (r*chunk + suffix) prefill
    concat_chunks = torch.cat(chunk_ids_list, dim=1)
    blend_times: Dict[str, List[float]] = {}
    for r in ratios:
        n_rec = max(1, int(np.ceil(r * total_chunk_len)))
        keep_len = max(0, int(total_chunk_len) - n_rec)
        partial_cache = _trim_fused_cache(fused_cache, keep_len)
        rec_ids = concat_chunks[:, -n_rec:]
        cb_input = torch.cat([rec_ids, suffix_ids], dim=1)
        blend_times[f"cacheblend_r{r:.2f}"] = _timed_prefill(
            model, cb_input, partial_cache, warmup, repeats
        )

    sizes = {
        "total_chunk_tokens": int(total_chunk_len),
        "suffix_tokens": int(suffix_ids.shape[1]),
        "full_prompt_tokens": int(full_ids.shape[1]),
        "num_chunks": len(chunks),
    }
    return {"full_recompute": full_times, "full_reuse": reuse_times, **blend_times}, sizes


# ---------------------------------------------------------------- aggregation
def _agg(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {}
    sorted_t = sorted(times_ms)
    return {
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "p50_ms": round(sorted_t[len(sorted_t) // 2], 3),
        "p95_ms": round(sorted_t[max(0, int(np.ceil(0.95 * len(sorted_t))) - 1)], 3),
        "n_runs": len(times_ms),
    }


# ---------------------------------------------------------------- main loop
def run_latency(config_path: str, n_examples: int, warmup: int, repeats: int) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"].get("dtype", "float16"))
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[latency] loading {name} dtype={dtype} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()

    ratios = cfg["strategy"]["recompute_ratios"]
    summary: Dict[str, Any] = {
        "model": name,
        "dtype": str(dtype),
        "ratios": ratios,
        "warmup": warmup,
        "repeats": repeats,
        "approximation": (
            "single-forward latency model: cacheblend(r) ~ forward(r*chunk + suffix) "
            "against a prefilled cache of (1-r)*chunk tokens"
        ),
        "datasets": {},
    }

    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            print(f"[latency] dataset={ds_name}: {ds_cfg['path']} missing, skipping")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], min(n_examples, ds_cfg["n"]))
        print(f"[latency] dataset={ds_name} n_examples={len(examples)}")

        per_strategy: Dict[str, List[float]] = {}
        size_acc: List[Dict[str, int]] = []
        for i, ex in enumerate(examples):
            try:
                times, sizes = measure_example(
                    model, tokenizer, ds_name, ex, ratios, warmup, repeats, device, dtype
                )
            except torch.cuda.OutOfMemoryError as e:  # pragma: no cover
                print(f"  ex {i+1}/{len(examples)} OOM: {e}; skipping")
                torch.cuda.empty_cache()
                continue
            for k, v in times.items():
                per_strategy.setdefault(k, []).extend(v)
            size_acc.append(sizes)
            if (i + 1) % 5 == 0 or i == len(examples) - 1:
                cb_keys = [k for k in per_strategy if k.startswith("cacheblend_")]
                cb_min = min(per_strategy[k] for k in cb_keys) if cb_keys else []
                print(
                    f"  ex {i+1}/{len(examples)} "
                    f"full_recompute_mean_ms={np.mean(per_strategy['full_recompute']):.1f} "
                    f"full_reuse_mean_ms={np.mean(per_strategy['full_reuse']):.1f}"
                )

        agg = {k: _agg(v) for k, v in per_strategy.items()}
        # Speedups relative to full_recompute (mean).
        base = agg.get("full_recompute", {}).get("mean_ms")
        if base:
            for k, v in agg.items():
                if v and "mean_ms" in v and k != "full_recompute":
                    v["speedup_over_full_recompute"] = round(base / v["mean_ms"], 2)

        avg_sizes = {
            "avg_full_prompt_tokens": int(
                np.mean([s["full_prompt_tokens"] for s in size_acc])
            ),
            "avg_total_chunk_tokens": int(
                np.mean([s["total_chunk_tokens"] for s in size_acc])
            ),
            "avg_suffix_tokens": int(np.mean([s["suffix_tokens"] for s in size_acc])),
            "avg_num_chunks": round(
                float(np.mean([s["num_chunks"] for s in size_acc])), 2
            ),
            "n_examples_measured": len(size_acc),
        }

        summary["datasets"][ds_name] = {"sizes": avg_sizes, "strategies": agg}
        print(f"[latency] {ds_name} done: {json.dumps(agg, indent=2)}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=10, help="examples per dataset")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", default="workspace/results/latency.json")
    args = p.parse_args()

    summary = run_latency(args.config, args.n, args.warmup, args.repeats)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[latency] wrote {out_path}")


if __name__ == "__main__":
    main()
