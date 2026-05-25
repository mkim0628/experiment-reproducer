"""CacheBlend evaluation -- accuracy AND TTFT from one single-pass implementation.

Iterates ``(dataset x strategy x recompute_ratio)`` and, for the three
strategies ``full_recompute`` / ``full_reuse`` / ``cacheblend``, measures BOTH:

* **accuracy** -- full-length generation + metric (F1 / Rouge-L / claim acc).
* **TTFT**     -- wall-clock latency to the first generated token.

``cacheblend`` is the single-pass selective recompute
(:func:`cacheblend.single_pass.cacheblend_selective_generate`); the SAME function
is timed (TTFT) and scored (accuracy), so each row's acc/ttft come from one
inference path -- the accuracy-drop vs TTFT-saving trade-off the paper reports,
read off one implementation.

``run_combined`` (the default driver) loads the model ONCE and measures TTFT
FIRST on a clean device, THEN accuracy: the accuracy phase's long generations,
KV-cache growth and GPU clock drift therefore happen strictly AFTER the timed
region and cannot perturb it. Re-seeding before the accuracy phase keeps its
metrics independent of the timed phase.

Run as::

    python -m eval.run_eval --config configs/default.yaml                 # acc + TTFT
    python -m eval.run_eval --config configs/default.yaml --repeats 5 --warmup 2
    python -m eval.run_eval --config configs/default.yaml --check-correctness
    python -m eval.run_eval --config configs/default.yaml --accuracy-only  # accuracy only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from cacheblend.baselines import full_recompute_generate, full_reuse_generate
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.selective_recompute import BlendConfig
from cacheblend.single_pass import cacheblend_selective_generate, check_r1_matches_full_forward
from eval.datasets import (
    Example,
    build_claim_verification_prompt,
    build_multinews_prompt,
    build_qa_prompt,
    build_summarization_prompt,
    load_hotpotqa,
    load_hover,
    load_multihop_rag,
    load_multinews,
    load_musique,
    load_nq_dpr,
    load_samsum,
    load_wikimqa,
)
from eval.metrics import (
    compute_claim_verification_max,
    compute_f1_max,
    compute_rouge_l,
)


# --------------------------------------------------------------- shared helpers
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(cfg: Dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"].get("dtype", "float16"))
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device, dtype


def _score(metric: str, pred: str, ex: Example, tokenizer) -> float:
    if metric == "f1":
        return compute_f1_max(pred, ex.answers, tokenizer)
    if metric == "rouge_l":
        return max(compute_rouge_l(pred, g) for g in ex.answers)
    if metric == "claim_verification":
        return compute_claim_verification_max(pred, ex.answers)
    raise ValueError(f"unknown metric {metric}")


def _build_prompts(dataset_name: str, ex: Example):
    if dataset_name in ("wikimqa", "musique", "hotpotqa", "multihop_rag", "nq_dpr"):
        return build_qa_prompt(ex.question, ex.contexts)
    if dataset_name == "samsum":
        dialogue = ex.metadata.get("input", ex.question)
        return build_summarization_prompt(dialogue, ex.contexts)
    if dataset_name == "hover":
        return build_claim_verification_prompt(ex.question, ex.contexts)
    if dataset_name == "multinews":
        return build_multinews_prompt(ex.contexts)
    raise ValueError(f"unknown dataset {dataset_name}")


def _load_dataset(name: str, path: str, n: int) -> List[Example]:
    loaders = {
        "wikimqa": load_wikimqa, "musique": load_musique, "samsum": load_samsum,
        "hotpotqa": load_hotpotqa, "multihop_rag": load_multihop_rag,
        "hover": load_hover, "multinews": load_multinews, "nq_dpr": load_nq_dpr,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name}")
    return loaders[name](path, n)


def _suffix_of(full_prompt: str, chunk_strs: List[str]) -> str:
    """The query/instruction text the prompt template places AFTER the chunks."""
    return full_prompt[sum(len(c) for c in chunk_strs):]


# --------------------------------------------------------------- timing helpers
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[], Any], repeats: int, warmup: int) -> List[float]:
    """Return per-iteration wall-clock latencies in milliseconds.

    ``warmup`` untimed iterations run first (CUDA autotune + populate the chunk
    KV store); the next ``repeats`` iterations are timed with a CUDA sync on both
    sides so the number reflects device time, not just kernel launch.
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


def _environment_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Capture the hardware / software stack the run executed on."""
    import platform
    import sys as _sys

    import transformers

    info: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        "gpu": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_memory_total_gb": None,
        "model": cfg.get("model", {}).get("name"),
        "dtype": cfg.get("model", {}).get("dtype"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_total_gb"] = round(props.total_memory / 1e9, 2)
    return info


# ----------------------------------------------------------------- accuracy
def _budget_points(strat: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, float]]:
    """Resolve the recompute-budget sweep from the strategy config.

    Returns ``(budget_mode, points, extra)``. ``budget_mode="ratio"`` sweeps
    ``recompute_ratios`` (fixed top-r%); ``"threshold"`` sweeps ``thresholds`` (tau)
    with shared ``min_frac``/``max_frac`` clamps. The sweep axis (the per-cell
    "point") is a ratio r or a threshold tau respectively.
    """
    if strat.get("budget_mode", "ratio") == "threshold":
        return "threshold", list(strat.get("thresholds", [0.5])), {
            "min_frac": strat.get("min_frac", 0.0),
            "max_frac": strat.get("max_frac", 1.0)}
    return "ratio", list(strat["recompute_ratios"]), {}


def _blend_cfg(strat: Dict[str, Any], sel: str, budget_mode: str, pt: float,
               extra: Dict[str, float]) -> BlendConfig:
    """Build the BlendConfig for one (selection, budget point) cell."""
    common = dict(check_layer=strat["check_layer"],
                  deviation_mode=strat.get("deviation_mode", "v"),
                  selection=sel, mass_source=strat.get("mass_source", "suffix"))
    if budget_mode == "threshold":
        return BlendConfig(budget_mode="threshold", threshold=pt,
                           min_frac=extra["min_frac"], max_frac=extra["max_frac"],
                           **common)
    return BlendConfig(recompute_ratio=pt, **common)


def _budget_fields(budget_mode: str, pt: float,
                   mean_budget: Optional[float] = None) -> Dict[str, Any]:
    """Result-row fields naming the budget point (kept distinct so _merge keys it).

    Ratio mode puts the point in ``ratio`` (back-compat); threshold mode leaves
    ``ratio`` None and records ``threshold`` plus, when known, the realized
    ``mean_budget`` (mean recomputed fraction actually selected).
    """
    if budget_mode == "threshold":
        out: Dict[str, Any] = {"ratio": None, "threshold": pt, "budget_mode": "threshold"}
        if mean_budget is not None:
            out["mean_budget"] = mean_budget
        return out
    return {"ratio": pt, "budget_mode": "ratio"}


def measure_accuracy(model, tokenizer, dtype, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accuracy phase: full-length generation + metric per (dataset, strategy, ratio).

    Takes an already-loaded model/tokenizer so the combined driver shares a single
    model load across phases. Does full-length generation and is never timed, so
    in the combined driver it runs strictly AFTER the TTFT phase.
    """
    n_layers = model.config.num_hidden_layers
    max_new = cfg["generation"]["max_new_tokens"]
    strat = cfg["strategy"]
    check_layer = strat["check_layer"]
    deviation_mode = strat.get("deviation_mode", "v")
    selection_modes = strat.get("selection_modes", ["raw"])
    budget_mode, points, extra = _budget_points(strat)

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            print(f"[eval] dataset={ds_name}: file {ds_cfg['path']} not found, "
                  "skipping (run scripts/download_extra_datasets.py to fetch)")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        metric = ds_cfg["metric"]
        ds_max_new = ds_cfg.get("max_new_tokens", max_new)  # MultiNews needs ~150-200
        print(f"[eval] accuracy: dataset={ds_name} n={len(examples)} metric={metric} "
              f"max_new={ds_max_new} selection={selection_modes} {budget_mode}={points}")

        scores_full, scores_reuse = [], []
        scores_cb: Dict[Tuple[str, float], List[float]] = {
            (sel, pt): [] for sel in selection_modes for pt in points}
        # realized recompute fraction per cell (threshold mode's adaptive budget)
        budget_cb: Dict[Tuple[str, float], List[float]] = {
            (sel, pt): [] for sel in selection_modes for pt in points}
        n_skipped = 0
        for i, ex in enumerate(examples):
            full_prompt, chunk_strs = _build_prompts(ds_name, ex)
            suffix_text = _suffix_of(full_prompt, chunk_strs)
            # Fresh store PER EXAMPLE: examples have distinct contexts (no
            # cross-example chunk reuse), so a dataset-scoped store would just
            # accumulate every example's chunk KV on CPU RAM and OOM-kill the
            # process at large n. Within an example the store is still shared by
            # full_reuse + all cacheblend cells (precompute is cache-aware).
            store = ChunkKVStore(n_layers, dtype, device="cpu")

            # Compute every strategy's score into temps first; only commit if the
            # whole example succeeds, so a mid-example failure (e.g. an OOM on a
            # long-context example) can't misalign the per-strategy score lists.
            # One bad example is skipped, not fatal to the whole run.
            try:
                s_full = _score(metric, full_recompute_generate(
                    model, tokenizer, full_prompt, ds_max_new), ex, tokenizer)
                s_reuse = _score(metric, full_reuse_generate(
                    model, tokenizer, chunk_strs, query="", store=store,
                    max_new_tokens=ds_max_new, suffix=suffix_text), ex, tokenizer)
                s_cb: Dict[Tuple[str, float], float] = {}
                b_cb: Dict[Tuple[str, float], float] = {}
                for sel in selection_modes:
                    for pt in points:
                        bc = _blend_cfg(strat, sel, budget_mode, pt, extra)
                        stats: Dict[str, Any] = {}
                        s_cb[(sel, pt)] = _score(metric, cacheblend_selective_generate(
                            model, tokenizer, chunk_strs, query="", store=store,
                            cfg=bc, max_new_tokens=ds_max_new, suffix=suffix_text,
                            stats=stats), ex, tokenizer)
                        b_cb[(sel, pt)] = stats.get("budget_frac", float("nan"))
            except Exception as e:  # noqa: BLE001 -- isolate one bad example
                n_skipped += 1
                print(f"[eval] WARN accuracy: skipping {ds_name} ex {i}: "
                      f"{type(e).__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            scores_full.append(s_full)
            scores_reuse.append(s_reuse)
            for key, val in s_cb.items():
                scores_cb[key].append(val)
                budget_cb[key].append(b_cb[key])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(examples)} full={np.mean(scores_full):.3f} "
                      f"reuse={np.mean(scores_reuse):.3f} skipped={n_skipped}")

        if n_skipped:
            print(f"[eval] accuracy: dataset={ds_name} skipped {n_skipped}/{len(examples)} "
                  f"examples; means are over the {len(scores_full)} that ran")
        if not scores_full:
            print(f"[eval] accuracy: dataset={ds_name} ALL examples failed, no rows")
            continue
        results.append({"dataset": ds_name, "strategy": "full_recompute", "ratio": None,
                        "mean": float(np.mean(scores_full)), "n": len(scores_full),
                        "n_skipped": n_skipped})
        results.append({"dataset": ds_name, "strategy": "full_reuse", "ratio": None,
                        "mean": float(np.mean(scores_reuse)), "n": len(scores_reuse),
                        "n_skipped": n_skipped})
        for sel in selection_modes:
            for pt in points:
                key = (sel, pt)
                mean_budget = (float(np.mean(budget_cb[key]))
                               if budget_mode == "threshold" and budget_cb[key] else None)
                row = {"dataset": ds_name, "strategy": "cacheblend",
                       "selection": sel, "deviation_mode": deviation_mode,
                       "mean": float(np.mean(scores_cb[key])), "n": len(scores_cb[key]),
                       **_budget_fields(budget_mode, pt, mean_budget)}
                results.append(row)
                budget_str = f" budget={mean_budget:.3f}" if mean_budget is not None else ""
                print(f"  cacheblend sel={sel} {budget_mode}={pt} dev={deviation_mode}: "
                      f"mean={np.mean(scores_cb[key]):.3f}{budget_str}")
    return results


# --------------------------------------------------------------------- TTFT
def measure_ttft(
    model, tokenizer, dtype, cfg: dict, repeats: int = 3, warmup: int = 1
) -> List[Dict[str, Any]]:
    """TTFT phase: median first-token latency per (dataset, strategy, ratio).

    Chunk KV is pre-warmed by the warmup iterations (``precompute_chunk_kv`` is
    cache-aware), so the timed region excludes offline chunk precompute -- the
    paper's assumption that chunk KV is already cached.
    """
    n_layers = model.config.num_hidden_layers
    strat = cfg["strategy"]
    check_layer = strat["check_layer"]
    deviation_mode = strat.get("deviation_mode", "v")
    selection_modes = strat.get("selection_modes", ["raw"])
    budget_mode, points, extra = _budget_points(strat)

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            print(f"[eval] ttft: dataset={ds_name}: {ds_cfg['path']} not found, skipping")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        print(f"[eval] ttft: dataset={ds_name} n={len(examples)} "
              f"selection={selection_modes} {budget_mode}={points}")

        max_new = 1  # TTFT = latency to the first token
        recompute_ms, reuse_ms = [], []
        cb_ms: Dict[Tuple[str, float], List[float]] = {
            (sel, pt): [] for sel in selection_modes for pt in points}
        n_skipped = 0

        for i, ex in enumerate(examples):
            full_prompt, chunk_strs = _build_prompts(ds_name, ex)
            suffix_text = _suffix_of(full_prompt, chunk_strs)
            # Fresh store per example (see measure_accuracy): bounds CPU RAM to a
            # single example's chunk KV instead of accumulating all n examples'.
            store = ChunkKVStore(n_layers, dtype, device="cpu")

            # As in measure_accuracy: time every strategy into temps and commit
            # atomically, so a single failing example is skipped (not fatal) and
            # the per-strategy latency lists stay aligned across examples.
            try:
                rec = np.median(_time_call(
                    lambda: full_recompute_generate(model, tokenizer, full_prompt, max_new),
                    repeats, warmup))
                reu = np.median(_time_call(
                    lambda: full_reuse_generate(
                        model, tokenizer, chunk_strs, query="", store=store,
                        max_new_tokens=max_new, suffix=suffix_text),
                    repeats, warmup))
                cb_one: Dict[Tuple[str, float], float] = {}
                for sel in selection_modes:
                    for pt in points:
                        bc = _blend_cfg(strat, sel, budget_mode, pt, extra)
                        cb_one[(sel, pt)] = np.median(_time_call(
                            lambda bc=bc: cacheblend_selective_generate(
                                model, tokenizer, chunk_strs, query="", store=store,
                                cfg=bc, max_new_tokens=max_new, suffix=suffix_text),
                            repeats, warmup))
            except Exception as e:  # noqa: BLE001 -- isolate one bad example
                n_skipped += 1
                print(f"[eval] WARN ttft: skipping {ds_name} ex {i}: "
                      f"{type(e).__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            recompute_ms.append(rec)
            reuse_ms.append(reu)
            for key, val in cb_one.items():
                cb_ms[key].append(val)

        if n_skipped:
            print(f"[eval] ttft: dataset={ds_name} skipped {n_skipped}/{len(examples)} examples")
        if not recompute_ms:
            print(f"[eval] ttft: dataset={ds_name} ALL examples failed, no rows")
            continue
        rec_agg = _agg(recompute_ms)
        rec_median = rec_agg["ttft_ms_median"]
        results.append({"dataset": ds_name, "strategy": "full_recompute", "ratio": None,
                        "n": len(recompute_ms), "speedup_vs_recompute": 1.0, **rec_agg})
        reuse_agg = _agg(reuse_ms)
        results.append({"dataset": ds_name, "strategy": "full_reuse", "ratio": None,
                        "n": len(reuse_ms),
                        "speedup_vs_recompute": rec_median / reuse_agg["ttft_ms_median"],
                        **reuse_agg})
        for sel in selection_modes:
            for pt in points:
                cb_agg = _agg(cb_ms[(sel, pt)])
                results.append({"dataset": ds_name, "strategy": "cacheblend",
                                "selection": sel, "deviation_mode": deviation_mode,
                                "n": len(cb_ms[(sel, pt)]),
                                "speedup_vs_recompute": rec_median / cb_agg["ttft_ms_median"],
                                **_budget_fields(budget_mode, pt), **cb_agg})

        print(f"  full_recompute median={rec_median:.1f}ms  "
              f"full_reuse median={reuse_agg['ttft_ms_median']:.1f}ms "
              f"({rec_median / reuse_agg['ttft_ms_median']:.2f}x)")
        for sel in selection_modes:
            for pt in points:
                cb_agg = _agg(cb_ms[(sel, pt)])
                spd = rec_median / cb_agg["ttft_ms_median"]
                print(f"  cacheblend sel={sel} {budget_mode}={pt} "
                      f"median={cb_agg['ttft_ms_median']:.1f}ms ({spd:.2f}x)")
    return results


# ----------------------------------------------------------------- combine
def _merge(ttft_rows: List[Dict[str, Any]],
           acc_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join the two phases' rows on (dataset, strategy, ratio).

    Accuracy rows contribute ``mean``; TTFT rows contribute ``ttft_ms_*`` and
    ``speedup_vs_recompute``. Order follows the accuracy rows.
    """
    TTFT_FIELDS = ("ttft_ms_median", "ttft_ms_mean", "ttft_ms_p90", "speedup_vs_recompute")
    CARRY = ("budget_mode", "threshold", "mean_budget")

    def key(r: Dict[str, Any]) -> Tuple[str, str, Optional[float], Optional[str]]:
        # Budget point is `ratio` (ratio mode) or `threshold` (threshold mode);
        # keying on it keeps multiple threshold cells per selection distinct.
        point = r.get("ratio") if r.get("ratio") is not None else r.get("threshold")
        return (r["dataset"], r["strategy"], point, r.get("selection"))

    merged: "Dict[Tuple[str, str, Optional[float], Optional[str]], Dict[str, Any]]" = {}
    order: List[Tuple[str, str, Optional[float], Optional[str]]] = []

    def slot(r: Dict[str, Any]) -> Dict[str, Any]:
        k = key(r)
        if k not in merged:
            merged[k] = {"dataset": r["dataset"], "strategy": r["strategy"],
                         "ratio": r.get("ratio"), "n": r.get("n")}
            if r.get("selection") is not None:
                merged[k]["selection"] = r["selection"]
            order.append(k)
        if "deviation_mode" in r and "deviation_mode" not in merged[k]:
            merged[k]["deviation_mode"] = r["deviation_mode"]
        for f in CARRY:
            if f in r and f not in merged[k]:
                merged[k][f] = r[f]
        return merged[k]

    for r in acc_rows:
        slot(r)["mean"] = r["mean"]
    for r in ttft_rows:
        row = slot(r)
        for f in TTFT_FIELDS:
            if f in r:
                row[f] = r[f]
    return [merged[k] for k in order]


def _check_correctness(model, tokenizer, dtype, cfg: dict) -> Optional[dict]:
    """Sanity gate: single-pass at r=1 must match a full forward (first example)."""
    blend_cfg = BlendConfig(check_layer=cfg["strategy"]["check_layer"],
                            deviation_mode=cfg["strategy"].get("deviation_mode", "v"))
    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            continue
        ex = _load_dataset(ds_name, ds_cfg["path"], 1)[0]
        full_prompt, chunk_strs = _build_prompts(ds_name, ex)
        store = ChunkKVStore(model.config.num_hidden_layers, dtype, device="cpu")
        ok, diff = check_r1_matches_full_forward(
            model, tokenizer, chunk_strs, store, blend_cfg,
            suffix=_suffix_of(full_prompt, chunk_strs))
        print(f"[eval] correctness: r1==full_forward={ok} "
              f"(max|logit diff|={diff:.4g}) on {ds_name} ex0")
        return {"dataset": ds_name, "r1_matches_full_forward": ok, "max_logit_diff": diff}
    return None


def run_combined(config_path: str, repeats: int = 3, warmup: int = 1,
                 check_correctness: bool = False) -> dict:
    """Load the model once; measure TTFT (timed, first) then accuracy (untimed)."""
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed = cfg.get("seed", 42)
    _set_seed(seed)
    print(f"[eval] seed={seed} config={config_path} repeats={repeats} "
          f"warmup={warmup} check_correctness={check_correctness}")

    model, tokenizer, device, dtype = _load_model(cfg)

    correctness = None
    if check_correctness:
        correctness = _check_correctness(model, tokenizer, dtype, cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _set_seed(seed)

    # TTFT FIRST on a clean device -- accuracy's long generations / cache growth
    # / clock drift then happen strictly after the timed region.
    print("[eval] phase 1/2: TTFT (timed)")
    ttft_rows = measure_ttft(model, tokenizer, dtype, cfg, repeats=repeats, warmup=warmup)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _set_seed(seed)

    print("[eval] phase 2/2: accuracy (untimed)")
    acc_rows = measure_accuracy(model, tokenizer, dtype, cfg)

    results = _merge(ttft_rows, acc_rows)
    summary = {
        "config": cfg,
        "environment": _environment_info(cfg),
        "ttft": {"repeats": repeats, "warmup": warmup, "max_new_tokens": 1,
                 "note": "cacheblend is the single-pass selective recompute; the "
                         "same implementation is timed (TTFT) and scored (accuracy), "
                         "so each row's acc/ttft come from one inference path"},
        "phase_order": ["ttft", "accuracy"],
        "isolation": ("TTFT measured first on a clean device; CUDA cache emptied and "
                      "RNG re-seeded before the accuracy phase; accuracy does "
                      "full-length generation but is never inside a timed region, so "
                      "it cannot affect the TTFT numbers."),
        "correctness": correctness,
        "results": results,
    }
    _write(cfg, summary, "_combined")

    print("\n=== accuracy + TTFT ===")
    if correctness is not None:
        print(f"  correctness: r1==full_forward={correctness['r1_matches_full_forward']} "
              f"(max|logit diff|={correctness['max_logit_diff']:.4g})")
    for row in results:
        ratio = row.get("ratio")
        if isinstance(ratio, (int, float)):
            budget_s = f"r={ratio:.2f}"          # fixed-budget (ratio) cell
        elif isinstance(row.get("threshold"), (int, float)):
            mb = row.get("mean_budget")
            mb_s = f"~{mb:.2f}" if isinstance(mb, (int, float)) else "?"
            budget_s = f"t={row['threshold']:.2f}({mb_s})"  # threshold cell + realized budget
        else:
            budget_s = "-"
        sel_s = row.get("selection") or "-"
        mean = row.get("mean")
        mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else "n/a"
        ttft = row.get("ttft_ms_median")
        ttft_s = f"{ttft:.1f}ms" if isinstance(ttft, (int, float)) else "n/a"
        spd = row.get("speedup_vs_recompute")
        spd_s = f"{spd:.2f}x" if isinstance(spd, (int, float)) else "n/a"
        print(f"  {row['dataset']:<14} {row['strategy']:<16} {sel_s:<13} {budget_s:<14} "
              f"acc={mean_s} ttft={ttft_s} ({spd_s}) n={row.get('n')}")
    return summary


def run_ttft_only(config_path: str, repeats: int = 3, warmup: int = 1) -> dict:
    """TTFT-only driver: time first-token latency, skip accuracy. Captures env."""
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed = cfg.get("seed", 42)
    _set_seed(seed)
    _bm, _pts, _ = _budget_points(cfg["strategy"])
    print(f"[eval] TTFT-only seed={seed} config={config_path} "
          f"repeats={repeats} warmup={warmup} {_bm}={_pts}")

    model, tokenizer, device, dtype = _load_model(cfg)
    results = measure_ttft(model, tokenizer, dtype, cfg, repeats=repeats, warmup=warmup)
    summary = {
        "config": cfg,
        "environment": _environment_info(cfg),
        "ttft": {"repeats": repeats, "warmup": warmup, "max_new_tokens": 1,
                 "note": "cacheblend is the single-pass selective recompute (the "
                         "same implementation scored for accuracy in run_combined)"},
        "results": results,
    }
    _write(cfg, summary, "_ttft")

    print("\n=== TTFT (ms) ===")
    for row in results:
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        sel_s = row.get("selection") or "-"
        print(f"  {row['dataset']:<14} {row['strategy']:<16} {sel_s:<13} {ratio_s:<8} "
              f"median={row['ttft_ms_median']:.1f}ms p90={row['ttft_ms_p90']:.1f}ms "
              f"speedup={row['speedup_vs_recompute']:.2f}x n={row['n']}")
    return summary


def run_eval(config_path: str) -> dict:
    """Accuracy-only driver (kept for the Modal app ``run_eval_modal.py``)."""
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    print(f"[eval] accuracy-only seed={cfg.get('seed', 42)} config={config_path}")
    model, tokenizer, device, dtype = _load_model(cfg)
    results = measure_accuracy(model, tokenizer, dtype, cfg)
    summary = {"config": cfg, "environment": _environment_info(cfg), "results": results}
    _write(cfg, summary, "")
    return summary


def _write(cfg: dict, summary: dict, suffix: str) -> None:
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + suffix
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=3, help="timed TTFT iterations per example")
    p.add_argument("--warmup", type=int, default=1, help="untimed TTFT warmup iterations")
    p.add_argument("--deviation-mode", choices=("v", "k", "kv"), default=None,
                   help="override strategy.deviation_mode from the YAML")
    p.add_argument("--check-correctness", action="store_true",
                   help="first verify single-pass r=1 reproduces a full forward")
    p.add_argument("--accuracy-only", action="store_true",
                   help="skip TTFT; run accuracy only (the Modal-app code path)")
    args = p.parse_args()

    config_path = args.config
    if args.deviation_mode is not None:
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        cfg.setdefault("strategy", {})["deviation_mode"] = args.deviation_mode
        config_path = str(Path(args.config).with_suffix(".override.yaml"))
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

    if args.accuracy_only:
        run_eval(config_path)
    else:
        run_combined(config_path, repeats=args.repeats, warmup=args.warmup,
                     check_correctness=args.check_correctness)


if __name__ == "__main__":
    main()
