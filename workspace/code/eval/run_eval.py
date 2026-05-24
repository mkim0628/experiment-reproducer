"""Top-level eval driver.

Iterates ``(dataset x strategy x recompute_ratio)`` from a YAML config and
writes ``workspace/results/<run_id>.json``. Each row records the dataset,
strategy, ratio (when applicable), per-example metric, and aggregate mean.

Run as:
    python -m eval.run_eval --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from cacheblend.baselines import (
    full_recompute_generate,
    full_reuse_generate,
)
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.selective_recompute import BlendConfig
from cacheblend.single_pass import cacheblend_selective_generate
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
    if name == "wikimqa":
        return load_wikimqa(path, n)
    if name == "musique":
        return load_musique(path, n)
    if name == "samsum":
        return load_samsum(path, n)
    if name == "hotpotqa":
        return load_hotpotqa(path, n)
    if name == "multihop_rag":
        return load_multihop_rag(path, n)
    if name == "hover":
        return load_hover(path, n)
    if name == "multinews":
        return load_multinews(path, n)
    if name == "nq_dpr":
        return load_nq_dpr(path, n)
    raise ValueError(f"unknown dataset {name}")


def measure_accuracy(model, tokenizer, dtype, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accuracy phase: full-length generation + metric per (dataset, strategy, ratio).

    Takes an already-loaded model/tokenizer so a combined driver
    (see ``eval/run_combined.py``) can share a single model load across the
    accuracy and TTFT phases. This does full-length generation and is never
    timed, so in a combined driver it must run *after* the TTFT phase. Returns
    the per-row result dicts; the caller owns serialization.
    """
    n_layers = model.config.num_hidden_layers
    max_new = cfg["generation"]["max_new_tokens"]
    ratios = cfg["strategy"]["recompute_ratios"]
    check_layer = cfg["strategy"]["check_layer"]
    deviation_mode = cfg["strategy"].get("deviation_mode", "v")

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            print(
                f"[run_eval] dataset={ds_name}: file {ds_cfg['path']} not found, "
                "skipping (run scripts/download_extra_datasets.py to fetch)"
            )
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        metric = ds_cfg["metric"]
        # Per-dataset max_new_tokens override (e.g. MultiNews needs ~150-200
        # tokens for a multi-doc summary; QA needs ~32).
        ds_max_new = ds_cfg.get("max_new_tokens", max_new)
        print(
            f"[run_eval] dataset={ds_name} n={len(examples)} "
            f"metric={metric} max_new_tokens={ds_max_new}"
        )

        # full_recompute (no ratio dependence)
        store = ChunkKVStore(n_layers, dtype, device="cpu")
        scores_full = []
        for i, ex in enumerate(examples):
            full_prompt, _ = _build_prompts(ds_name, ex)
            pred = full_recompute_generate(model, tokenizer, full_prompt, ds_max_new)
            scores_full.append(_score(metric, pred, ex, tokenizer))
            if (i + 1) % 10 == 0:
                print(f"  full_recompute {i+1}/{len(examples)} mean={np.mean(scores_full):.3f}")
        results.append({"dataset": ds_name, "strategy": "full_recompute", "ratio": None, "mean": float(np.mean(scores_full)), "n": len(scores_full)})

        # full_reuse (no ratio dependence)
        scores_reuse = []
        for i, ex in enumerate(examples):
            _, chunk_strs = _build_prompts(ds_name, ex)
            suffix = (chunk_strs[-1].split("\n\n")[-1]) if False else ""
            # Use the last chunk's content as the suffix? No -- the prompt
            # template puts the query OUTSIDE the chunks (in the suffix).
            # Re-derive the suffix from the full prompt.
            full_prompt, _ = _build_prompts(ds_name, ex)
            suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]
            pred = full_reuse_generate(
                model, tokenizer, chunk_strs, query="", store=store,
                max_new_tokens=ds_max_new, suffix=suffix_text,
            )
            scores_reuse.append(_score(metric, pred, ex, tokenizer))
        results.append({"dataset": ds_name, "strategy": "full_reuse", "ratio": None, "mean": float(np.mean(scores_reuse)), "n": len(scores_reuse)})

        # cacheblend for each ratio
        for r in ratios:
            blend_cfg = BlendConfig(
                recompute_ratio=r,
                check_layer=check_layer,
                deviation_mode=deviation_mode,
            )
            scores_cb = []
            for i, ex in enumerate(examples):
                _, chunk_strs = _build_prompts(ds_name, ex)
                full_prompt, _ = _build_prompts(ds_name, ex)
                suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]
                pred = cacheblend_selective_generate(
                    model, tokenizer, chunk_strs, query="", store=store,
                    cfg=blend_cfg, max_new_tokens=ds_max_new, suffix=suffix_text,
                )
                scores_cb.append(_score(metric, pred, ex, tokenizer))
            results.append({
                "dataset": ds_name,
                "strategy": "cacheblend",
                "ratio": r,
                "deviation_mode": deviation_mode,
                "mean": float(np.mean(scores_cb)),
                "n": len(scores_cb),
            })
            print(f"  cacheblend r={r} dev={deviation_mode}: mean={np.mean(scores_cb):.3f}")
    return results


def run_eval(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    print(f"[run_eval] seed={cfg.get('seed', 42)} config={config_path}")
    print(f"[run_eval] config: {json.dumps(cfg, indent=2)}")

    model, tokenizer, device, dtype = _load_model(cfg)
    results = measure_accuracy(model, tokenizer, dtype, cfg)

    # ----- write JSON ---------------------------------------------------------
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{run_id}.json"
    summary = {"config": cfg, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_eval] wrote {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--deviation-mode",
        choices=("v", "k", "kv"),
        default=None,
        help="Override strategy.deviation_mode from the YAML (v=paper default).",
    )
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.deviation_mode is not None:
        cfg.setdefault("strategy", {})["deviation_mode"] = args.deviation_mode
        # Re-serialize a temporary config to pass downstream. We keep run_eval's
        # YAML-path interface stable by writing the override to a sibling tmp.
        tmp_path = Path(args.config).with_suffix(".override.yaml")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        run_eval(str(tmp_path))
    else:
        run_eval(args.config)


if __name__ == "__main__":
    main()
