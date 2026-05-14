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
    cacheblend_generate,
    full_recompute_generate,
    full_reuse_generate,
)
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.selective_recompute import BlendConfig
from eval.datasets import (
    Example,
    build_qa_prompt,
    build_summarization_prompt,
    load_musique,
    load_samsum,
    load_wikimqa,
)
from eval.metrics import compute_f1_max, compute_rouge_l


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
    raise ValueError(f"unknown metric {metric}")


def _build_prompts(dataset_name: str, ex: Example):
    if dataset_name in ("wikimqa", "musique"):
        return build_qa_prompt(ex.question, ex.contexts)
    if dataset_name == "samsum":
        dialogue = ex.metadata.get("input", ex.question)
        return build_summarization_prompt(dialogue, ex.contexts)
    raise ValueError(f"unknown dataset {dataset_name}")


def _load_dataset(name: str, path: str, n: int) -> List[Example]:
    if name == "wikimqa":
        return load_wikimqa(path, n)
    if name == "musique":
        return load_musique(path, n)
    if name == "samsum":
        return load_samsum(path, n)
    raise ValueError(f"unknown dataset {name}")


def run_eval(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    print(f"[run_eval] seed={cfg.get('seed', 42)} config={config_path}")
    print(f"[run_eval] config: {json.dumps(cfg, indent=2)}")

    model, tokenizer, device, dtype = _load_model(cfg)
    n_layers = model.config.num_hidden_layers
    max_new = cfg["generation"]["max_new_tokens"]
    ratios = cfg["strategy"]["recompute_ratios"]
    check_layer = cfg["strategy"]["check_layer"]

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        metric = ds_cfg["metric"]
        print(f"[run_eval] dataset={ds_name} n={len(examples)} metric={metric}")

        # full_recompute (no ratio dependence)
        store = ChunkKVStore(n_layers, dtype, device="cpu")
        scores_full = []
        for i, ex in enumerate(examples):
            full_prompt, _ = _build_prompts(ds_name, ex)
            pred = full_recompute_generate(model, tokenizer, full_prompt, max_new)
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
                max_new_tokens=max_new, suffix=suffix_text,
            )
            scores_reuse.append(_score(metric, pred, ex, tokenizer))
        results.append({"dataset": ds_name, "strategy": "full_reuse", "ratio": None, "mean": float(np.mean(scores_reuse)), "n": len(scores_reuse)})

        # cacheblend for each ratio
        for r in ratios:
            blend_cfg = BlendConfig(recompute_ratio=r, check_layer=check_layer)
            scores_cb = []
            for i, ex in enumerate(examples):
                _, chunk_strs = _build_prompts(ds_name, ex)
                full_prompt, _ = _build_prompts(ds_name, ex)
                suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]
                pred = cacheblend_generate(
                    model, tokenizer, chunk_strs, query="", store=store,
                    cfg=blend_cfg, max_new_tokens=max_new, suffix=suffix_text,
                )
                scores_cb.append(_score(metric, pred, ex, tokenizer))
            results.append({"dataset": ds_name, "strategy": "cacheblend", "ratio": r, "mean": float(np.mean(scores_cb)), "n": len(scores_cb)})
            print(f"  cacheblend r={r}: mean={np.mean(scores_cb):.3f}")

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
    args = p.parse_args()
    run_eval(args.config)


if __name__ == "__main__":
    main()
