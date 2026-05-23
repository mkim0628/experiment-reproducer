"""Validate the single-pass selective recompute and report its real TTFT.

Three things, all sharing one model load:

1. Forward correctness (the rigorous test) -- for each example, run single-pass
   at r=1.0 and a plain full forward over the *identical* ``full_ids``, and
   compare the first-token logits. At r=1 every token is "active", so the custom
   selective forward must reduce to an ordinary full forward; matching argmax
   (and a tiny max-logit-diff) proves the hand-written attention/RoPE/MLP wiring
   is correct, independent of tokenization or long-decode fp16 drift.
2. Quality cross-check -- single-pass F1 vs the two-pass ``cacheblend_generate``
   F1 at the config's recompute ratios (close, not identical: see
   cacheblend/single_pass.py).
3. The deliverable -- single-pass TTFT vs full_recompute, i.e. the paper-style
   selective-recompute speedup that the two-pass path could not show.

Run as::

    python -m eval.validate_singlepass --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from cacheblend.baselines import _tok_ids, cacheblend_generate, full_recompute_generate
from cacheblend.kv_cache import ChunkKVStore
from cacheblend.selective_recompute import BlendConfig
from cacheblend.single_pass import (
    _selective_prefill,
    cacheblend_selective_generate,
    prepare_selective_inputs,
)
from eval.run_eval import _build_prompts, _load_dataset, _load_model, _score, _set_seed
from eval.run_ttft import _agg, _time_call


@torch.no_grad()
def validate_singlepass(
    model, tokenizer, dtype, cfg: dict, repeats: int = 3, warmup: int = 1,
    max_new_tokens: int = 32, compare_twopass: bool = False,
) -> dict:
    n_layers = model.config.num_hidden_layers
    check_layer = cfg["strategy"]["check_layer"]
    deviation_mode = cfg["strategy"].get("deviation_mode", "v")
    ratios = cfg["strategy"]["recompute_ratios"]

    results: List[Dict[str, Any]] = []
    for ds_name, ds_cfg in cfg["datasets"].items():
        if not os.path.exists(ds_cfg["path"]):
            print(f"[validate] dataset={ds_name}: {ds_cfg['path']} not found, skipping")
            continue
        examples = _load_dataset(ds_name, ds_cfg["path"], ds_cfg["n"])
        metric = ds_cfg["metric"]
        store = ChunkKVStore(n_layers, dtype, device="cpu")
        print(f"[validate] dataset={ds_name} n={len(examples)} metric={metric}")

        cfg_r1 = BlendConfig(recompute_ratio=1.0, check_layer=check_layer,
                             deviation_mode=deviation_mode)
        prompts = []  # cache (full_prompt, chunks, suffix) per example
        n_first_match = 0          # single-pass r=1 first-token argmax == full forward
        logit_diffs: List[float] = []
        sample = None
        f1_single: Dict[float, List[float]] = {r: [] for r in ratios}
        f1_two: Dict[float, List[float]] = {r: [] for r in ratios}

        for ei, ex in enumerate(examples):
            full_prompt, chunk_strs = _build_prompts(ds_name, ex)
            suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]
            prompts.append((full_prompt, chunk_strs, suffix_text))

            # ---- forward correctness on IDENTICAL ids ----
            # Build full_ids cheaply (tokenizer only) and run the reference full
            # forward FIRST, so its transient activations are freed before we
            # allocate the chunk KV cache + selective forward. Keeping the two
            # apart is what fits this on a 24 GB L4.
            chunk_id_list = [
                _tok_ids(tokenizer, c, add_special_tokens=(i == 0)).to(model.device)
                for i, c in enumerate(chunk_strs)
            ]
            suffix_ids = _tok_ids(tokenizer, suffix_text, add_special_tokens=False).to(model.device)
            full_ids = torch.cat(chunk_id_list + [suffix_ids], dim=1)
            if ei == 0:
                print(f"[validate] {ds_name} ex0: chunks={len(chunk_strs)} "
                      f"chunk_tokens={full_ids.shape[1] - suffix_ids.shape[1]} "
                      f"suffix_tokens={suffix_ids.shape[1]} total={full_ids.shape[1]}")
            try:
                ref_logits = model(full_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            except TypeError:  # older signature without logits_to_keep
                ref_logits = model(full_ids, use_cache=False).logits[:, -1, :]
            ref_arg = int(torch.argmax(ref_logits, dim=-1).item())
            ref_logits = ref_logits.float()
            torch.cuda.empty_cache()

            # Now build the chunk KV cache and run the selective forward (r=1 ->
            # every token active -> must reduce to the full forward above).
            fused_cache, full_ids2, C = prepare_selective_inputs(
                model, tokenizer, chunk_strs, store, suffix=suffix_text)
            _, sp_logits, _ = _selective_prefill(
                model, fused_cache, full_ids2, C, cfg_r1, build_cache=False)
            sp_arg = int(torch.argmax(sp_logits, dim=-1).item())
            n_first_match += int(ref_arg == sp_arg)
            logit_diffs.append(float((sp_logits.float() - ref_logits).abs().max().item()))
            if sample is None:
                sample = {
                    "first_token_match": ref_arg == sp_arg,
                    "max_logit_diff": logit_diffs[-1],
                    "ref_argmax_token": tokenizer.decode([ref_arg]),
                    "singlepass_argmax_token": tokenizer.decode([sp_arg]),
                }
            del fused_cache, ref_logits, sp_logits
            torch.cuda.empty_cache()

            # ---- quality at the operating ratios (single-pass; optional 2-pass) ----
            for r in ratios:
                bc = BlendConfig(recompute_ratio=r, check_layer=check_layer,
                                 deviation_mode=deviation_mode)
                t_single = cacheblend_selective_generate(
                    model, tokenizer, chunk_strs, query="", store=store,
                    cfg=bc, max_new_tokens=max_new_tokens, suffix=suffix_text)
                f1_single[r].append(_score(metric, t_single, ex, tokenizer))
                torch.cuda.empty_cache()
                if compare_twopass:
                    t_two = cacheblend_generate(
                        model, tokenizer, chunk_strs, query="", store=store,
                        cfg=bc, max_new_tokens=max_new_tokens, suffix=suffix_text)
                    f1_two[r].append(_score(metric, t_two, ex, tokenizer))
                    torch.cuda.empty_cache()

        # ---- TTFT: full_recompute vs single-pass selective at each ratio ----
        rec_ms: List[float] = []
        sel_ms: Dict[float, List[float]] = {r: [] for r in ratios}
        for full_prompt, chunk_strs, suffix_text in prompts:
            rec_ms.append(np.median(_time_call(
                lambda: full_recompute_generate(model, tokenizer, full_prompt, 1),
                repeats, warmup)))
            for r in ratios:
                bc = BlendConfig(recompute_ratio=r, check_layer=check_layer,
                                 deviation_mode=deviation_mode)
                sel_ms[r].append(np.median(_time_call(
                    lambda bc=bc: cacheblend_selective_generate(
                        model, tokenizer, chunk_strs, query="", store=store,
                        cfg=bc, max_new_tokens=1, suffix=suffix_text),
                    repeats, warmup)))

        rec_agg = _agg(rec_ms)
        rec_med = rec_agg["ttft_ms_median"]
        ratio_rows = {}
        for r in ratios:
            sel_agg = _agg(sel_ms[r])
            ratio_rows[f"{r}"] = {
                **sel_agg,
                "speedup_vs_recompute": rec_med / sel_agg["ttft_ms_median"],
                "f1_singlepass": float(np.mean(f1_single[r])),
                "f1_twopass": float(np.mean(f1_two[r])) if f1_two[r] else None,
            }

        ds_res = {
            "dataset": ds_name,
            "n": len(examples),
            "r1_first_token_matches_full_forward": f"{n_first_match}/{len(examples)}",
            "r1_max_logit_diff_median": float(np.median(logit_diffs)),
            "r1_max_logit_diff_max": float(np.max(logit_diffs)),
            "ttft_full_recompute_median_ms": rec_med,
            "cacheblend_selective": ratio_rows,
            "sample": sample,
        }
        results.append(ds_res)

        print(f"  r1 first-token match: {n_first_match}/{len(examples)}  "
              f"max|logit diff| med={np.median(logit_diffs):.4f} max={np.max(logit_diffs):.4f}")
        print(f"  full_recompute TTFT={rec_med:.1f}ms")
        for r in ratios:
            row = ratio_rows[f"{r}"]
            f2 = row["f1_twopass"]
            f2s = f"{f2:.3f}" if f2 is not None else "n/a"
            print(f"  selective r={r}: TTFT={row['ttft_ms_median']:.1f}ms "
                  f"({row['speedup_vs_recompute']:.2f}x)  "
                  f"f1_single={row['f1_singlepass']:.3f} f1_two={f2s}")

    return {"config": cfg, "results": results}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--compare-twopass", action="store_true",
                   help="also generate with the two-pass cacheblock for an F1 cross-check (heavier)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    model, tokenizer, device, dtype = _load_model(cfg)
    summary = validate_singlepass(model, tokenizer, dtype, cfg,
                                  repeats=args.repeats, warmup=args.warmup,
                                  max_new_tokens=args.max_new_tokens,
                                  compare_twopass=args.compare_twopass)
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_singlepass_validate"
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[validate] wrote {out_path}")


if __name__ == "__main__":
    main()
