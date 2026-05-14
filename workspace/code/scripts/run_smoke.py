"""Minimal end-to-end smoke run.

Runs 3 examples on 2WikiMQA across the three strategies at r=0.15 and
prints one F1 per strategy. If the Mistral model is not available
(no HF_TOKEN, no cached weights, no GPU) the run is SKIPPED with a clear
message and the script exits 0.

Invoke as::

    python -m scripts.run_smoke           # from workspace/code/
    python scripts/run_smoke.py --smoke   # also accepts --smoke for the harness
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
from typing import Optional


_CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


def _try_load_model(model_name: str):
    """Return (model, tokenizer, device) or None on any failure."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        print(f"[smoke] transformers/torch import failed: {e}")
        return None

    if not torch.cuda.is_available():
        print("[smoke] no CUDA device available; CacheBlend smoke requires GPU.")
        return None

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()
        model.eval()
    except Exception as e:
        print(f"[smoke] could not load {model_name}: {e}")
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            print("[smoke] hint: set HF_TOKEN to a HuggingFace access token to download weights.")
        return None
    return model, tok, "cuda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="alias for the no-arg path")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument(
        "--wikimqa",
        default=str(_CODE_DIR / "data" / "wikimqa_s.json"),
    )
    args = parser.parse_args()
    _ = args.smoke  # accepted but unused

    print(f"[smoke] cwd={os.getcwd()} code_dir={_CODE_DIR}")
    print(f"[smoke] config: model={args.model} n={args.n} ratio={args.ratio} wikimqa={args.wikimqa}")

    loaded = _try_load_model(args.model)
    if loaded is None:
        print("[smoke] SKIPPED (model unavailable). Component tests still run via pytest.")
        sys.exit(0)
    model, tok, device = loaded
    print(f"[smoke] loaded {args.model} on {device}; num_hidden_layers={model.config.num_hidden_layers}")

    # --- Late imports (avoid touching transformers if smoke is skipping). ---
    from cacheblend.baselines import (
        cacheblend_generate,
        full_recompute_generate,
        full_reuse_generate,
    )
    from cacheblend.kv_cache import ChunkKVStore
    from cacheblend.selective_recompute import BlendConfig
    from eval.datasets import build_qa_prompt, load_wikimqa
    from eval.metrics import compute_f1_max
    import torch

    examples = load_wikimqa(args.wikimqa, args.n)
    store = ChunkKVStore(model.config.num_hidden_layers, torch.float16, device="cpu")
    blend_cfg = BlendConfig(recompute_ratio=args.ratio, check_layer=1)

    f1_full, f1_reuse, f1_cb = [], [], []
    for i, ex in enumerate(examples):
        full_prompt, chunk_strs = build_qa_prompt(ex.question, ex.contexts)
        suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]

        pred_full = full_recompute_generate(model, tok, full_prompt, max_new_tokens=32)
        pred_reuse = full_reuse_generate(model, tok, chunk_strs, query="", store=store,
                                         max_new_tokens=32, suffix=suffix_text)
        pred_cb = cacheblend_generate(model, tok, chunk_strs, query="", store=store,
                                      cfg=blend_cfg, max_new_tokens=32, suffix=suffix_text)
        f1_full.append(compute_f1_max(pred_full, ex.answers, tok))
        f1_reuse.append(compute_f1_max(pred_reuse, ex.answers, tok))
        f1_cb.append(compute_f1_max(pred_cb, ex.answers, tok))
        print(f"[smoke] ex {i+1}: full={f1_full[-1]:.2f} reuse={f1_reuse[-1]:.2f} cb={f1_cb[-1]:.2f} | pred_cb={pred_cb!r}")

    n = len(examples) or 1
    print(f"[smoke] F1_full={sum(f1_full)/n:.3f} F1_reuse={sum(f1_reuse)/n:.3f} F1_cacheblend={sum(f1_cb)/n:.3f}")


if __name__ == "__main__":
    main()
