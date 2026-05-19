"""Modal smoke test: does CacheBlend transfer to Qwen3.6-27B (Qwen3Next)?

We use Qwen/Qwen3.6-27B as the test bed - a 27B dense hybrid model whose
64 layers form 16 blocks of [GatedDeltaNet x 3, GatedAttention x 1]. So
16/64 = 25% of layers carry a standard K/V cache; the remaining 75% are
linear-attention (Gated DeltaNet) layers with a recurrent state instead.

Structural expectations going in
--------------------------------
CacheBlend's chunk-isolated KV precompute assumes the input to attention
layer L is roughly the same whether chunk N was seen alone or as part of
a long concatenated prompt. On a pure transformer that holds because most
cross-token mixing happens within the chunk's attention pattern. On
Qwen3.6, between every pair of attention layers there are 3 sequential
GatedDeltaNet layers whose state mixes the *entire* prior context. So the
chunk-only V_pre is structurally a worse approximation to the full-prefill
V_new than on Llama / Mistral. We expect cacheblend_hybrid F1 to drop
substantially vs full_recompute, especially at low recompute_ratio.

What this script does
---------------------
1. Loads Qwen/Qwen3.6-27B on an A100-80GB container (fp16, ~54 GB weights).
2. Calls :func:`cacheblend.hybrid.detect_attention_layers` to identify the
   16 standard-attention layer indices.
3. On N wikimqa examples (default 5), runs three strategies and computes F1:

     * full_recompute    -- vanilla HF generate from the full prompt
     * cacheblend_hybrid -- attention-layer-only selective recompute at r=0.15
     * full_reuse_hybrid -- (cacheblend_hybrid at r=0, i.e. all chunk KV kept)

4. Prints a comparison table and writes the JSON to the results volume.

There is also an ``introspect`` local entrypoint that loads the model and
prints layer structure without running any examples - useful as a cheap
sanity check before paying for the full smoke (especially before paying for
the first weight download).

Cost budget
-----------
A100-80GB on Modal is roughly $3.40-$4.00/hr.
- First run: ~5 min weight download + 60 s load + ~10 s per example x 5 x 3
  = ~7 min => ~$0.40-0.45.
- Subsequent runs (weights cached on the cacheblend-hf-cache Volume):
  ~60 s load + 2.5 min eval = ~3.5 min => ~$0.20.

Per CLAUDE.md cost rules: this DOES upgrade past L4 (the project's default),
because the 27B fp16 weights need 54 GB of VRAM and A100-80GB is the
cheapest tier that fits. L4 (24 GB) and A10G (24 GB) would OOM at load.

Run:
    cd workspace/code
    modal run run_hybrid_smoke_modal.py                 # the smoke
    modal run run_hybrid_smoke_modal.py::introspect     # layer structure only
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import modal


CODE_DIR = pathlib.Path(__file__).resolve().parent
REMOTE_CODE_DIR = "/root/code"
HF_CACHE_DIR = "/root/.cache/huggingface"
RESULTS_DIR = "/root/results"

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


# Pin transformers >= 5.8 which ships qwen3_next (the architecture class
# used by Qwen3.6-27B). 5.8.1 is the same version the main eval uses.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==5.8.1",
        "accelerate>=0.30",
        "numpy<2",
        "tqdm",
        "sentencepiece",
        "protobuf",
        # causal-conv1d / mamba-ssm would speed up the GatedDeltaNet layers
        # but require an nvcc build at image-build time. Skip - slow path is
        # acceptable for a 5-example smoke.
    )
    .add_local_dir(
        str(CODE_DIR),
        REMOTE_CODE_DIR,
        ignore=["*.log", "__pycache__", "*.pyc", "results/", ".pytest_cache"],
    )
)

hf_cache_vol = modal.Volume.from_name("cacheblend-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cacheblend-results", create_if_missing=True)

app = modal.App("cacheblend-hybrid-smoke")


@app.cls(
    image=image,
    # A100-80GB: cheapest tier that fits Qwen3.6-27B fp16 (~54 GB weights
    # plus activations + KV cache for ~2k-token prompts). L4 / A10G OOM at
    # load. See CLAUDE.md "Modal GPU cost rules": upgrade past L4 only when
    # the workload actually requires it.
    gpu="A100-80GB",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=30 * 60,        # 30 min hard cap (first-run download dominates)
    scaledown_window=60,    # never keep the container warm after the call
    enable_memory_snapshot=True,
)
class HybridSmoke:
    @modal.method()
    def run(
        self,
        model_name: str = DEFAULT_MODEL,
        n_examples: int = 5,
        recompute_ratio: float = 0.15,
        dataset_path: str = "data/wikimqa_s.json",
        max_new_tokens: int = 32,
    ) -> dict:
        sys.path.insert(0, REMOTE_CODE_DIR)
        os.chdir(REMOTE_CODE_DIR)
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; aborting before billing GPU.")
        print(f"[modal] device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[modal] loading {model_name} (fp16) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, trust_remote_code=True,
        ).to("cuda")
        model.eval()
        print(
            f"[modal] loaded: arch={type(model).__name__} "
            f"num_hidden_layers={model.config.num_hidden_layers}"
        )

        # ---------------- layer structure introspection
        from cacheblend.hybrid import (
            cacheblend_generate_hybrid,
            detect_attention_layers,
            full_recompute_generate_hybrid,
        )
        from cacheblend.kv_cache import ChunkKVStore
        from cacheblend.selective_recompute import BlendConfig

        attn_indices = detect_attention_layers(model)
        layer_types = getattr(model.config, "layer_types", None)
        print(
            f"[modal] attention layers detected: {attn_indices} "
            f"({len(attn_indices)}/{model.config.num_hidden_layers} "
            f"= {len(attn_indices) / model.config.num_hidden_layers:.1%})"
        )
        if layer_types is not None:
            print(f"[modal] config.layer_types[:10]={layer_types[:10]} ... [-4:]={layer_types[-4:]}")
        if not attn_indices:
            return {
                "error": "no attention layers detected; model is pure SSM "
                "and CacheBlend is not applicable",
                "model": model_name,
                "num_hidden_layers": model.config.num_hidden_layers,
            }

        # ---------------- dataset
        from eval.datasets import build_qa_prompt, load_wikimqa
        from eval.metrics import compute_f1_max

        examples = load_wikimqa(dataset_path, n=n_examples)
        print(f"[modal] loaded {len(examples)} wikimqa examples")

        results_per_example = []
        sums = {"full_recompute": 0.0, "full_reuse_hybrid": 0.0, "cacheblend_hybrid": 0.0}
        per_strategy_errors = {"full_reuse_hybrid": 0, "cacheblend_hybrid": 0}

        for ei, ex in enumerate(examples):
            full_prompt, chunk_strs = build_qa_prompt(ex.question, ex.contexts)
            suffix_text = full_prompt[sum(len(c) for c in chunk_strs):]
            chunks = chunk_strs
            query_for_hybrid = suffix_text

            row = {"index": ei, "answers": ex.answers}

            # 1. full recompute
            pred_full = full_recompute_generate_hybrid(
                model, tokenizer, full_prompt, max_new_tokens=max_new_tokens
            )
            f1_full = compute_f1_max(pred_full, ex.answers, tokenizer)
            sums["full_recompute"] += f1_full
            row["full_recompute"] = {"pred": pred_full, "f1": f1_full}

            # 2. cacheblend_hybrid at r > 0
            store = ChunkKVStore(
                num_layers=len(attn_indices), dtype=torch.float16, device="cpu"
            )
            cfg = BlendConfig(
                recompute_ratio=recompute_ratio,
                check_layer=0,        # ignored by the hybrid runner; kept for API
                deviation_mode="v",   # match official CacheBlend
            )
            try:
                pred_cb = cacheblend_generate_hybrid(
                    model, tokenizer, chunks, query_for_hybrid, store, cfg,
                    attn_layer_indices=attn_indices, max_new_tokens=max_new_tokens,
                )
                f1_cb = compute_f1_max(pred_cb, ex.answers, tokenizer)
                sums["cacheblend_hybrid"] += f1_cb
                row["cacheblend_hybrid"] = {"pred": pred_cb, "f1": f1_cb}
            except Exception as e:  # noqa: BLE001 - record then continue
                per_strategy_errors["cacheblend_hybrid"] += 1
                row["cacheblend_hybrid"] = {"error": f"{type(e).__name__}: {e}"}

            # 3. full_reuse equivalent (cacheblend_hybrid at r=0).
            store2 = ChunkKVStore(
                num_layers=len(attn_indices), dtype=torch.float16, device="cpu"
            )
            cfg_reuse = BlendConfig(
                recompute_ratio=0.0,
                check_layer=0,
                deviation_mode="v",
            )
            try:
                pred_reuse = cacheblend_generate_hybrid(
                    model, tokenizer, chunks, query_for_hybrid, store2, cfg_reuse,
                    attn_layer_indices=attn_indices, max_new_tokens=max_new_tokens,
                )
                f1_reuse = compute_f1_max(pred_reuse, ex.answers, tokenizer)
                sums["full_reuse_hybrid"] += f1_reuse
                row["full_reuse_hybrid"] = {"pred": pred_reuse, "f1": f1_reuse}
            except Exception as e:  # noqa: BLE001
                per_strategy_errors["full_reuse_hybrid"] += 1
                row["full_reuse_hybrid"] = {"error": f"{type(e).__name__}: {e}"}

            print(
                f"  ex {ei}: full={f1_full:.3f}  "
                f"reuse={row['full_reuse_hybrid'].get('f1', 'ERR')}  "
                f"cb={row['cacheblend_hybrid'].get('f1', 'ERR')}"
            )
            results_per_example.append(row)

        n = len(examples)
        summary = {
            "model": model_name,
            "num_hidden_layers": model.config.num_hidden_layers,
            "attn_layer_indices": attn_indices,
            "n_examples": n,
            "recompute_ratio": recompute_ratio,
            "mean_f1": {k: v / n for k, v in sums.items()},
            "errors_per_strategy": per_strategy_errors,
        }
        print("[modal] summary:", json.dumps(summary, indent=2))

        out_path = pathlib.Path(RESULTS_DIR) / f"hybrid_smoke_{model_name.replace('/', '_')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"summary": summary, "per_example": results_per_example},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        results_vol.commit()
        return summary

    @modal.method()
    def introspect(self, model_name: str = DEFAULT_MODEL) -> dict:
        """Load the model and print its layer structure without running examples.

        Used as a cheap pre-flight before the actual smoke. On first call
        this still pays the weight download (~$0.10-0.40 depending on
        cache state); on subsequent calls it's ~60 s ($0.06).
        """
        sys.path.insert(0, REMOTE_CODE_DIR)
        os.chdir(REMOTE_CODE_DIR)
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; aborting.")
        print(f"[modal] device={torch.cuda.get_device_name(0)}")

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"[modal] config: type={type(cfg).__name__} arch={cfg.architectures}")
        layer_types = getattr(cfg, "layer_types", None)
        print(f"[modal] num_hidden_layers={cfg.num_hidden_layers}")
        if layer_types is not None:
            from collections import Counter
            print(f"[modal] layer_types Counter={Counter(layer_types)}")
            print(f"[modal] layer_types (first 16): {layer_types[:16]}")
        else:
            print("[modal] no config.layer_types - this isn't a hybrid model")

        # Now actually load to confirm our detection matches the config.
        print(f"[modal] loading {model_name} (fp16) for runtime introspection ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, trust_remote_code=True,
        ).to("cuda")
        from cacheblend.hybrid import detect_attention_layers
        detected = detect_attention_layers(model)
        print(f"[modal] detect_attention_layers -> {detected}")
        print(f"[modal] count: {len(detected)} / {cfg.num_hidden_layers}")

        per_layer = [(i, type(l).__name__) for i, l in enumerate(model.model.layers)]
        # only print a small slice to keep the log readable
        head = per_layer[:8]
        tail = per_layer[-4:]
        print(f"[modal] layer classes head={head} tail={tail}")
        return {
            "model": model_name,
            "num_hidden_layers": cfg.num_hidden_layers,
            "config_layer_types_count": (
                dict(__import__("collections").Counter(layer_types))
                if layer_types is not None else None
            ),
            "detected_attention_layers": detected,
            "detect_matches_config": (
                layer_types is not None
                and detected == [i for i, t in enumerate(layer_types) if t == "full_attention"]
            ),
        }


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    n: int = 5,
    ratio: float = 0.15,
):
    """Run the hybrid CacheBlend smoke test on Modal.

    Examples:
        modal run run_hybrid_smoke_modal.py
        modal run run_hybrid_smoke_modal.py --n 5 --ratio 0.30
    """
    runner = HybridSmoke()
    result = runner.run.remote(
        model_name=model,
        n_examples=n,
        recompute_ratio=ratio,
    )
    print("\n=== Local summary ===")
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def introspect(model: str = DEFAULT_MODEL):
    """Cheap pre-flight: load the model and print its layer structure.

    Run this BEFORE the full smoke if you've never run this model before -
    it confirms `detect_attention_layers` matches `config.layer_types`
    before we pay for the full eval. Cost on a warm volume: ~$0.06.
    """
    runner = HybridSmoke()
    result = runner.introspect.remote(model_name=model)
    print("\n=== Introspect result ===")
    print(json.dumps(result, indent=2, default=str))
