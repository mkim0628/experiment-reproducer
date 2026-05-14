# Implementation Plan: CacheBlend Quality-Only Reproduction on Mistral-7B

## 1. Strategy decision

**From-scratch on top of HuggingFace transformers.**

Reference (for algorithmic details only): https://github.com/YaoJiayi/CacheBlend

Justification: The official YaoJiayi/CacheBlend release is a vendored vLLM ~0.4.x
fork pinned to PyTorch 2.0 and CUDA 12.1, with custom CUDA extensions and a
heavyweight model runner. The user has narrowed scope (gate 1) to F1 / Rouge-L
quality on Mistral-7B fp16 across three strategies (selective recompute, full
recompute, full reuse); TTFT and throughput are explicitly out of scope. That
removes the only reason to take the vLLM dependency. The CacheBlend algorithms
that actually matter for quality (V-deviation top-r%, single check layer at
decoder index 1, in-place index_copy of K/V, on-the-fly RoPE re-rotation) are
small and already pinned with HIGH-confidence resolutions in
`workspace/spec/ambiguity_log.json`. We can reimplement them as a thin wrapper
that monkey-patches `MistralAttention.forward` — on the order of ~200 LoC —
against `transformers>=4.40`, which is dramatically more tractable than
installing and modifying the vendored vLLM fork.

We are NOT trying to bit-match the throughput results, and we accept that not
using vLLM means we cannot reproduce TTFT numbers; that is the explicit user
trade.

## 2. Stack

- Language: Python 3.10
- Framework: PyTorch 2.2+ with HuggingFace transformers 4.40-4.45
- Key libs: torch, transformers, accelerate, datasets, rouge_score (>=0.1.2),
  numpy, tqdm, pyyaml, pytest

## 3. Module layout

```
code/
├── cacheblend/
│   ├── __init__.py
│   ├── kv_cache.py            # ChunkKVStore + sha256_cbor hashing
│   ├── precompute.py          # offline per-chunk KV warm-up
│   ├── blend.py               # RoPE recovery (rotary_emb at new positions)
│   ├── selective_recompute.py # core: status 0/1/2, V-deviation top-r%, in-place index_copy
│   └── baselines.py           # full_recompute_generate, full_reuse_generate, cacheblend_generate
├── eval/
│   ├── datasets.py            # 2WikiMQA/Musique/SAMSum/MultiNews loaders + prompt templates
│   ├── metrics.py             # SQuAD-token F1 + rouge_score Rouge-L
│   └── run_eval.py            # (dataset x strategy x r) sweep driver
├── scripts/
│   └── run_smoke.py           # 5-example 2WikiMQA sanity bench
├── configs/
│   └── default.yaml           # single source of truth for run parameters
└── tests/
    ├── test_kv_cache.py
    ├── test_rope_recovery.py
    ├── test_hkvd_selector.py
    ├── test_selective_recompute.py
    └── test_metrics.py
```

Per-module purpose, public API, and which method_spec component(s) they
implement are listed in detail in `plan.json` (`modules` field).

## 4. Test plan

Each component of `method_spec.json` is pinned by at least one unit test:

- **KVCacheStore**: hash determinism, put/fetch round-trip, F1/Rouge-L numeric
  agreement with `rouge_score` library, normalize_answer behavior.
- **PositionalEncodingRecovery**: applying RoPE to a stored un-rotated K at new
  positions must equal a fresh HF Mistral forward at those positions (fp32,
  tol 1e-4); shape preservation.
- **HKVDSelector**: per-token squared L2 reduction over (head, head_dim);
  top-r% with `ceil` rule; deterministic ordering given fixed inputs.
- **SelectiveKVRecompute**: with recompute_ratio=1.0 the wrapper matches the
  unmodified HF layer output bit-close (sanity); in-place index_copy preserves
  non-selected positions; zero-recompute path matches pure-reuse baseline.
- **Fusor**: integration smoke test `smoke_5_examples_2wikimqa` runs the full
  CacheBlend pipeline on 5 examples and produces finite F1 > 0.

## 5. Reproduction targets

Carried forward from `method_spec.json`, restricted to the user-approved scope
(Mistral-7B fp16, quality-only):

1. **F1 / Rouge-L drop vs full KV recompute** on 2WikiMQA, Musique (F1) and
   SAMSum, MultiNews (Rouge-L): paper says <=0.02 average; <=0.01-0.03 worst
   case. Tolerance: 0.03.
2. **F1 / Rouge-L improvement vs full KV reuse**: paper says 0.15-0.35
   absolute (qualitative). Tolerance: qualitative range.
3. **Quality loss across r in [5%, 18%]**: paper reports <=0.002 for Yi-34B in
   Figure 16; we substitute Mistral-7B per user scope. Tolerance: 0.03.

Out of scope (per user gate 1): all TTFT, throughput, and Yi-34B / Llama-70B
quantization targets.

## 6. Open decisions — RESOLVED at user gate 2

1. **Recompute-ratio grid r** → **`[0.05, 0.10, 0.15, 0.18]`** (4-point grid).
2. **Examples per dataset** → **100 per dataset**.
3. **HKVD selection schedule** → **Single check layer at decoder index 1**
   (matches official code).
4. **Mistral-7B-Instruct version** → **mistralai/Mistral-7B-Instruct-v0.2**.
5. **Number of context chunks per query** → **Native dataset chunking** (use
   whatever count is in the bundled `inputs/wikimqa_s.json` /
   `inputs/musique_s.json`; for SAMSum/MultiNews use the dataset's native
   chunking).

## 7. Build order

1. Scaffold `code/` tree, `configs/default.yaml`, requirements, pytest harness.
2. Implement `code/eval/datasets.py` with the exact `build_qa_prompt` /
   `build_fewshot_prompt` templates from `example/utils.py`; verify against
   bundled `wikimqa_s.json`.
3. Implement `code/eval/metrics.py` (SQuAD-style F1 + rouge_score Rouge-L) and
   `tests/test_metrics.py`; gate.
4. Implement `code/cacheblend/kv_cache.py` (sha256_cbor + in-memory CPU store)
   and `tests/test_kv_cache.py`; gate.
5. Implement `code/cacheblend/precompute.py`: load Mistral-7B-Instruct-v0.2
   fp16, run per-chunk forward, capture pre-RoPE K and V per layer; sanity
   check shapes (num_kv_heads=8, head_dim=128, num_layers=32).
6. Implement `code/cacheblend/blend.py` (RoPE recovery) and
   `tests/test_rope_recovery.py`; gate against a direct HF forward at the new
   positions.
7. Implement V-deviation + top-r% selection in
   `code/cacheblend/selective_recompute.py`; gate on
   `tests/test_hkvd_selector.py`.
8. Implement the layer-wrapper `selective_layer_forward` and `blend_forward`;
   gate on `tests/test_selective_recompute.py` (recompute_ratio=1.0 must
   reproduce HF output bit-close).
9. Implement `code/cacheblend/baselines.py`: `full_recompute_generate`
   (unmodified HF) and `full_reuse_generate` (RoPE recovery only, no
   recompute).
10. Wire `code/scripts/run_smoke.py` to run 5 examples on 2WikiMQA across all
    three strategies; integration gate.
11. Implement `code/eval/run_eval.py`: iterate (dataset x strategy x r) grid
    from `configs/default.yaml`, write JSON to `workspace/results/`.
12. Resolve open_decisions with the user; freeze `configs/default.yaml`; run
    the full quality grid; hand results to the validator agent.
