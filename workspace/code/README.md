# CacheBlend quality-only reproduction — developer README

A from-scratch HuggingFace `transformers` re-implementation of the CacheBlend
selective-KV-recompute algorithm on `mistralai/Mistral-7B-Instruct-v0.2` (fp16).
Quality only: F1 / Rouge-L. TTFT / throughput are out of scope.

The top-level [`/README.md`](../../README.md) is the user-facing entry point and
records the reproduction verdict. This document is for developers extending the
code.

## Module map

```
workspace/code/
  cacheblend/
    __init__.py
    kv_cache.py            # ChunkKVStore: per-(chunk_hash, layer_id) CPU store with sha256_cbor hashing
    blend.py               # PositionalEncodingRecovery: re-applies rotary_emb to stored pre-RoPE K
    selective_recompute.py # compute_v_deviation, select_hkvd_indices, BlendConfig, layer wrapper
    precompute.py          # captures pre-RoPE K (via k_proj forward hook) and V per layer
    baselines.py           # full_recompute_generate / full_reuse_generate / cacheblend_generate
  eval/
    datasets.py            # bundled-JSON loaders (wikimqa_s, musique_s, samsum) + official prompts
    metrics.py             # SQuAD-style token F1 + rouge_score Rouge-L (use_stemmer=True)
    run_eval.py            # iterates (dataset x strategy x recompute_ratio) and writes JSON
  scripts/
    run_smoke.py           # 3-example 2WikiMQA smoke run; SKIPs gracefully when no GPU
  configs/
    default.yaml           # single source of truth: model, dtype, dataset sizes, r-grid, output dir
  tests/
    test_kv_cache.py
    test_rope_recovery.py
    test_hkvd_selector.py
    test_selective_recompute.py
    test_metrics.py
    test_datasets.py
  data/
    wikimqa_s.json         # bundled verbatim from official CacheBlend repo
    musique_s.json
    samsum.json
  requirements.txt
```

### Component contracts

- **`cacheblend.kv_cache.ChunkKVStore`** — keyed by SHA-256 over CBOR(`(parent_hash, token_id_block)`); falls back to JSON serialization if `cbor2` is unavailable. The fallback is deterministic across processes but not byte-equal to vLLM's canonical hash. Stores per-layer K (pre-RoPE) and V on CPU; no eviction.
- **`cacheblend.blend.recover_rope_k(k_pre, new_positions, rotary_emb)`** — calls `rotary_emb(positions=new_positions, q=fake_q, k=k_pre)` and returns the rotated K. Tested against a direct HF Mistral forward at the new positions to max abs diff < 1e-4 in fp32.
- **`cacheblend.selective_recompute.compute_v_deviation(v_new, v_pre)`** — `torch.sum((v_new - v_pre) ** 2, dim=[1, 2])`; matches `vllm_blend/vllm/attention/backends/xformers.py:210` exactly.
- **`cacheblend.selective_recompute.select_hkvd_indices(deviation, r)`** — `torch.topk(deviation, k=ceil(r * N)).indices.sort().values`; ascending sort is for deterministic test layout, not algorithmic correctness.
- **`cacheblend.baselines.cacheblend_generate`** — see CODER_NOTES item 2 (runs the model twice; bit-equivalent to a single-pass implementation).

## How tests work (algorithmic gates)

Run `pytest workspace/code/tests/ -q` (currently 25 tests, ~4 s wall, no GPU needed).

Each module ships with a dedicated test file that locks in one algorithmic
property. These are the same "gates" the validator checks in
[`workspace/validation/report.json`](../validation/report.json):

| Test file | Gate | What it pins down |
|---|---|---|
| `test_kv_cache.py` | sha256_cbor hashing, dtype preservation | hash is deterministic; put/fetch is bit-exact; miss returns `None` |
| `test_rope_recovery.py` | RoPE positional invariance (Appendix A) | re-rotation matches a direct HF forward at the new positions (fp32, < 1e-4) |
| `test_hkvd_selector.py` | V-deviation + top-r% rule | per-token squared L2; `ceil(rN)` selection on synthetic vectors with known ordering |
| `test_selective_recompute.py` | layer wrapper bounds | r=1.0 reproduces unmodified HF layer; r=0.0 reproduces full-reuse; in-place index_copy preserves unselected |
| `test_metrics.py` | scoring | `normalize_answer` strips articles+punct; F1 on canned pairs; Rouge-L on canned pairs |
| `test_datasets.py` | data ingestion | loaders parse the bundled JSON; QA and SAMSum prompts assemble to the exact official strings |

If a test fails, the corresponding row in [`workspace/validation/report.json`](../validation/report.json) is the source of truth — update there too.

## How to extend to a different decoder-only model

The algorithm assumes:
1. RoPE positional encoding (paper Section 9 explicitly limits scope).
2. Standard decoder-only transformer (attention → MLP) with separate Q / K / V
   projections.
3. KV cache exposed at the layer level (HF `past_key_values` or equivalent).

Three monkey-patch points need updating for a new HF model `XForCausalLM`:

### 1. Pre-RoPE K capture (`precompute.py`)

Modern HF attention applies RoPE in `forward` before `past_key_values` is
populated, so reading `past_key_values` gives you **post-RoPE** K. We work
around this with a forward hook on `k_proj`:

```python
# in cacheblend/precompute.py
def _capture_k_proj(model):
    handles = []
    captured = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        def hook(mod, inp, out, idx=layer_idx):
            # out: (batch, seq_len, num_kv_heads * head_dim)
            captured[idx] = out.detach()
        handles.append(attn.k_proj.register_forward_hook(hook))
    return handles, captured
```

For a non-Mistral model, change `model.model.layers` to the model-specific
attribute path (e.g., `model.transformer.h` for GPT-NeoX-style models) and
adjust the `k_proj` name (e.g., `attention.wk` for some forks).

### 2. RoPE re-rotation (`blend.py`)

`recover_rope_k` calls the layer's `rotary_emb` directly. Different models
expose this differently:

- **Mistral / Llama (HF)**: `layer.self_attn.rotary_emb(value_states, position_ids)` returns `(cos, sin)`; then `apply_rotary_pos_emb(q, k, cos, sin)`.
- **Older Llama variants**: `rotary_emb(positions, q, k)` returns `(q_rot, k_rot)` directly (the vLLM fork's signature).
- **Yi**: identical to Llama.
- **Falcon / GPT-NeoX**: positional embedding is wrapped differently; you'll
  need to call the model's `apply_rotary_pos_emb` manually.

Adjust the helper for whichever signature your target model uses; the test in
`test_rope_recovery.py` will catch a mistake here because it compares against a
direct forward.

### 3. Layer wrapper / blend forward (`selective_recompute.py` and `baselines.py`)

`cacheblend_generate` currently uses a two-pass approach (see
[`CODER_NOTES.md`](CODER_NOTES.md) item 2): pass 1 captures `K_new`/`V_new` for
the full sequence; pass 2 builds a blended `past_key_values` and decodes. The
blended cache is constructed per-layer as:

```
# at the check layer (idx=1):
deviation = compute_v_deviation(V_new, V_pre)        # (N,)
imp = select_hkvd_indices(deviation, r)              # ceil(rN) indices
K_blend = K_pre_rotated.clone(); V_blend = V_pre.clone()
K_blend[..., imp, :] = K_new[..., imp, :]            # in-place
V_blend[..., imp, :] = V_new[..., imp, :]

# at every other layer: same `imp` indices, no recomputation of deviation
```

To extend to a different model:

- Confirm the per-layer K / V tensor shape returned by `past_key_values`. HF
  uses `(batch, num_kv_heads, seq_len, head_dim)`; some forks transpose.
- If the new model uses MQA / GQA with a different num_kv_heads layout, the
  index_copy axis still applies along `seq_len`.
- The check layer index is hardcoded to `1` in `BlendConfig.check_layer` to
  match the official `cache_fuse_metadata['check_layers']=[1]`. For very deep
  or very shallow models you may want to ablate this.

After the three patches, re-run `pytest workspace/code/tests/ -q`. The
`test_rope_recovery.py` and `test_selective_recompute.py` tests will catch
plumbing mistakes before you launch the full eval grid.

## Frozen decisions

- Recompute-ratio grid: `r ∈ {0.05, 0.10, 0.15, 0.18}`.
- 100 examples per dataset.
- Single check layer at decoder index 1.
- Native chunking from the bundled JSON (no Langchain re-chunk in this run).
- Greedy decoding (`do_sample=False`), matching `blend_wikimqa.py` /
  `blend_musique.py`.

See [`/README.md`](../../README.md) for the user-facing summary,
[`workspace/spec/ambiguity_log.json`](../spec/ambiguity_log.json) for every
`[UNKNOWN]` and how it was resolved, and
[`workspace/docs/REPRODUCTION_NOTES.md`](../docs/REPRODUCTION_NOTES.md) for the
long-form timeline.
