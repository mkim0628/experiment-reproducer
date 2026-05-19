# CacheBlend on Qwen3.6-27B (Qwen3Next hybrid): structural analysis

Status: **prototype implemented; empirical validation deferred** because the
Modal workspace hit its billing cycle spend limit before the smoke could
run. See `workspace/code/run_hybrid_smoke_modal.py` for the prepared
entrypoints (`introspect` and the smoke proper).

This note explains, with citations to the actual transformers source, why
the original CacheBlend algorithm does not transfer cleanly to a
GatedDeltaNet+Attention hybrid like Qwen3.6-27B, what our prototype does
about it, and what we expect to see if/when the smoke is unblocked.

## Target model

**Qwen/Qwen3.6-27B** (released 2026-04-22). Architecture class
`Qwen3NextForCausalLM` in transformers 5.x. 64 decoder layers organized as
16 blocks of `[GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, GatedAttention]`,
giving:

* 48 / 64 = **75% linear-attention layers** (Gated DeltaNet, no K/V cache,
  recurrent state)
* 16 / 64 = **25% standard attention layers** with K/V cache, located at
  indices `{3, 7, 11, ..., 63}`

This is reflected in the config as `layer_types` = a 64-element list of
either `"linear_attention"` or `"full_attention"`. See
`transformers/models/qwen3_next/modeling_qwen3_next.py:821-825`.

## Why CacheBlend does not directly apply

CacheBlend's published algorithm
([paper](https://arxiv.org/abs/2405.16444),
[code](https://github.com/YaoJiayi/CacheBlend)) assumes:

1. **Every decoder layer carries K, V tensors** that can be precomputed
   per-chunk and concatenated. Qwen3.6 violates this on 75% of its layers;
   they carry a recurrent `state` that depends on every prior token
   sequentially and cannot be composed by concatenation.
2. **The input to each attention layer L is dominated by within-chunk
   context.** This is what makes the chunk-isolated K_pre / V_pre a good
   approximation to the K_new / V_new produced during the full-sequence
   prefill. On Qwen3.6, between consecutive attention layers there are
   **three GatedDeltaNet layers** whose `S_t = S_{t-1} + β_t (v_t -
   S_{t-1} k_t^T) k_t` recurrence mixes the entire prior context into
   the hidden state that feeds the next attention layer. So the input to
   attention layer N=3 already differs between chunk-isolated and full
   prefills by O(prefix-context) - much worse than for a pure
   transformer where the input only differs through layer-(N-1)'s
   attention pattern.
3. **TTFT savings come from skipping K, V projections on most tokens at
   most layers.** On Qwen3.6 the attention layers are only 25% of the
   stack and most of the prefill compute is in the GatedDeltaNet
   layers, which must still be run fully on the entire concatenated
   sequence. So even if quality held, the TTFT speedup ceiling is
   bounded by the fraction of prefill time spent in attention - well
   under 25% in practice because the MLP and DeltaNet branches dominate.

## What this prototype does

Two new files were added (see `workspace/code/cacheblend/hybrid.py` and
`workspace/code/tests/test_hybrid_layer_detection.py`):

1. `detect_attention_layers(model)` walks `model.model.layers` and returns
   the indices whose layer module exposes K/V projections. On Qwen3.6
   this returns the 16 full-attention indices; on Falcon-H1 it returns
   all 36 indices (Falcon-H1 places mamba and attention IN PARALLEL within
   each layer); on a pure-Mamba stack it returns the empty list.
2. `precompute_chunk_kv_hybrid` registers hooks **only on attention
   layers**, capturing K_pre (post-`k_norm` for Qwen3-family models that
   have one, post-`k_proj` otherwise) and V via `past_key_values`.
3. `cacheblend_generate_hybrid` follows the homogeneous variant's flow
   restricted to attention layers:
   * Run a chunks-only prefill via `model.model(input_ids=chunks)` with
     hooks capturing K_new / V_new at every attention layer. **Crucially,
     the GatedDeltaNet layers' state at the end of this prefill is the
     correct state for the chunks - it is left untouched in the cache
     so the downstream `model.generate(suffix+query, past_kv=...)` call
     decodes from the right linear-attn state.**
   * Pick HKVD indices at the first attention layer (`attn_layer_indices[0]`,
     which is layer 3 on Qwen3.6). Note that the homogeneous variant
     uses model-global layer 1, which on Qwen3.6 is a DeltaNet layer
     where V deviation is undefined.
   * For each attention layer, overwrite non-HKVD chunk positions in the
     prefill cache with the chunk-isolated stored K_pre / V_pre (RoPE-
     rotated at the chunk's absolute positions, using the model's own
     `apply_rotary_pos_emb` to respect Qwen3Next's partial-rotation
     RoPE which differs from Mistral's).
   * Decode the suffix+query with the blended cache.

This is a *quality probe*: it does NOT realize a TTFT saving because the
prefill is still run in full. If quality already collapses there's no
point chasing the more delicate version that skips chunk prefill on
attention layers.

## Three Qwen3-specific gotchas the prototype handles

These are exactly the bugs you would hit if you applied the homogeneous
CacheBlend code unchanged. The new module's commit message (`ba313e0`)
covers them too:

| What | Mistral / Llama | Qwen3 / Qwen3Next | Resolution |
|---|---|---|---|
| Stored K_pre = output of | `k_proj` | `k_norm(k_proj(...))` | `_k_capture_module` hooks `k_norm` when present |
| RoPE recovery uses | full-tensor rotation (`mistral.apply_rotary_pos_emb`) | partial-rotation: only the first `rotary_dim` dims (`qwen3_next.apply_rotary_pos_emb`) | `_resolve_apply_rotary` imports the model's own implementation |
| Check layer | layer 1 (`BlendConfig.check_layer`) | layer 3 (first `full_attention` in the stack) | `cacheblend_generate_hybrid` uses `attn_layer_indices[0]` and ignores `cfg.check_layer` |

The hybrid module also writes a smaller, sparser KV store (16 slots vs 64
for Qwen3.6) since DeltaNet layers contribute nothing.

## Expected smoke-run outcome (when budget allows)

5 wikimqa examples on Qwen/Qwen3.6-27B (A100-80GB, fp16, ~$0.50 first
run / ~$0.30 warm):

* `full_recompute`: this is the model's natural quality on the bundled
  wikimqa prompts. Baseline F1 expected in the 0.30-0.40 range based on
  Qwen3.6's reported MultiHop QA performance.
* `full_reuse_hybrid` (= ratio=0): KV from chunk-isolated prefills
  concatenated verbatim. **Predicted to collapse much more sharply than
  on Mistral-7B**, because the attention layer at index 3 reads from a
  hidden state that has already gone through three DeltaNet layers
  whose chunk-isolated state ≠ full-sequence state.
* `cacheblend_hybrid` at r=0.15: should partially recover toward
  `full_recompute`, with the size of the gap revealing whether HKVD
  selection can localize the divergence on a hybrid. **The qualitative
  test** is whether the gap shrinks meaningfully vs `full_reuse_hybrid`.
  If it does, CacheBlend retains some signal on hybrids - though the
  TTFT case is still not made. If it doesn't, the hybrid + DeltaNet
  recurrence eats the per-token selection entirely.

The reference reproduction's recipe on Mistral-7B gets
`full_reuse_generate F1 ≈ 0.215` vs `cacheblend_generate F1 ≈ 0.266` vs
`full_recompute F1 ≈ 0.271` at r=0.18 (see
`workspace/code/run_eval_modal.py:80-92` and the comments cite
"reference reuse=0.215 / cacheblend=0.266 / recompute=0.271 numbers on
local A100"). The gap there is small because Mistral's attention layers
see a well-localized input. On Qwen3.6 the gap is expected to be much
larger.

## How to actually run the smoke later

```bash
cd workspace/code

# 1. Cheap pre-flight: confirms layer-detection matches config.layer_types
#    and validates the model loads at all. ~$0.30 (mostly weight download).
modal run run_hybrid_smoke_modal.py::introspect

# 2. The smoke proper. ~$0.50 first run (weights cached after step 1),
#    ~$0.30 each subsequent run.
modal run run_hybrid_smoke_modal.py

# 3. Sweep a few ratios after a successful first smoke:
for r in 0.05 0.10 0.15 0.30 0.50; do
  modal run run_hybrid_smoke_modal.py --ratio "$r"
done
```

Output JSON lands on the `cacheblend-results` Volume at
`/root/results/hybrid_smoke_Qwen_Qwen3.6-27B.json` per run; the local
entrypoint also prints a summary table.

## Open questions left for the empirical phase

1. Does `model.generate` accept `past_key_values=blended_cache` correctly
   when the cache spans hybrid layer types? Qwen3Next's `Qwen3NextModel`
   uses `DynamicCache(config=self.config)` which is supposed to handle
   the mixed layout, but no eval has exercised it from a manually
   manipulated state. The introspect entrypoint also surfaces the cache
   class so this can be confirmed cheaply.
2. Does `Qwen/Qwen3.6-27B` ship a custom `modeling_qwen3_6.py` via
   `trust_remote_code=True` that overrides the in-tree `Qwen3NextModel`?
   If so the layer-detection still works (it walks the layer-list
   structure), but `_resolve_apply_rotary` would need a sniff at the
   module the model class lives in. Confirmed cheaply by the introspect
   entrypoint.
3. The HKVD selection currently uses a single check layer (the first
   attention layer = layer 3). With only 16 attention layers, the
   "schedule" hyperparameter from the paper (single_check vs every_layer)
   has more headroom; an every-layer schedule might recover more quality
   on hybrid. Out of scope for the smoke but useful for a follow-up if
   the smoke results justify continuing.
