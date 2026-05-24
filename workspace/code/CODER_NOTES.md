# Coder notes

Running log of implementation decisions and any deviations from the spec /
plan. Append-only.

## Implementation decisions

1. **Pre-RoPE K capture via `k_proj` forward hook.** Modern HuggingFace
   `MistralAttention.forward` immediately applies rotary embedding to `K`
   before it is exposed in `past_key_values`. To follow the official
   convention of storing un-rotated K, we register a forward hook on each
   layer's `k_proj` linear and grab its output (reshaped to
   `(batch, num_kv_heads, seq_len, head_dim)`). V is taken straight from the
   layer cache because RoPE does not touch V.

2. **Single-pass CacheBlend forward.** The official vllm_blend fork patches
   `xformers.memory_efficient_attention` to swap K/V at the check layer
   mid-forward. Modern HF does not expose that hook cleanly, so
   `cacheblend.single_pass.cacheblend_selective_generate` hand-writes the
   decoder forward (GQA + RoPE + blended KV + causal attention over the
   scattered active queries): layers `0..check_layer` run full to pick the
   HKVD tokens, then deeper layers recompute q/k/v only for those HKVD chunk
   tokens + the suffix and serve every other chunk token from cache. One
   forward, so the same call gives both accuracy and the paper's TTFT.
   `check_r1_matches_full_forward` asserts the `r=1` reduction to a full
   forward is bit-exact (first-token logits, max|diff|=0).

3. **Top-k via `torch.topk` then sort ascending.** `index_copy_` does not
   require sorted indices, but ascending order keeps the resulting K/V slice
   layout deterministic for tests.

4. **CBOR fallback.** If `cbor2` is unavailable we fall back to a JSON
   serialization that includes the parent-hash hex and the token list; the
   resulting SHA-256 is still deterministic across processes, just not
   bit-equal to the canonical vLLM hash. Tests do not check vLLM byte
   equality.

5. **Sampling.** All three generators use `do_sample=False` greedy decoding,
   which matches what blend_wikimqa.py / blend_musique.py do (no sampling
   args set).

## Assumptions made for `[UNKNOWN]` items

(All [LOW-CONF] entries in `workspace/spec/ambiguity_log.json` resolve only
to TTFT / Throughput / storage cost choices that are out of scope here.)

* `LRU eviction threshold`: not implemented; the in-memory dict has unbounded
  capacity. For a 100-example, 4-6 chunks/query eval the working set is well
  under 5 GB so eviction would never fire.
* `Synchronization primitive`: not implemented; the reproduction is
  sequential.
* `Storage cost C_store`: not implemented.

## Open TODOs (none blocking quality reproduction)

* The full selective-recompute forward is implemented end to end in
  `cacheblend.single_pass.cacheblend_selective_generate` (single pass; serves
  accuracy and TTFT from one path). The earlier `selective_layer_forward`
  stub has been removed.

## Spec deviations

None. All algorithmic choices (V-only deviation, single check layer at idx 1,
ceil(r*N) selection, in-place index_copy, RoPE recovery on the loaded K) match
the spec exactly.
