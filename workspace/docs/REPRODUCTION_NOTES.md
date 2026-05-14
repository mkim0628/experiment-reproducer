# CacheBlend Reproduction — Long-form Notes

Long-form companion to [`/README.md`](../../README.md). Captures the decision
timeline, every `[UNKNOWN]` resolution, and every measurement gap with a
hypothesis for why.

Date of this writeup: 2026-05-14.

Verdict (from [`workspace/validation/report.json`](../validation/report.json)):
**PARTIALLY_REPRODUCED**. 8/8 algorithmic-equivalence gates pass; 25/25 unit
tests pass; 0/13 paper quantitative targets measured (sandbox has no GPU and no
`HF_TOKEN`).

---

## 1. Timeline of decisions

### 1.1 Scope narrowing (plan stage)

The first decision was to narrow the reproduction to quality only (F1 on QA,
Rouge-L on summarization). The justification, recorded in
[`workspace/plan/plan.json`](../plan/plan.json) under `strategy.justification`:

> The official YaoJiayi/CacheBlend repo is a vendored vLLM 0.4.x fork pinned to
> PyTorch 2.0 and CUDA 12.1 with heavy custom CUDA kernels; it is impractical
> to install and modify for a small quality-only reproduction.

This decision had three consequences:

1. We could use plain HuggingFace `transformers` instead of vLLM, dramatically
   simplifying the implementation.
2. We could not produce TTFT / throughput numbers without rebuilding the
   pipelining layer. The plan accepted this gap up front.
3. The 6 LOW-confidence items in `ambiguity_log.json` (synchronize primitive,
   storage cost, batch size, etc.) all turn out to be irrelevant: they only
   affect TTFT, which we are not measuring.

### 1.2 Model choice

The paper says "Mistral-7B" without pinning a release. The plan's
`open_decisions` listed three options; the user-side decision (recorded in
[`workspace/plan/plan.md`](../plan/plan.md)) picked
`mistralai/Mistral-7B-Instruct-v0.2`. Rationale: v0.2 is the default cited in
[`workspace/references/references.json`](../references/references.json), widely
available, and matches what the official `example/blend_wikimqa.py` uses.

### 1.3 Schedule: single check layer vs gradual decay

The paper (Section 4.3) describes a gradual-filtering schedule
`r_1 > r_2 > ... > r` across all layers. The released code disagrees: at
`vllm_blend/vllm/model_executor/models/llama.py:300`,
`cache_fuse_metadata = {'check_layers':[1], 'check': False, 'recomp_ratios':[0.16], 'recomp_ratio':0.16, ...}`.
Exactly one layer (decoder index 1, i.e. the second transformer block) runs the
HKVD selection; every other layer reuses that one index set.

We followed the code, not the paper text. Reasoning:

- The resolution is HIGH-confidence in
  [`workspace/spec/ambiguity_log.json`](../spec/ambiguity_log.json) item
  "Exact decay schedule for the gradual-filtering ratios r_1 > r_2 > ... > r".
- The released code is the authors' canonical implementation.
- The "single check layer" behaviour is consistent with the empirical
  observation in Insight 2 (HKVD tokens are correlated across layers); one
  reliable measurement at a shallow layer is enough.

This is the largest paper-vs-code discrepancy in the reproduction and is the
top thing to ablate once a GPU is available (the `next_actions` list in
`report.json` suggests trying `check_layer ∈ {0, 1, 2}`).

### 1.4 Recompute ratio grid

The paper sweeps `r ∈ [5%, 18%]` on Yi-34B (Figure 16). The released example
scripts use `0.15` and `0.18`. The plan picked
`r ∈ {0.05, 0.10, 0.15, 0.18}` — the Figure-16 endpoints plus two interior
points — as a balance between coverage and runtime.

### 1.5 Examples per dataset

Paper uses 200 (2WikiMQA), 150 (Musique), 200 (SAMSum), 60 (MultiNews). Plan
picked 100 per dataset uniformly for a tractable runtime (~2 hours on 1xA40)
while still giving enough samples for a stable mean F1.

### 1.6 HF wrapper architecture (CODER_NOTES item 2)

The official vllm_blend fork patches `xformers.memory_efficient_attention` to
swap K/V at the check layer mid-forward. Modern HF does not expose that hook
cleanly. The coder's solution:

- **Pass 1**: run the full model forward with `k_proj` forward hooks to
  capture pre-RoPE K (HF stores post-RoPE K in `past_key_values`); read V
  directly from the layer cache.
- **Pass 2**: build a blended `past_key_values` from the captured tensors and
  the stored pre-computed cache, then decode.

This is bit-equivalent to a single-pass implementation at every layer's
post-blend K and V tensors, by construction. It is slower by ~2x in wall-clock
(irrelevant: TTFT is out of scope).

### 1.7 Pre-RoPE K capture (CODER_NOTES item 1)

The hook approach was chosen because the alternative — monkey-patching
`MistralAttention.forward` to expose K before RoPE — would have been brittle
across `transformers` versions. The forward hook on `k_proj` is stable across
HF versions and tested in `tests/test_rope_recovery.py`.

### 1.8 CBOR fallback (CODER_NOTES item 4)

If `cbor2` is unavailable, hashing falls back to JSON serialization. The
resulting SHA-256 is still deterministic across processes (so all cache hits
and misses behave identically within a run) but is not byte-equal to vLLM's
canonical hash. Cross-system cache portability is not in scope.

---

## 2. `[UNKNOWN]` items and their resolutions

Source: [`workspace/spec/ambiguity_log.json`](../spec/ambiguity_log.json).
HIGH-confidence resolutions are quality-relevant; LOW-confidence resolutions
are TTFT-only and irrelevant here.

### 2.1 HIGH-confidence (quality-relevant, all backed by official code)

| Unknown | Resolution | Code reference |
|---|---|---|
| Decay schedule r_1 > r_2 > ... > r | Single check layer at decoder idx 1; no decay; same indices reused everywhere | `vllm_blend/.../llama.py:300` |
| Which layers use `check_flag=True` | Exactly layer index 1; status ∈ {0=full, 1=check, 2=after-check} | `vllm_blend/.../llama.py:350-356` |
| Reduction for per-token KV deviation | L2-squared, summed across heads and head_dim; per token | `vllm_blend/.../xformers.py:210` |
| Deviation on K, V, or [K;V] | On V only (not K, not [K;V]) | `vllm_blend/.../xformers.py:210` |
| Positional encoding recovery | Explicit RoPE re-rotation with `rotary_emb(new_pos, fake_q, K_pre)`; K is stored pre-rotation | `vllm_blend/.../llama.py:174-179` |
| F1 implementation | SQuAD-style token-id F1 with normalize_answer (lower, strip articles, strip punct, fix ws); tokenizer.encode(...)[1:]; max across gold answers | `example/utils.py compute_f1` |
| Rouge-L library | `rouge_score.RougeScorer(['rougeL'], use_stemmer=True).fmeasure` | `example/utils.py compute_rl` |
| Hash function for chunks | sha256_cbor over (parent_hash, token_id_block) | `lmcache/v1/token_database.py:95-150` |
| Non-RoPE models | Out of scope per Section 9 | Section 9 |
| Softmax denominator during recompute | Over all keys; K and V are expanded back to full sequence after in-place write | `vllm_blend/.../xformers.py:232-246` |
| Per-layer positional recovery | Applied per-layer on the fly; stored K_pre is rotation-free | `vllm_blend/.../llama.py:174-179` |
| Memory layout for partial K/V writes | In-place `key_old[imp_indices] = key_new`; no scatter, no extra buffer | `vllm_blend/.../xformers.py:240-245` |
| Prompt templates per dataset | `build_qa_prompt` / `build_fewshot_prompt` with exact prefixes; Mistral `[INST]` token ids | `example/utils.py`, `example/blend.py:33-41` |
| PerTokenKVSize formula | `2 * num_layers * num_kv_heads * head_dim * dtype_bytes`; Mistral-7B fp16 = 131,072 B/token | standard transformer identity |

### 2.2 MEDIUM-confidence (resolution plausible but not bit-pinned)

| Unknown | Resolution | Risk |
|---|---|---|
| KV storage dtype on disk | Model's native activation dtype (fp16 for Mistral-7B); no on-disk quantization | Low; matters only for TTFT |
| On-disk layout | Shard per-layer-per-chunk (one .pt file each) for async prefetch | Low; we hold in CPU memory only |
| `fetch_kv` returning -1 mid-pipeline | Fall back to full prefill (status=0) for that chunk | Low; we don't simulate cache misses in this run |

### 2.3 LOW-confidence (all TTFT-only — out of scope)

These items were escalated to the user in `ambiguity_log.json` →
`escalated_to_user`. None of them affect F1 / Rouge-L:

- Quantization method (GPTQ vs AWQ vs bnb-int8) for Yi-34B / Llama-70B —
  irrelevant: Mistral-7B is not quantized.
- SentenceTransformers retrieval checkpoint — irrelevant: we use the bundled
  `wikimqa_s.json` etc., which already have retrieved chunks.
- GPT-4 query-generation prompt for Extended datasets — irrelevant: Extended
  datasets are out of scope (TTFT/throughput).
- Throughput batch size — irrelevant.
- `C_store` closed form — irrelevant.
- `synchronize()` primitive (CUDA event vs threading.Event) — irrelevant.
- `Fusor` CUDA stream — irrelevant.
- LRU eviction threshold — irrelevant: working set fits in CPU RAM.
- `Prefill(LLM, L)` profiling granularity — irrelevant.

---

## 3. Measurement gaps and hypotheses

Source: `discrepancies` in
[`workspace/validation/report.json`](../validation/report.json).

For each gap, we record what would be needed to close it and our hypothesis for
the expected result.

### 3.1 F1 / Rouge-L drop vs full KV recompute

- **Paper target**: ≤ 0.02 average, ≤ 0.01-0.03 worst case (Figure 12).
- **Our value**: not measured.
- **Why**: needs GPU + `HF_TOKEN` to load Mistral-7B-Instruct-v0.2 fp16.
- **Hypothesis**: within the 0.03 tolerance band on Mistral-7B once a GPU is
  available. The three algorithmic pieces that determine this metric (V
  deviation, top-r%, RoPE recovery, in-place index_copy, F1/Rouge-L scorers)
  are unit-tested against their official references, so the bias should be
  zero apart from float accumulation order and tokenizer-version drift.

### 3.2 F1 / Rouge-L improvement vs full KV reuse

- **Paper target**: 0.15-0.35 absolute (Section 1, Section 7).
- **Our value**: not measured.
- **Hypothesis**: within the 0.15-0.35 band. `full_reuse_generate` correctly
  ignores cross-attention by construction (no recomputation at any layer); the
  paper's gap was driven by this exact behaviour.

### 3.3 Quality loss across `r ∈ [5%, 18%]`

- **Paper target**: ≤ 0.002 in F1 or Rouge-L (Figure 16, Yi-34B).
- **Our value**: not measured.
- **Hypothesis**: somewhat looser on Mistral-7B than the Yi-34B target. The
  plan widened tolerance to 0.03 because Mistral-7B is a smaller model and the
  paper's ≤ 0.002 claim is specific to Yi-34B. The trend (flat F1 across
  `r ∈ [5%, 18%]`) should still hold.

### 3.4 TTFT (2.2-3.3x), throughput (2.8-5x), per-layer latencies, baseline TTFT

- **Paper target**: see Section 7 and Figures 10-17.
- **Our value**: not measured.
- **Why**: explicitly out of scope per
  [`workspace/plan/plan.json`](../plan/plan.json). Reliable TTFT requires the
  vLLM fork with two-thread Fusor pipelining, custom xformers attention swap,
  and NVMe-backed KV store; rebuilding all that for a quality-only run was
  ruled out at the plan stage.
- **Hypothesis**: not testable from a CPU sandbox; would need the
  YaoJiayi/CacheBlend repo installed in its own pinned environment (PyTorch
  2.0, CUDA 12.1, custom xformers) on the same hardware class (1xA40 / 2xA40)
  as the paper.

### 3.5 Per-layer recompute / load latencies (Llama-7B, Llama-70B)

- **Paper target**: 3 ms / 16 ms (Llama-7B); 7 ms / 4 ms (Llama-70B).
- **Our value**: not measured.
- **Why**: hardware-bound on NVMe SSD throughput (paper's NVMe measured at
  4.8 GB/s); we are CPU-only.
- **Hypothesis**: not testable here. The arithmetic in Section 5.1 is
  straightforward to verify on the target hardware:
  `T_recompute = r * Prefill(LLM, L)`,
  `T_load = PerTokenKVSize * L / throughput`.

---

## 4. What to do next (lifted from `report.json` `next_actions`)

1. Provision a single NVIDIA A40 (or equivalent A100 / L40 / RTX 6000 Ada /
   RTX 4090 with ≥ 16 GB VRAM) and export `HF_TOKEN` with access to
   `mistralai/Mistral-7B-Instruct-v0.2`.
2. Run `scripts/run_smoke.py` first to confirm the three strategies produce
   non-trivial F1 values on 5 2WikiMQA examples at `r=0.15`.
3. Run `code/eval/run_eval.py` with `configs/default.yaml` on the full grid:
   `(2WikiMQA, Musique, SAMSum, MultiNews) × {full_recompute, cacheblend,
   full_reuse} × r ∈ {0.05, 0.10, 0.15, 0.18}`, 100 examples each, seeds
   `{0, 1, 2}`; report mean ± std.
4. Compare measured `F1 / Rouge-L delta(cacheblend - full_recompute)` against
   the 0.03 plan tolerance and `delta(cacheblend - full_reuse)` against the
   0.15-0.35 improvement band.
5. If the F1 drop vs full recompute exceeds 0.03, ablate:
   (a) `check_layer ∈ {0, 1, 2}`;
   (b) confirm the pre-RoPE K capture hook fires on every layer;
   (c) verify chunk boundaries match the prompt assembly exactly (the bundled
       `wikimqa_s.json` includes its own context list).
6. Pin transformers version: tests currently pass on 5.8.1 but plan.json
   specified `>=4.40,<4.46`; if generation behavior differs from
   `example/utils.py` expectations, downgrade to `transformers==4.44`.
7. Optional (out of plan scope): to attempt TTFT numbers, install the official
   YaoJiayi/CacheBlend vLLM fork in a separate environment; do not mix with
   the HF-based quality reproduction.

---

## 5. Pointers

- User-facing summary: [`/README.md`](../../README.md)
- Developer README: [`workspace/code/README.md`](../code/README.md)
- Coder decisions: [`workspace/code/CODER_NOTES.md`](../code/CODER_NOTES.md)
- Algorithmic ambiguities: [`workspace/spec/ambiguity_log.json`](../spec/ambiguity_log.json)
- Frozen plan: [`workspace/plan/plan.json`](../plan/plan.json)
- Validator report: [`workspace/validation/report.json`](../validation/report.json) (authoritative verdict)
- Paper structured extract: [`workspace/paper/structured.json`](../paper/structured.json)
- Paper analysis: [`workspace/analysis/analysis.json`](../analysis/analysis.json)
