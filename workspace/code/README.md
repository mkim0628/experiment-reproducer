# CacheBlend reproduction — developer README

A from-scratch HuggingFace `transformers` re-implementation of the CacheBlend
selective-KV-recompute algorithm on `mistralai/Mistral-7B-Instruct-v0.2` (fp16).
Measures BOTH quality (F1 / Rouge-L) and TTFT (time-to-first-token) from the
**same** single-pass implementation, so the accuracy-drop vs TTFT-saving
trade-off is read off one code path.

The top-level [`/README.md`](../../README.md) is the user-facing entry point and
records the reproduction verdict. This document is for developers extending the
code.

## Running the eval

There are three ways to run the grid, all driven from this directory
(`workspace/code/`) and all reusing the same core (`eval/run_eval.py`):

| Path | When to use | Cost model |
|---|---|---|
| **Modal** | The full grid; easiest cloud GPU. Picks the GPU per call. | per-second GPU; cheap with caching |
| **Cerebrium** | Cloud GPU as a persistent endpoint; async fire-and-forget. | per-second GPU; GPU fixed at deploy |
| **Local server** | You already have a CUDA GPU box. | your hardware |

All three need a HuggingFace token with access to the gated
`mistralai/Mistral-7B-Instruct-v0.2` repo. `smoke` = wikimqa, n=3, ratio 0.15
(pipeline sanity check, ~$0.05 on cloud L4); `full` = the YAML grid. Per
[`CLAUDE.md`](../../CLAUDE.md), **always smoke before a full sweep**.

### A. Modal (`run_eval_modal.py`)

One-time:

```bash
cd workspace/code
pip install modal
modal token new                                                  # auth
# HF token as a Modal secret named "huggingface" (keys HF_TOKEN + HUGGING_FACE_HUB_TOKEN):
modal secret create huggingface HF_TOKEN=hf_xxx HUGGING_FACE_HUB_TOKEN=hf_xxx
```

Run:

```bash
modal run run_eval_modal.py --mode smoke                 # wikimqa n=3, ~couple min on L4
modal run run_eval_modal.py --mode full                  # full grid from configs/default.yaml
modal run run_eval_modal.py --mode full --n 50           # override examples/dataset
modal run run_eval_modal.py --mode full --gpu A10G       # GPU override (default L4; see CLAUDE.md)
modal run run_eval_modal.py --mode full --deviation-mode v   # paper's V-deviation HKVD selector
```

Modal **picks the GPU per call** via `--gpu` (default `L4`). Weights are cached
on a Modal Volume so re-runs skip the ~14 GB download; one container loads the
model once and runs the whole grid. The local entrypoint prints a result table
and downloads the newest JSON to `workspace/results/` (plus `LATEST.json`).

Datasets needing huggingface.co / wiki egress (multinews, hotpotqa, hover,
multihop_rag) can be fetched on a CPU container (no GPU bill):

```bash
modal run run_eval_modal.py::download --which multinews --n 60
modal run run_eval_modal.py::download --which hotpotqa --n 200
```

### B. Cerebrium (`main.py` + `cerebrium.toml`)

Cerebrium is run as a **deployed persistent endpoint** invoked **async**, not
via `cerebrium run` — the ephemeral path can't carry this workload (4 MB upload
tar cap vs ~26 MB of data, no persistent volume so the 14 GB model re-downloads,
and a fixed ~5–7 min polling timeout shorter than cold-start+eval).

One-time:

```bash
cd workspace/code
pip install cerebrium
cerebrium login                                          # or set CEREBRIUM_SERVICE_ACCOUNT_TOKEN
cerebrium secrets set HF_TOKEN hf_xxx                    # gated Mistral repo
# datasets are NOT bundled (tar cap); upload to the volume once:
cerebrium cp data/wikimqa_s.json cacheblend-data/wikimqa_s.json   # smoke needs only this
cerebrium cp data/musique_s.json cacheblend-data/musique_s.json   # + samsum.json, nq_dpr.json for full
cerebrium deploy                                         # build + deploy on the toml's GPU
```

Run (async wrapper, then read results off the volume). One entry point
(`run_cerebrium`) measures BOTH accuracy and TTFT in one pass:

```bash
scripts/run_cerebrium.sh --mode smoke                    # async POST (acc + TTFT)
scripts/run_cerebrium.sh --mode full --n 200
scripts/run_cerebrium.sh --mode full --deviation-mode v
scripts/run_cerebrium.sh --mode smoke --check-correctness  # verify r=1==full forward first
SYNC=1 scripts/run_cerebrium.sh                          # wait for the JSON inline
DRY_RUN=1 scripts/run_cerebrium.sh --mode full           # print request, send nothing (no GPU)

cerebrium ls cacheblend-results/
cerebrium download cacheblend-results/<id>_combined.json
```

The GPU is set **at deploy time** in `cerebrium.toml` (`compute = "ADA_L4"`) —
Cerebrium has **no per-call GPU override**, so to change GPU edit the toml and
`cerebrium deploy` again. `min_replicas = 0` + `cooldown = 30` mean no idle GPU
billing after a run; the model is cached on `/persistent-storage`. In the managed
web env, `CEREBRIUM_SERVICE_ACCOUNT_TOKEN` / `CEREBRIUM_PROJECT_ID` are already
set, so the wrapper works without `cerebrium login`. Set `CEREBRIUM_WEBHOOK_URL`
to get a completion callback (otherwise the async run shows `processing` until
the result JSON lands on the volume).

### C. Local server (direct, no cloud)

For a machine that already has an NVIDIA GPU + CUDA:

```bash
cd workspace/code
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                                    # gated Mistral repo
# datasets: wikimqa_s/musique_s/samsum are bundled under data/; fetch the rest with
#   python scripts/download_extra_datasets.py --which all

python -m scripts.run_smoke                               # 3-example wikimqa sanity run
python -m eval.run_eval --config configs/default.yaml     # accuracy + TTFT -> workspace/results/<run_id>_combined.json
python -m eval.run_eval --config configs/default.yaml --check-correctness   # verify r=1==full forward first
python -m eval.run_eval --config configs/default.yaml --accuracy-only       # accuracy only (Modal-style)
python -m eval.run_eval --config configs/default.yaml --deviation-mode v    # paper HKVD selector
```

`eval.run_eval` is the single eval driver: it measures BOTH accuracy and TTFT
in one pass (single-pass cacheblend, one model load), TTFT timed first on a
clean device so the accuracy phase can't perturb it. `run_smoke` SKIPs
gracefully (exit 0) when there's no GPU / no weights — the component tests in
`tests/` still run CPU-only via `pytest`. Per-run JSON goes to the
`output.results_dir` in the YAML (`workspace/results/`).

## Module map

```
workspace/code/
  cacheblend/
    __init__.py
    kv_cache.py            # ChunkKVStore: per-(chunk_hash, layer_id) CPU store with sha256_cbor hashing
    blend.py               # PositionalEncodingRecovery: re-applies rotary_emb to stored pre-RoPE K
    selective_recompute.py # compute_kv_deviation, select_hkvd_indices, merge_selective_kv, BlendConfig
    precompute.py          # captures pre-RoPE K (via k_proj forward hook) and V per layer
    baselines.py           # full_recompute_generate / full_reuse_generate + fused-cache builder
    single_pass.py         # cacheblend_selective_generate: the single-pass selective recompute
  eval/
    datasets.py            # bundled-JSON loaders (wikimqa_s, musique_s, samsum) + official prompts
    metrics.py             # SQuAD-style token F1 + rouge_score Rouge-L (use_stemmer=True)
    run_eval.py            # the single eval driver: accuracy + TTFT per (dataset x strategy x ratio)
  scripts/
    run_smoke.py           # 3-example 2WikiMQA smoke run; SKIPs gracefully when no GPU
    run_cerebrium.sh       # the single Cerebrium wrapper -> run_cerebrium (accuracy + TTFT)
    run_latency.py         # Modal-path latency-model approximation (driven by run_latency_modal.py)
    download_extra_datasets.py / build_*.py / audit_datasets.py  # dataset fetch/build/audit
  configs/
    default.yaml           # single source of truth: model, dtype, dataset sizes, r-grid, output dir
  tests/
    test_kv_cache.py
    test_rope_recovery.py
    test_hkvd_selector.py
    test_selective_recompute.py
    test_metrics.py
    test_datasets.py
    test_audit.py
  data/
    wikimqa_s.json         # bundled verbatim from official CacheBlend repo
    musique_s.json
    samsum.json
  requirements.txt
```

### Component contracts

- **`cacheblend.kv_cache.ChunkKVStore`** — keyed by SHA-256 over CBOR(`(parent_hash, token_id_block)`); falls back to JSON serialization if `cbor2` is unavailable. The fallback is deterministic across processes but not byte-equal to vLLM's canonical hash. Stores per-layer K (pre-RoPE) and V on CPU; no eviction.
- **`cacheblend.blend.recover_rope_k(k_pre, new_positions, rotary_emb)`** — calls `rotary_emb(k_pre, new_positions)` to get `(cos, sin)`, then applies the rotation to `k_pre` and returns the rotated K. Tested against a direct HF Mistral forward at the new positions to max abs diff < 1e-4 in fp32.
- **`cacheblend.selective_recompute.compute_v_deviation(v_new, v_pre)`** — per-token squared L2 (`torch.sum((v_new - v_pre) ** 2, dim=[1, 2])`); matches `vllm_blend/vllm/attention/backends/xformers.py:210`. `compute_k_deviation` / `compute_kv_deviation` add the K and K+V variants; `BlendConfig.deviation_mode` (`v` / `k` / `kv`) selects which — the YAML default is `k`, switch to `v` to match the paper's measured numbers.
- **`cacheblend.selective_recompute.select_hkvd_indices(deviation, r)`** — `torch.topk(deviation, k=ceil(r * N)).indices.sort().values`; ascending sort is for deterministic test layout, not algorithmic correctness.
- **`cacheblend.single_pass.cacheblend_selective_generate`** — the single-pass selective recompute: one forward that recomputes only the HKVD chunk tokens + suffix and serves the rest from cache. Scored for accuracy AND timed for TTFT, so both come from one implementation. `check_r1_matches_full_forward` verifies the `r=1` reduction to a full forward (bit-exact first-token logits).

## How tests work (algorithmic gates)

Run `pytest workspace/code/tests/ -q` (currently 49 tests across 7 files, no GPU needed).

Each module ships with a dedicated test file that locks in one algorithmic
property. These are the same "gates" the validator checks in
[`workspace/validation/report.json`](../validation/report.json):

| Test file | Gate | What it pins down |
|---|---|---|
| `test_kv_cache.py` | sha256_cbor hashing, dtype preservation | hash is deterministic; put/fetch is bit-exact; miss returns `None` |
| `test_rope_recovery.py` | RoPE positional invariance (Appendix A) | re-rotation matches a direct HF forward at the new positions (fp32, < 1e-4) |
| `test_hkvd_selector.py` | V-deviation + top-r% rule | per-token squared L2; `ceil(rN)` selection on synthetic vectors with known ordering |
| `test_selective_recompute.py` | selective merge bounds | r=1.0 selects all (full recompute); r=0.0 selects none (no-op); in-place index_copy preserves unselected |
| `test_metrics.py` | scoring | `normalize_answer` strips articles+punct; F1 on canned pairs; Rouge-L on canned pairs |
| `test_datasets.py` | data ingestion | loaders parse the bundled JSON; QA and SAMSum prompts assemble to the exact official strings |
| `test_audit.py` | dataset audit | `scripts/audit_datasets.py` per-dataset auditors flag degenerate sets (e.g. all-SUPPORTED HoVer) on synthetic fixtures |

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

### 3. Selective-recompute forward (`cacheblend/single_pass.py`)

`cacheblend_selective_generate` is a **single forward**: layers `0..check_layer`
run over all tokens (to measure the fresh-vs-cached deviation and pick the
top-`r%` HKVD chunk indices), then deeper layers recompute q/k/v ONLY for those
HKVD chunk tokens + the suffix, serving every other chunk token from its
precomputed cache. The check-layer selection is:

```
# at the check layer:
deviation = compute_kv_deviation(K_new, K_pre, V_new, V_pre, mode)  # (N,)
hkvd = select_hkvd_indices(deviation, r)             # ceil(rN) indices

# deeper layers: active = hkvd (chunk) + suffix; attention queries are those
# active tokens; keys/values are blended (fresh at active positions, the
# precomputed cache elsewhere). No second pass, no per-layer re-selection.
```

Because it is one forward, the SAME call is scored for accuracy and timed for
TTFT (both in `eval.run_eval`) -- the paper's accuracy-vs-TTFT trade-off is
read off one implementation. Correctness anchor: at `r=1` every token is HKVD,
so the forward reduces to a plain full forward; `check_r1_matches_full_forward`
asserts its first-token logits are bit-identical to `model(full_ids)`.

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

(the `configs/default.yaml` grid — override per run with `--n` / `--ratios` /
`--deviation-mode`)

- Recompute-ratio grid: `r ∈ {0.05, 0.10, 0.15, 0.18}`.
- 200 examples per dataset (MultiNews 60, the paper's eval-set size).
- HKVD deviation tensor: `deviation_mode: k` is the YAML default (an ablation);
  switch to `v` to match the official vllm_blend release / the paper's numbers.
- Single check layer at decoder index 1.
- Native chunking from the bundled JSON (no Langchain re-chunk in this run).
- Greedy decoding (`do_sample=False`), matching `blend_wikimqa.py` /
  `blend_musique.py`.

See [`/README.md`](../../README.md) for the user-facing summary,
[`workspace/spec/ambiguity_log.json`](../spec/ambiguity_log.json) for every
`[UNKNOWN]` and how it was resolved, and
[`workspace/docs/REPRODUCTION_NOTES.md`](../docs/REPRODUCTION_NOTES.md) for the
long-form timeline.
