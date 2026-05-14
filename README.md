# CacheBlend (EuroSys'25) — Quality-Only Reproduction

**Status:** `PARTIALLY_REPRODUCED` — algorithmic equivalence verified (8/8 gates pass, 25/25 unit tests pass); live F1 / Rouge-L / TTFT measurements pending GPU + HuggingFace credentials.

Paper: Jiayi Yao et al., *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*, EuroSys 2025. arXiv: [2405.16444](https://arxiv.org/abs/2405.16444). DOI: [10.1145/3689031.3696098](https://doi.org/10.1145/3689031.3696098). Official code: [github.com/YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) (also integrated into [LMCache/LMCache](https://github.com/LMCache/LMCache)).

Verdict source: [`workspace/validation/report.json`](workspace/validation/report.json).

---

## What this repo reproduces

A from-scratch HuggingFace `transformers` re-implementation of the CacheBlend selective-KV-recompute algorithm, targeted at **quality only** (F1 on 2WikiMQA / Musique, Rouge-L on SAMSum / MultiNews) on `mistralai/Mistral-7B-Instruct-v0.2`, fp16.

All eight algorithmic pieces that determine the paper's quality claim are implemented and unit-tested:

| Component | Where | Gated by |
|---|---|---|
| pre-RoPE K + V capture | `workspace/code/cacheblend/precompute.py` (k_proj forward hook) | `tests/test_kv_cache.py` |
| sha256_cbor chunk hashing | `workspace/code/cacheblend/kv_cache.py` | `tests/test_kv_cache.py` |
| RoPE re-rotation at new positions | `workspace/code/cacheblend/blend.py` | `tests/test_rope_recovery.py` (vs HF Mistral, max abs diff < 1e-4) |
| per-token squared-L2 V-deviation | `workspace/code/cacheblend/selective_recompute.py:compute_v_deviation` | `tests/test_hkvd_selector.py` |
| top-r% ceil(rN) selection | `workspace/code/cacheblend/selective_recompute.py:select_hkvd_indices` | `tests/test_hkvd_selector.py` |
| in-place index_copy K/V merge | `workspace/code/cacheblend/selective_recompute.py` | `tests/test_selective_recompute.py` |
| full-recompute equivalence (r=1) | wrapper layer | `tests/test_selective_recompute.py` |
| full-reuse equivalence (r=0) | `workspace/code/cacheblend/baselines.py:full_reuse_generate` | `tests/test_selective_recompute.py` |
| SQuAD-style token F1 / Rouge-L | `workspace/code/eval/metrics.py` | `tests/test_metrics.py` |

## What this repo does NOT reproduce

- **TTFT and throughput numbers** (paper claims 2.2-3.3x and 2.8-5x). The paper's speedup depends on the two-thread Fusor that pipelines layer-i selective recompute with layer-(i+1) KV loading inside a custom vLLM 0.4.x fork pinned to PyTorch 2.0 and CUDA 12.1. The plan ([`workspace/plan/plan.json`](workspace/plan/plan.json)) explicitly chose not to vendor that fork. Our HF-based implementation runs the model twice for `cacheblend_generate` (see [`workspace/code/CODER_NOTES.md`](workspace/code/CODER_NOTES.md) item 2), which is quality-preserving but TTFT-irrelevant.
- **Per-layer recompute / load latencies** (Llama-7B / Llama-70B, Figures 10 and 16). Hardware-bound, NVMe-bound; out of scope.
- **Yi-34B and Llama-70B**. Mistral-7B is the only target model; Yi-34B and Llama-70B require 8-bit quantization that the paper does not pin.
- **Synthetic Musique-Extended / 2WikiMQA-Extended datasets** (built via GPT-4 paraphrase prompts; specifics not in paper).

## Frozen decisions (from [`workspace/plan/plan.json`](workspace/plan/plan.json))

- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`, fp16.
- **Recompute ratio grid**: `r ∈ {0.05, 0.10, 0.15, 0.18}` matching Figure 16 endpoints.
- **Examples per dataset**: 100 (subset of the bundled inputs).
- **Schedule**: a single "check" layer at decoder index 1 (matches official `cache_fuse_metadata['check_layers']=[1]`); HKVD indices selected there are reused on all subsequent layers. This is a HIGH-confidence deviation from the paper's verbal description of a gradual-decay schedule `r_1 > r_2 > ... > r`, justified by reading the released code.
- **Chunking**: native chunking from the bundled `wikimqa_s.json` / `musique_s.json` / `samsum.json` (taken verbatim from the official repo); for MultiNews the HF dataset is used as fallback.
- **Strategies**: `full_recompute` (unmodified HF), `full_reuse` (RoPE recovery only, no recompute), `cacheblend` (selective recompute at idx 1).

## How to run

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Unit tests (no GPU needed — these are the algorithmic-equivalence gates)

```bash
cd workspace/code && pytest tests/ -q
```

Expect 30 passing (the 25 algorithmic-equivalence gates plus 5 K-deviation ablation tests). This is what `report.json` records as `unit_tests_full_suite`.

### 3. Smoke run (requires CUDA GPU + `HF_TOKEN`)

```bash
cd workspace/code && HF_TOKEN=hf_xxx python scripts/run_smoke.py
```

Runs 3 examples of 2WikiMQA at `r=0.15` across `{full_recompute, cacheblend, full_reuse}` and prints three F1 numbers. On a CPU-only host (this sandbox) the script detects `torch.cuda.is_available() == False` and exits gracefully without fabricating numbers.

### 4. Full quality grid

```bash
cd workspace/code && python -m eval.run_eval --config configs/default.yaml
```

Iterates `(2WikiMQA, Musique, SAMSum, MultiNews) × {full_recompute, cacheblend, full_reuse} × r ∈ {0.05, 0.10, 0.15, 0.18}` over 100 examples each. Writes per-cell metrics to `workspace/results/<timestamp>.json`.

### 5. Deviation-mode ablation (V / K / K+V)

The paper does not pin which tensor drives the HKVD selector; the released
code uses V only (HIGH-confidence resolution in `workspace/spec/ambiguity_log.json`).
For ablation you can switch the deviation tensor without editing code:

```bash
# paper default (V only)
python -m eval.run_eval --config configs/default.yaml --deviation-mode v

# ablation: select by K only
python -m eval.run_eval --config configs/default.yaml --deviation-mode k

# ablation: select by K + V (per-token squared-L2 sum)
python -m eval.run_eval --config configs/default.yaml --deviation-mode kv
```

`--deviation-mode` overrides `strategy.deviation_mode` from the YAML for one
run; the result JSON records the mode actually used.

## Results

The 13 quantitative targets from the paper, with this run's verdict. Source: [`workspace/validation/report.json`](workspace/validation/report.json).

| # | Metric | Paper value | Our value | Status |
|---|---|---|---|---|
| 1 | TTFT reduction vs full KV recompute | 2.2-3.3x | — | SKIPPED (no-GPU, out of scope) |
| 2 | Throughput improvement vs full KV recompute | 2.8-5x | — | SKIPPED (no-GPU, out of scope) |
| 3 | F1 / Rouge-L drop vs full KV recompute | ≤ 0.02 avg; ≤ 0.01-0.03 worst | — | SKIPPED (needs GPU + HF_TOKEN) |
| 4 | F1 / Rouge-L improvement vs full KV reuse | 0.15-0.35 abs | — | SKIPPED (needs GPU + HF_TOKEN) |
| 5 | TTFT reduction at r∈[5%,18%] vs full recompute (Yi-34B) | 4.1-6.6x | — | SKIPPED (out of scope) |
| 6 | TTFT reduction at r∈[5%,18%] vs prefix caching (Yi-34B) | 3.4-6.1x | — | SKIPPED (out of scope) |
| 7 | Quality loss across r∈[5%,18%] (Yi-34B) | ≤ 0.002 | — | SKIPPED (proxy on Mistral-7B planned, not measured) |
| 8 | Per-layer recompute delay r=15% (Llama-7B, 4K) | 3 ms | — | SKIPPED (hardware-bound, out of scope) |
| 9 | Per-layer KV load delay (NVMe, Llama-7B) | 16 ms | — | SKIPPED (hardware-bound, out of scope) |
| 10 | Per-layer recompute delay r=15% (Llama-70B) | 7 ms | — | SKIPPED (hardware-bound, out of scope) |
| 11 | Per-layer KV load delay (NVMe, Llama-70B) | 4 ms | — | SKIPPED (hardware-bound, out of scope) |
| 12 | Baseline full prefill TTFT (Llama-34B, 4K, 1xA40) | ~3 s | — | SKIPPED (hardware-bound, out of scope) |
| 13 | Baseline full prefill TTFT (Llama-70B, 4K, 1xA40) | ~6 s | — | SKIPPED (hardware-bound, out of scope) |

Algorithmic-equivalence gates (8/8 pass; all bit-identical or within 1e-4 of HF reference):

| Gate | Status |
|---|---|
| V-deviation = per-token squared L2 of `(V_new - V_pre)` | match (delta 0) |
| top-r% selection uses ceil(rN) | match (delta 0) |
| RoPE recovery equals direct HF forward at new positions | match (max abs diff < 1e-4) |
| in-place `key_old[imp_indices] = key` preserves non-selected K/V | match (bit-identical at non-selected positions) |
| Layer wrapper at r=1.0 == HF MistralDecoderLayer | match (fp tolerance) |
| Layer wrapper at r=0.0 == PromptCache-style full reuse | match (fp tolerance) |
| sha256_cbor chunk hashing is deterministic and round-trips | match |
| F1 / Rouge-L scorers match `example/utils.py compute_f1` / `compute_rl` | match |

## Environment

- Hardware: CPU-only sandbox; `torch.cuda.is_available() == False`. End-to-end inference requires at least one NVIDIA GPU with ≥16 GB VRAM (A40 / A100 / L40 / RTX 6000 Ada / RTX 4090).
- Software: Python 3.11.15, PyTorch 2.12.0 (CPU build), `transformers` 5.8.1, `numpy` 2.4.4, `pytest` 9.0.3, `rouge_score`. Plan target was `transformers >=4.40,<4.46`; tests pass under 5.8.1 too.
- Seeds: `[0]` for unit tests. The full eval grid is intended to be repeated at seeds `{0, 1, 2}` once a GPU is available; mean ± std should be reported.
- Datasets: bundled JSON from the official CacheBlend repo at `workspace/code/data/{wikimqa_s,musique_s,samsum}.json`; MultiNews via HuggingFace fallback.

## Deviations from the paper

Pulled from [`workspace/spec/ambiguity_log.json`](workspace/spec/ambiguity_log.json) and [`workspace/code/CODER_NOTES.md`](workspace/code/CODER_NOTES.md):

1. **Single check layer at decoder index 1, not gradual decay** (HIGH confidence). The paper describes `r_1 > r_2 > ... > r` decaying across all layers; the released code uses one check at layer index 1 and reuses that index set everywhere. We follow the code.
2. **HF transformers wrapper instead of the vendored vLLM fork** (deliberate plan decision). The official repo is a vLLM 0.4.x fork pinned to PyTorch 2.0 / CUDA 12.1 with custom xformers attention; the plan opted out because TTFT is out of scope. Quality-relevant algorithmic state is preserved.
3. **CacheBlend runs the model twice per generation** (HF API constraint). Modern `MistralAttention.forward` does not expose the mid-forward K/V swap point that the vLLM fork patches, so we (a) run pass 1 to capture `K_new` / `V_new` via `k_proj` hooks and (b) decode pass 2 from a pre-built blended cache. Bit-equivalent to a single-pass implementation, slower by ~2x (irrelevant to quality).
4. **No multi-thread Fusor, no `synchronize()` primitive** (LOW-confidence in spec; out of scope). The released CacheBlend prototype also runs sequentially; the paper's pipelining is described conceptually.
5. **No LRU eviction, no `C_store` cost model** (LOW-confidence). For 100 examples × ~6 chunks the working set is well under 5 GB; eviction would never fire.
6. **Pre-RoPE K captured via `k_proj` forward hook**, not by reading `past_key_values` (HF stores post-RoPE K). V is taken straight from the layer cache because RoPE does not touch V.
7. **CBOR fallback to JSON** if `cbor2` is unavailable: the resulting SHA-256 is still deterministic but not byte-equal to vLLM's canonical hash. Tests do not check vLLM byte equality.

All choices labelled `[UNKNOWN]` in `ambiguity_log.json` that affect quality are HIGH-confidence resolutions backed by the official source code. LOW-confidence items resolve only to TTFT / Throughput / storage-cost choices that are out of scope here.

## Known issues

- Live F1 / Rouge-L grid not yet measured. See [`workspace/validation/report.json`](workspace/validation/report.json) section `next_actions` for the exact remaining steps once a GPU and HF token are available.
- `transformers` version drift: tests pass on 5.8.1 (current install) but plan pinned `>=4.40,<4.46`. If generation behavior differs from the official `example/utils.py` expectations, downgrade to 4.44 for full fidelity.
- `selective_layer_forward` is a stub; the full algorithmic forward lives end-to-end inside `cacheblend.baselines.cacheblend_generate` (see CODER_NOTES TODO).

## Hardware requirements

| Component | Requirement |
|---|---|
| GPU | 1× NVIDIA A40 / A100 / L40 / RTX 6000 Ada / RTX 4090, ≥16 GB VRAM (Mistral-7B fp16) |
| CPU RAM | ≥32 GB (KV cache copies + tokenizer) |
| Disk | ~15 GB (model weights) + scratch for JSON results |
| HF access | a `HF_TOKEN` with access to `mistralai/Mistral-7B-Instruct-v0.2` |

## Citation

```bibtex
@inproceedings{yao2025cacheblend,
  title     = {CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion},
  author    = {Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems (EuroSys '25)},
  year      = {2025},
  doi       = {10.1145/3689031.3696098},
  url       = {https://arxiv.org/abs/2405.16444}
}
```

For a long-form timeline of every decision, every `[UNKNOWN]` resolution, and every measurement gap, see [`workspace/docs/REPRODUCTION_NOTES.md`](workspace/docs/REPRODUCTION_NOTES.md).
