# Ambiguity Log — CacheBlend (arXiv 2405.16444)

Companion to `ambiguity_log.json`. Each entry below maps to a `resolutions[*]` entry in the JSON
sidecar. Citations point to either the paper, the official prototype
(https://github.com/YaoJiayi/CacheBlend) or its successor (https://github.com/LMCache/LMCache).

Legend: source kind = paper | official_code | third_party_code | related_paper | user.

---

## R1. Gradual filtering schedule r_1 > r_2 > ... > r — HIGH (official_code)

**Original unknown:** Section 4.3 describes a strictly-decreasing per-layer ratio schedule but
gives no closed form.

**Resolution:** The released reference implementation does NOT use a per-layer decaying
schedule. It uses a **single** check layer at decoder index 1 (the second layer); the HKVD
indices chosen there are reused for all subsequent layers. Default recompute ratio is
`r in {0.15, 0.16, 0.18}` depending on dataset.

**Evidence:** `vllm_blend/vllm/model_executor/models/llama.py:300`:
```python
self.cache_fuse_metadata = {"check_layers":[1],
                            "check": False,
                            "recomp_ratios":[0.16],
                            "recomp_ratio":0.16,
                            ...}
```
and `example/blend_wikimqa.py:107`: `cache_fuse_metadata['recomp_ratio'] = 0.18`.

---

## R2. Per-layer check_flag schedule — HIGH (official_code)

**Original unknown:** Which layers run `check_flag=True` vs `False`.

**Resolution:** Exactly **one** layer is check-on: layer index 1. All others run with
check_flag=False and reuse the index set selected at layer 1. The implementation encodes this
via a three-valued `status` (0=full prefill / 1=check layer / 2=after check).

**Evidence:** `llama.py:300, 350-356`.

---

## R3. Reduction in |KV_i[j] − KV_full_i[j]| — HIGH (official_code)

**Original unknown:** L1 vs L2 vs max-across-heads vs concat.

**Resolution:** Squared L2 summed across the head and head_dim axes, per token.

**Evidence:** `vllm_blend/vllm/attention/backends/xformers.py:210-211`:
```python
temp_diff = torch.sum((value[:-last_len,:,:] - value_old[:-last_len,:,:])**2, dim=[1,2])
top_indices = torch.topk(temp_diff, k=topk_num).indices
```

---

## R4. Deviation on K, V, or [K;V] — HIGH (official_code)

**Resolution:** **V only.** Only the value tensors are diffed for HKVD selection. K is not
involved in the diff metric.

**Evidence:** same line as R3 (`value` vs `value_old`).

---

## R5. Positional encoding recovery: explicit rotation vs dummy prefix — HIGH (official_code)

**Resolution:** **Explicit forward RoPE rotation at the new absolute positions.** The loaded K
is fed back through `self.rotary_emb` with `cache_fuse_metadata['org_pos']` (the new positions
of the concatenated context). A throwaway `fake_q` is provided to satisfy the kernel signature.
This implies the stored K is in pre-rotation form (or equivalently rotated at position 0 and
"unrotated" by reapplication at the new positions).

**Evidence:** `llama.py:174-179`:
```python
if status in [1,2]:
    if cache_fuse_metadata["fake_q"] is None:
        cache_fuse_metadata['fake_q'] = torch.rand_like(q)
    _, old_kv[0] = self.rotary_emb(cache_fuse_metadata['org_pos'],
                                   cache_fuse_metadata['fake_q'],
                                   old_kv[0])
```

---

## R6. F1 implementation — HIGH (official_code)

**Resolution:** **SQuAD-style token F1** over multisets of HF tokenizer ids, with normalization
= lower + strip articles (a/an/the) + strip punctuation + whitespace fix. Take max F1 across
gold answers. (Differs slightly from LongBench F1, which uses regex-tokenized strings.)

**Evidence:** `example/utils.py:26-74`.

---

## R7. Rouge-L library — HIGH (official_code)

**Resolution:** `rouge_score` Python package; `RougeScorer(['rougeL'], use_stemmer=True)`;
report f-measure.

**Evidence:** `example/utils.py:76-79`, `requirements.txt`.

---

## R8 (= R18 dup). Chunk hash function — HIGH (official_code, LMCache)

**Resolution:** vLLM's prefix block-hash via `vllm.utils.hashing.get_hash_fn_by_name('sha256_cbor')`
(falls back to `sha256_cbor_64bit` for older vLLM, then to builtin `hash()` with
`PYTHONHASHSEED` for old versions).

**Evidence:** `lmcache/v1/token_database.py:95-150`.

---

## R9 (= R20 dup). KV dtype on disk — MEDIUM (official_code)

**Resolution:** Model's native activation dtype (fp16 for Mistral-7B, bf16 for Yi-34B,
fp16/bf16 for Llama-70B). No on-disk quantization. PerTokenKVSize = 2 · num_layers ·
num_kv_heads · head_dim · sizeof(dtype).

**Evidence:** `xformers.py:229`: `cache_fuse_metadata["kv_cache_dtype"] = value.dtype`.

---

## R10 (= R22 dup). Non-RoPE / non-transformer models — HIGH (paper)

**Resolution:** Out of scope (Section 9 Future Work).

---

## R11. synchronize() primitive — LOW (assumption)

**Resolution:** Released prototype is **not actually pipelined**. ASSUMPTION for reproduction:
one non-default `torch.cuda.Stream` for the host→device KV copies, default stream for prefill,
`torch.cuda.Event` to gate per-layer hand-off; host-side `threading.Thread` for the fetch loop.
**Escalated.**

---

## R12. LRU eviction capacity — LOW (assumption)

**Resolution:** Not pinned. ASSUMPTION: size the store to fit the eval working set (~25 GB)
on the 1 TB NVMe; eviction effectively disabled.

---

## R13. C_store closed form — LOW (assumption)

**Resolution:** Not pinned. ASSUMPTION:
`C_store = PerTokenKVSize(LLM) · L · T · price_per_byte_per_time(device)`.
For the paper's single-NVMe + CPU-RAM setup this collapses to latency-only optimization.
**Escalated.**

---

## R14. 8-bit quantization backend for Yi-34B / Llama-70B — LOW (assumption)

**Resolution:** Not stated. The fork supports GPTQ / AWQ / SqueezeLLM / FP8 KV.
ASSUMPTION: bitsandbytes int8 or TheBloke GPTQ-8bit. **Escalated.**

---

## R15. SentenceTransformers checkpoint — LOW (assumption)

**Resolution:** Not named. ASSUMPTION: `sentence-transformers/all-mpnet-base-v2`. **Escalated.**

---

## R16. Throughput batch size (Figure 14) — LOW (assumption)

**Resolution:** Not stated. ASSUMPTION: vLLM continuous batching, `max_num_seqs=256`, request
rate as the swept variable. **Escalated.**

---

## R17. GPT-4 query-paraphrase prompt — LOW (assumption)

**Resolution:** Not stated. ASSUMPTION: `gpt-4-1106-preview`, temperature 0.7, paraphrase
prompt. **Escalated.**

---

## R19. Prompt templates per dataset — HIGH (official_code)

**Resolution:** Templates pinned by `example/blend_wikimqa.py` (and friends) and
`example/utils.py build_qa_prompt / build_fewshot_prompt`. Special tokens:
`[INST]` = `[733, 16289, 28793]`, `[/INST]` = `[733, 28748, 16289, 28793]` (Mistral tokens).

---

## R21. On-disk file layout — MEDIUM (official_code)

**Resolution:** Per-layer, per-chunk shards (one file per `(chunk_hash, layer_id)`).
LMCache's CacheEngineKey is `model@world@worker@chunk_hash@dtype@layer_id`.

**Evidence:** `lmcache/utils.py:373-475`.

---

## R23. Softmax denominator scope — HIGH (official_code)

**Resolution:** **Standard:** softmax is over ALL keys; K and V are expanded back to full
sequence length (selected positions overwritten with newly computed K/V) before xformers
`memory_efficient_attention` is called with `LowerTriangularFromBottomRightMask`.

**Evidence:** `xformers.py:240-246, 420-441`.

---

## R24. Per-layer positional recovery vs up-front — HIGH (official_code)

**Resolution:** **Per-layer on the fly**, inside `LlamaAttention.forward` when `status in {1,2}`.
The stored K is rotation-free; rotation is reapplied each layer.

**Evidence:** `llama.py:174-179`.

---

## R25. Partial K/V write-back layout — HIGH (official_code)

**Resolution:** **In-place fancy-indexed assignment** along the sequence axis:
`key_old[imp_indices] = key; value_old[imp_indices] = value`. Equivalent to
`index_copy_(0, imp_indices, key)`.

**Evidence:** `xformers.py:240-245`.

---

## R26. Behavior on fetch_kv == -1 — MEDIUM (official_code)

**Resolution:** Fall back to full prefill (status=0 path). Conceptually, if any chunk in the
request is missing, run normal full-recompute. The released code path supports this implicitly
through `cache_fuse_metadata['check'] = False`.

**Evidence:** `llama.py:330-356`.

---

## R27. Prefill(LLM, L) offline profile — LOW (assumption)

**Resolution:** Not implemented. ASSUMPTION: at batch_size=1, sweep `L in {512, 1024, 2048,
4096, 8192}`, fit `Prefill(LLM, L) = a·L + b·L²`. Persist (a, b).

---

## R28. PerTokenKVSize formula — HIGH (paper + standard)

**Resolution:** `2 · num_layers · num_kv_heads · head_dim · sizeof(dtype)`. Concrete values:
- Mistral-7B fp16 (32, 8, 128, 2): 131,072 B/token (128 KiB)
- Yi-34B fp16 (60, 8, 128, 2): 245,760 B/token (240 KiB) [bf16 same byte count]
- Llama-2-70B fp16 (80, 8, 128, 2): 327,680 B/token (320 KiB)

---

## Escalated to user (6 items)

1. 8-bit quantization backend choice for Yi-34B / Llama-70B.
2. SentenceTransformers retrieval checkpoint.
3. GPT-4 query-paraphrase prompt + temperature + model version.
4. Throughput experiment batch size (Figure 14).
5. C_store closed form.
6. synchronize() primitive choice (whether to model two-thread pipeline at all).

---

## Counts

- HIGH: 14
- MEDIUM: 4
- LOW: 11
- Escalated to user: 6
