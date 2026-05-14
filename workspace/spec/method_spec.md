# Method Specification: CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

## 1. Components to reproduce

- KVCacheStore (chunk-hash -> per-layer KV; CPU RAM or NVMe SSD; LRU)
- PositionalEncodingRecovery (RoPE recovery for non-prefix chunks; Appendix A)
- SelectiveKVRecompute (per-layer partial prefill of HKVD tokens; Section 4.2)
- HKVDSelector / Gradual Filtering (top-r% by KV deviation per layer; Section 4.3)
- Fusor (per-request layer-by-layer orchestration; Sections 5.1 and 6)
- LoadingController (chooses r% and device; Section 5.1)

## 2. Component: KVCacheStore

### Purpose
Maps each retrieved text chunk (hash of chunk text) to its pre-computed per-layer KV cache. Stores caches on a single storage level (CPU RAM or NVMe SSD), evicts LRU when full, and asynchronously writes newly computed caches back to disk.

### Inputs / Outputs
- Input: `text_chunk: str` (Langchain chunk, ~512 tokens; 200-400 for SAMSum), `layer_id: int`
- Output: per-layer `K, V` tensors of shape `(num_kv_heads, chunk_len, head_dim)`; sentinel `-1` if absent.

### Algorithm
Section 5.1 (KV cache store) and Section 6 (Managing KV cache). Hashing follows vLLM block-hashing; load from disk via `torch.load`, load from CPU via `torch.cuda`. Newly computed caches go through `torch.cpu()` then `torch.save` on a background thread.

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| chunk_size_tokens | 512 (200-400 for SAMSum) | Section 7.1 |
| eviction_policy | LRU | Section 5.1 |
| hash_table_location | CPU (~16 MB / 1M chunks) | Section 6 |
| storage_levels | single CPU RAM or NVMe SSD | Section 5.1 |
| measured_ssd_throughput_GBps | 4.8 | Section 7.1 |
| hash_function | `vllm.utils.hashing.sha256_cbor` over `(parent_hash, token_id_block)` | [RESOLVED via official_code, LMCache `lmcache/v1/token_database.py:95-150`] |
| kv_dtype_on_disk | model native activation dtype (fp16 / bf16); no quantization | [RESOLVED via official_code, vllm_blend `xformers.py:229`] |
| on_disk_file_layout | per `(chunk_hash, layer_id)` shard; `CacheEngineKey = model@world@worker@chunk_hash@dtype@layer_id` | [RESOLVED via official_code, LMCache `lmcache/utils.py:373-475`] |

### Unknowns
- `[UNKNOWN]` Capacity threshold triggering LRU eviction. ASSUMPTION: store sized to fit the eval working set (~25 GB on the 1 TB NVMe) so eviction is effectively disabled during measurement. [LOW confidence — escalated]

## 3. Component: PositionalEncodingRecovery

### Purpose
Recovers RoPE positional encoding for K (and Q at decode time) when a pre-computed chunk is placed at a different absolute position in the concatenated context. Justified by Proposition A.1: RoPE attention depends only on relative position.

### Inputs / Outputs
- Input: `K_pre` of shape `(num_kv_heads, chunk_len, head_dim)` stored **in pre-rotation form**, plus the new absolute positions of the concatenated context.
- Output: `K_pre_repositioned` of same shape.

### Algorithm
Appendix A. RoPE 2D rotation `[[cos(m*theta_i), -sin(m*theta_i)],[sin(m*theta_i), cos(m*theta_i)]]` with `theta_i = 10000^(-2i/d)`. The recovery exploits the identity
`q_{m+l} . k_m = sum_{i=0}^{d/2-1} (q[2i]*k[2i] + q[2i+1]*k[2i+1]) * cos(l * theta_i)`.

Concrete implementation: explicit forward RoPE rotation of loaded K with the new absolute positions, applied per-layer on-the-fly inside `LlamaAttention.forward` when `status in {1,2}`. A throwaway `fake_q = torch.rand_like(q)` is provided to satisfy the rotary kernel signature. [RESOLVED via official_code, vllm_blend `llama.py:174-179`]

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| theta_i | 10000^(-2i/d) | Definition 1 |
| rotation_block_size | 2 | Definition 1 |
| implementation | explicit forward rotation per-layer (not PromptCache dummy-prefix) | [RESOLVED via official_code] |
| non-RoPE / non-transformer models | out of scope, future work | [RESOLVED via paper, Section 9] |

### Unknowns
(none remain)

## 4. Component: SelectiveKVRecompute (per-layer)

### Purpose
Compute Q, K, V only for the selected HKVD tokens at the current layer; reuse loaded pre-computed K/V for the rest; run the layer's attention as usual.

### Inputs / Outputs
- Inputs: `input_tensor (batch, seq_len, hidden_dim)`, `input_metadata` (vLLM), `HKVD_indices (LongTensor)`, `check_flag: bool`, `KVCache_pre_layer = (K_pre, V_pre)` of shape `(num_kv_heads, seq_len, head_dim)`.
- Outputs: `output_tensor (batch, seq_len, hidden_dim)`, updated layer KV `(K_new, V_new)` of same shape, optionally updated `HKVD_indices_next`.

### Algorithm
Section 4.2 workflow (verbatim):
1. Apply a mask on the input of each layer i to reduce it to the subset of selected tokens.
2. Transform the reduced input into restricted `Q_i, K_i, V_i`.
3. Expand `K_i` and `V_i` by reusing pre-computed KV entries for unselected tokens, so attention includes attention between selected tokens and all other tokens.
4. Run the same attention module to produce the input for the next layer.

Compute overhead is proportional to r% of full prefill.

KV deviation per token: `Delta_kv(KV_i, KV_full_i)[j] = sum over (head, head_dim) of (V_new[j] - V_pre[j])^2` (per-token squared L2 of V only). [RESOLVED via official_code, vllm_blend `xformers.py:210-211`]

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| default_recompute_ratio_r | 15% | Section 4.3 / 5.1 |
| sweep_range | 5%-18% | Figure 16 |
| softmax_denominator_scope | all keys; standard 1/sqrt(d_k) scaling; selected Q attends to full-length K/V | [RESOLVED via official_code, vllm_blend `xformers.py:240-441`] |
| mask_expansion_strategy | positional recovery applied per-layer on-the-fly before expansion | [RESOLVED via official_code, vllm_blend `llama.py:174-179`] |
| partial_kv_writeback | in-place fancy-indexed assignment along seq dim (`key_old[imp_indices] = key`), equivalent to `index_copy_` | [RESOLVED via official_code, vllm_blend `xformers.py:240-245`] |

### Unknowns
(none remain)

## 5. Component: HKVDSelector (Gradual Filtering)

### Purpose
Pick which tokens to recompute. The **released code uses a single check layer at decoder index 1**; HKVD selected there is reused on all subsequent layers (i.e., the paper's qualitative `r_1 > r_2 > ... > r` schedule is implemented as a single-step selection in practice).

### Inputs / Outputs
- Inputs: candidate-set newly computed `V_new` and loaded `V_pre` at the check layer (each `(num_candidates, num_kv_heads, head_dim)`), target ratio `r%`.
- Output: `HKVD_indices` containing `ceil(r% * num_candidates)` indices with the highest token-wise V-deviation, reused on all later layers.

### Algorithm
Section 4.3:
- Insight 1: recomputing the top-`r%` tokens by `Delta_kv` reduces attention deviation most.
- Insight 2: HKVD tokens on layer i are likely HKVD on layer i+1 (Spearman correlation, Figure 8).
- **Implemented schedule** (single check at layer index 1, propagated): top-r% by squared-L2 V-deviation at layer 1; reuse for layers 2..N-1. [RESOLVED via official_code, vllm_blend `llama.py:300` (`check_layers:[1]`) and `350-356`]

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| target_average_r | 15% | Section 4.3 / 5.1 |
| quality_floor r* | 15% | Section 5.1 / Figure 16 |
| first_layer_candidate_set | all tokens of concatenated context | Section 4.3 |
| schedule | single check layer at decoder index 1 | [RESOLVED via official_code] |
| check_flag_per_layer | True only at layer index 1; False elsewhere (status 0/1/2 encoding) | [RESOLVED via official_code] |
| deviation_reduction | per-token squared L2, sum over head and head_dim axes | [RESOLVED via official_code] |
| deviation_tensor | V only (not K, not `[K;V]`) | [RESOLVED via official_code] |

### Unknowns
(none remain)

## 6. Component: Fusor

### Purpose
Layer-by-layer orchestrator. Pipelines `prefill_layer(layer=i)` with `fetch_kv(layer=i+1)` using two threads so KV loading hides recompute. NOTE: the released CacheBlend prototype runs sequentially; pipelining is the conceptual system design and must be added for the throughput experiments.

### Inputs / Outputs
- Inputs: ordered list of chunks, controller-chosen `r`.
- Output: full fused KV cache `[(K_l, V_l)]` for all layers, ready for decoding.

### Algorithm
Sections 5.1 (Fusor) and 5.2 + 6. Three interfaces:
- `fetch_kv(text, layer_id) -> KVCache`
- `prefill_layer(input_dict, KVCache) -> output_dict` with `input_dict` containing `input_org`, `check_flag`, `HKVD_indices`.
- `synchronize()` called before each `prefill_layer`.

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| num_pipeline_threads | 2 | Section 6 |
| fetch_kv_miss_behavior | fall back to full prefill (status=0); implicitly via `cache_fuse_metadata['check']=False` | [RESOLVED via official_code, vllm_blend `llama.py:330-356`] |

### Unknowns
- `[UNKNOWN]` Synchronization primitive (CUDA event vs threading.Event). ASSUMPTION: dedicated `torch.cuda.Stream` for host→device KV copy, default stream for prefill, `torch.cuda.Event` for layer hand-off, `threading.Thread` for the host fetch loop. [LOW confidence — escalated]
- `[UNKNOWN]` Whether `prefill_layer` uses a non-default CUDA stream. ASSUMPTION: yes, the fetch path uses a non-default stream; prefill uses the default stream. [LOW confidence — escalated]

## 7. Component: LoadingController

### Purpose
Pick recompute ratio `r%` and storage device so that `T_load >= T_recompute` per layer while enforcing `r >= r* = 15%`. Among admissible devices, choose the cheapest.

### Inputs / Outputs
- Inputs: offline `Prefill(LLM, L)` profile, `PerTokenKVSize(LLM)`, `L`, list of `(device, Throughput, C_store)` tuples.
- Outputs: `r_percent`, `device`.

### Algorithm
Section 5.1:
- `T_recompute(r%, LLM, L) = r% * Prefill(LLM, L)`
- `T_load(LLM, L, storage_device) = PerTokenKVSize(LLM) * L / Throughput(storage_device)`
- Step 1 (fixed device): solve `T_recompute(r%) == T_load(device)`, then `r = max(r, r*)`.
- Step 2 (fixed `r=15%`): pick cheapest device with `T_recompute(15%) >= T_load(device)` by `C_store`.

### Hyperparameters
| Hyperparameter | Value | Source |
| --- | --- | --- |
| r* | 15% | Section 5.1 / Figure 16 |
| PerTokenKVSize formula | `2 * num_layers * num_kv_heads * head_dim * sizeof(dtype)`; Mistral-7B fp16 = 131,072 B/token; Yi-34B fp16 = 245,760 B/token; Llama-70B fp16 = 327,680 B/token | [RESOLVED via paper + standard transformer KV accounting] |

### Unknowns
- `[UNKNOWN]` Closed form of `C_store`. ASSUMPTION: `bytes * time * price_per_byte_per_time(device)`; collapses to latency-only for the paper's setup. [LOW confidence — escalated]
- `[UNKNOWN]` How `Prefill(LLM, L)` is profiled (batch size, granularity). ASSUMPTION: batch_size=1, sweep `L in {512,1024,2048,4096,8192}`, fit `a*L + b*L^2`. [LOW confidence]

## 8. End-to-end pipeline

```
User query
  -> SentenceTransformers embed [ASSUMPTION: all-mpnet-base-v2; escalated]
       -> L2 nearest-neighbor over Langchain 512-token chunks -> top-6 chunks
  -> hash each chunk via vLLM sha256_cbor block-hash [RESOLVED via LMCache]
       -> KVCacheStore.fetch_kv(chunk, layer_id) for all layers
  -> LoadingController chooses (r%, device) using T_recompute, T_load, r*
  -> PositionalEncodingRecovery rewrites RoPE per-layer on-the-fly via rotary_emb(new_pos, fake_q, K_pre)  [RESOLVED via official_code]
  -> Fusor (2 threads — released prototype is sequential):
       Thread A: synchronize() -> prefill_layer(input_dict, KVCache_i)
                  - check layer (i=1): full prefill on candidate set, pick top-r% HKVD by squared-L2 of (V_new - V_pre), reuse for later layers  [RESOLVED]
                  - other layers (status=2): recompute only HKVD; key_old[imp_indices]=key; value_old[imp_indices]=value  [RESOLVED]
       Thread B: fetch_kv(layer=i+1) prefetch
  -> Fused KV cache -> vLLM decode loop generates answer
  -> Newly computed cache: torch.cpu() then async torch.save() write-back  [paper Section 6]
```

## 9. Reproduction targets

| Metric | Dataset / setting | Paper value | Source |
| --- | --- | --- | --- |
| TTFT reduction vs full KV recompute | 4 datasets x 3 models | 2.2-3.3x | Abstract; Figure 12 |
| Throughput vs full KV recompute | Musique/2WikiMQA Extended | 2.8-5x | Figure 14 |
| Quality drop vs full KV recompute | F1 / Rouge-L | <= 0.02 avg; <= 0.03 worst | Figure 12 |
| Quality gain vs full KV reuse | QA + summarization | 0.15-0.35 absolute | Section 7 |
| TTFT reduction at r in [5%, 18%] vs full recompute | Yi-34B, 4 datasets | 4.1-6.6x | Figure 16 |
| TTFT reduction at r in [5%, 18%] vs prefix caching | Yi-34B, 4 datasets | 3.4-6.1x | Figure 16 |
| Quality loss over r=5-18% sweep | Yi-34B, 4 datasets | <= 0.002 | Figure 16 |
| Per-layer recompute (r=15%) | Llama-7B, 4K | 3 ms | Section 5 |
| Per-layer KV load (NVMe) | Llama-7B, 4K | 16 ms | Section 5 |
| Per-layer recompute (r=15%) | Llama-70B, 4K | 7 ms | Section 5 |
| Per-layer KV load (NVMe) | Llama-70B, 4K | 4 ms | Section 5 |
| Full prefill TTFT | Llama-34B, 4K, 1xA40 | ~3 s | Section 2 |
| Full prefill TTFT | Llama-70B, 4K, 1xA40 | ~6 s | Section 2 |

## 10. Evaluation metrics — RESOLVED

- **F1**: SQuAD-style token-id F1 (HF tokenizer, BOS stripped) with `normalize_answer` = lower + remove articles (a/an/the) + remove punctuation + whitespace_fix. Reported as max over gold answers. [RESOLVED via official_code, `example/utils.py:26-74`]
- **Rouge-L**: `rouge_score` package, `RougeScorer(['rougeL'], use_stemmer=True)`, f-measure. [RESOLVED via official_code, `example/utils.py:76-79`]

## 11. Prompt templates — RESOLVED

For 2WikiMQA / Musique:
- prefix: `"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"`
- per-context: `f"{ctx['title']}\n\n{ctx['text']}\n\n"`
- query: `f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"`

For SAMSum: contexts appended verbatim with the query as suffix.

Mistral special tokens: `[INST]` = `[733, 16289, 28793]`, `[/INST]` = `[733, 28748, 16289, 28793]`.

[RESOLVED via official_code, `example/utils.py` + `example/blend_wikimqa.py`]

## Unknowns Summary (remaining, all LOW confidence)

- `[UNKNOWN]` Quantization method (GPTQ / AWQ / bitsandbytes) for 8-bit Yi-34B and Llama-70B. ASSUMPTION: bitsandbytes int8 OR TheBloke GPTQ-8bit. [escalated]
- `[UNKNOWN]` Specific SentenceTransformers checkpoint. ASSUMPTION: `sentence-transformers/all-mpnet-base-v2`. [escalated]
- `[UNKNOWN]` Batch sizes in Figure 14. ASSUMPTION: vLLM continuous batching, `max_num_seqs=256`, swept rate. [escalated]
- `[UNKNOWN]` GPT-4 paraphrase prompt / temperature / model. ASSUMPTION: `gpt-4-1106-preview`, T=0.7, paraphrase prompt. [escalated]
- `[UNKNOWN]` Closed-form C_store. ASSUMPTION: bytes·time·price_per_byte_per_time. [escalated]
- `[UNKNOWN]` synchronize() primitive. ASSUMPTION: torch.cuda.Stream + Event + threading.Thread. [escalated]
- `[UNKNOWN]` KV-store LRU capacity. ASSUMPTION: sized to fit ~25 GB working set; eviction off.
- `[UNKNOWN]` `Prefill(LLM, L)` offline profile. ASSUMPTION: batch_size=1, fit a·L + b·L^2 over L in {512,1024,2048,4096,8192}.

## Resolved unknowns (footnotes index)

[RESOLVED] = explicit citation present at the point of use above. Sources used: paper (arXiv 2405.16444); official_code = github.com/YaoJiayi/CacheBlend (vllm_blend fork) and github.com/LMCache/LMCache. See `ambiguity_log.md` / `ambiguity_log.json` for full mapping (29 resolutions, 6 escalations to user).
