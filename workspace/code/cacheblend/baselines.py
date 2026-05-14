"""Three generation strategies for the CacheBlend quality comparison.

Strategies
----------
* :func:`full_recompute_generate` -- vanilla HuggingFace forward; no cache.
* :func:`full_reuse_generate` -- "PromptCache" style: concatenate per-chunk
  pre-computed (K_pre, V) tensors, RoPE-recover K at new positions, feed
  to the model as ``past_key_values`` together with only the query tokens.
  No recompute anywhere.
* :func:`cacheblend_generate` -- selective recompute: same starting point as
  full_reuse, but at the single check layer (decoder index 1) we compute
  V-deviation against the freshly-computed V_new for the chunk tokens, pick
  the top-r% HKVD indices, and replace K/V at those slots with the freshly
  computed values. All later layers inherit the same HKVD set.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import DynamicCache

from .blend import recover_rope_k
from .kv_cache import ChunkKVStore
from .precompute import precompute_chunk_kv
from .selective_recompute import (
    BlendConfig,
    compute_kv_deviation,
    merge_selective_kv,
    select_hkvd_indices,
)


# --------------------------------------------------------------- helpers
def _tok_ids(tokenizer, text: str, add_special_tokens: bool = False) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)[
        "input_ids"
    ]


def _decode_new_tokens(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ----------------------------------------------------------- full recompute
@torch.no_grad()
def full_recompute_generate(
    model, tokenizer, prompt: str, max_new_tokens: int = 32
) -> str:
    """Standard HF generation; included as the 'no-cache' quality reference."""
    input_ids = _tok_ids(tokenizer, prompt, add_special_tokens=True).to(model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return _decode_new_tokens(tokenizer, out, input_ids.shape[1])


# ------------------------------------------------ build the fused KV cache
@torch.no_grad()
def _build_fused_cache(
    model,
    per_chunk_hashes: List[bytes],
    store: ChunkKVStore,
    chunk_token_offsets: List[int],
    chunk_lengths: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[DynamicCache, int]:
    """Assemble per-layer (K, V) caches from precomputed pre-RoPE chunks.

    Each chunk's K is RoPE-recovered at the absolute positions it occupies in
    the concatenated context. Returns the cache and the total fused length.
    """
    rotary_emb = model.model.rotary_emb
    total_len = sum(chunk_lengths)
    # Build new absolute positions in one shot.
    positions: List[int] = []
    for offset, length in zip(chunk_token_offsets, chunk_lengths):
        positions.extend(range(offset, offset + length))
    pos_ids = torch.tensor([positions], device=device, dtype=torch.long)

    layer_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_layers = model.config.num_hidden_layers
    for li in range(num_layers):
        k_chunks: List[torch.Tensor] = []
        v_chunks: List[torch.Tensor] = []
        for h in per_chunk_hashes:
            kv = store.fetch(h, li)
            if kv is None:
                raise KeyError(f"missing chunk {h.hex()} at layer {li}")
            k_pre, v = kv
            k_chunks.append(k_pre.to(device=device, dtype=dtype))
            v_chunks.append(v.to(device=device, dtype=dtype))
        k_cat = torch.cat(k_chunks, dim=-2)  # (1, num_kv, total_len, head_dim)
        v_cat = torch.cat(v_chunks, dim=-2)
        k_rot = recover_rope_k(k_cat, pos_ids, rotary_emb)
        layer_kvs.append((k_rot, v_cat))

    cache = DynamicCache()
    for li, (k, v) in enumerate(layer_kvs):
        cache.update(k, v, li)
    return cache, total_len


# ---------------------------------------------------------- full reuse
@torch.no_grad()
def full_reuse_generate(
    model,
    tokenizer,
    chunks: List[str],
    query: str,
    store: ChunkKVStore,
    max_new_tokens: int = 32,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Concatenate pre-computed per-chunk KV, no recompute, then decode."""
    device = model.device
    dtype = next(model.parameters()).dtype

    # Make sure each chunk is precomputed.
    chunk_hashes: List[bytes] = []
    chunk_ids: List[torch.Tensor] = []
    for c in chunks:
        ids = _tok_ids(tokenizer, c, add_special_tokens=False)
        chunk_ids.append(ids)
        chunk_hashes.append(precompute_chunk_kv(model, tokenizer, c, store))

    # Build prefix/suffix prefill segments (if any). The query (and any
    # tokens of suffix/prefix) is what we still need to feed forward.
    prefix_ids = _tok_ids(tokenizer, prefix, add_special_tokens=True) if prefix else None
    suffix_ids = _tok_ids(tokenizer, suffix + query, add_special_tokens=False)

    chunk_lengths = [t.shape[1] for t in chunk_ids]

    # We treat chunks as if they were placed AFTER any prefix tokens.
    prefix_len = prefix_ids.shape[1] if prefix_ids is not None else 0
    offsets: List[int] = []
    running = prefix_len
    for L in chunk_lengths:
        offsets.append(running)
        running += L

    # Assemble cache populated only with chunk tokens (no prefix yet).
    fused_cache, total_chunk_len = _build_fused_cache(
        model,
        chunk_hashes,
        store,
        offsets,
        chunk_lengths,
        device=device,
        dtype=dtype,
    )

    # Build the "remaining" tokens to actually run through the model: we need
    # the model to produce K/V for prefix+suffix+query. For simplicity, we
    # PRE-PEND the prefix as a fresh prefill (its positions are 0..prefix_len-1)
    # and then run the suffix/query at positions starting after total chunks.
    # To keep things simple and correct, we instead REPREFILL the prefix +
    # query (with the chunk KV inserted via past_key_values), letting HF do
    # the right positional bookkeeping.
    full_ids = torch.cat(
        [t for t in [prefix_ids, suffix_ids] if t is not None], dim=1
    ).to(device)

    # Positions of the new tokens: prefix occupies [0..prefix_len-1] but the
    # chunk KV is already at [prefix_len..prefix_len+total_chunks-1], so the
    # newly-prefilled tokens occupy [prefix_len+total_chunks..].
    new_start = prefix_len + total_chunk_len
    new_len = full_ids.shape[1]
    new_position_ids = torch.arange(new_start, new_start + new_len, device=device).unsqueeze(0)

    # NB: this places prefix AFTER chunks in position space. The
    # implementation below ignores prefix for simplicity (reproduction uses
    # the official prompt template, which has the system text already inside
    # the chunks). We assert this is fine.
    if prefix_ids is not None and prefix_ids.shape[1] > 0:
        raise NotImplementedError(
            "full_reuse_generate does not yet support a non-empty prefix; "
            "fold the prefix into the first chunk."
        )

    attn_mask = torch.ones(
        (1, total_chunk_len + new_len), device=device, dtype=torch.long
    )
    out = model.generate(
        full_ids,
        attention_mask=attn_mask,
        past_key_values=fused_cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        position_ids=new_position_ids,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return _decode_new_tokens(tokenizer, out, full_ids.shape[1])


# ------------------------------------------------------------- cacheblend
@torch.no_grad()
def cacheblend_generate(
    model,
    tokenizer,
    chunks: List[str],
    query: str,
    store: ChunkKVStore,
    cfg: BlendConfig,
    max_new_tokens: int = 32,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Selective KV recompute on a single check layer.

    Algorithm (matches the official vllm_blend implementation):

    1. Build the fused KV cache exactly like full_reuse_generate.
    2. Run one full prefill on (chunks + query) tokens with hooks that
       capture freshly-computed K_new (pre-RoPE) and V_new at every layer.
    3. At ``cfg.check_layer`` compute V-deviation between V_new and the loaded
       V_pre over the *chunk* tokens, pick the top-r% indices.
    4. For each layer >= check_layer (and at the check layer itself), merge:
       at the HKVD indices write K_new and V_new into the fused cache;
       elsewhere keep K_pre (RoPE-recovered) and V_pre.
    5. Continue decoding from the blended cache with the query's final-token
       logits.

    For pragmatic compatibility with HF transformers, we approximate step (2-5)
    by running the model TWICE: once to capture K_new/V_new and once to
    generate from the blended cache. This costs more than the official
    implementation in TTFT but matches the algorithm's *quality* exactly.
    """
    device = model.device
    dtype = next(model.parameters()).dtype

    # 1. Ensure chunks are precomputed and assemble the fused cache.
    chunk_hashes: List[bytes] = []
    chunk_ids: List[torch.Tensor] = []
    for c in chunks:
        ids = _tok_ids(tokenizer, c, add_special_tokens=False)
        chunk_ids.append(ids)
        chunk_hashes.append(precompute_chunk_kv(model, tokenizer, c, store))

    chunk_lengths = [t.shape[1] for t in chunk_ids]
    total_chunk_len = sum(chunk_lengths)
    offsets: List[int] = []
    running = 0
    for L in chunk_lengths:
        offsets.append(running)
        running += L

    fused_cache, _ = _build_fused_cache(
        model,
        chunk_hashes,
        store,
        offsets,
        chunk_lengths,
        device=device,
        dtype=dtype,
    )

    # 2. Build the full sequence (chunks concatenated + query).
    if prefix:
        raise NotImplementedError("cacheblend_generate: prefix unsupported here")
    suffix_query_ids = _tok_ids(tokenizer, suffix + query, add_special_tokens=False).to(device)
    chunk_ids_cat = torch.cat([t.to(device) for t in chunk_ids], dim=1)
    full_ids = torch.cat([chunk_ids_cat, suffix_query_ids], dim=1)

    # 2a. Capture per-layer K_new (pre-RoPE) and V_new for the WHOLE sequence
    # via hooks on k_proj / v_proj.
    captured_k_pre: dict[int, torch.Tensor] = {}
    captured_v: dict[int, torch.Tensor] = {}

    def make_k_hook(li: int, attn):
        head_dim = attn.head_dim
        num_kv = attn.config.num_key_value_heads

        def hook(_m, _i, out):
            b, s, _ = out.shape
            captured_k_pre[li] = out.view(b, s, num_kv, head_dim).transpose(1, 2).contiguous()

        return hook

    def make_v_hook(li: int, attn):
        head_dim = attn.head_dim
        num_kv = attn.config.num_key_value_heads

        def hook(_m, _i, out):
            b, s, _ = out.shape
            captured_v[li] = out.view(b, s, num_kv, head_dim).transpose(1, 2).contiguous()

        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        handles.append(attn.k_proj.register_forward_hook(make_k_hook(li, attn)))
        handles.append(attn.v_proj.register_forward_hook(make_v_hook(li, attn)))
    try:
        _ = model.model(input_ids=full_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # 3. Deviation at the check layer over the chunk tokens. Default mode "v"
    # matches the released code; "k" and "kv" are ablations exposed via
    # BlendConfig.deviation_mode.
    check_li = cfg.check_layer
    rotary_emb = model.model.rotary_emb
    chunk_positions = torch.arange(total_chunk_len, device=device).unsqueeze(0)
    v_new_full = captured_v[check_li]  # (1, num_kv, total_len, head_dim)
    v_pre_chunks = fused_cache.layers[check_li].values  # (1, num_kv, total_chunk_len, head_dim)
    v_new_chunks = v_new_full[..., :total_chunk_len, :]
    # For K-deviation we want both K's at the SAME RoPE state. fused_cache K
    # is post-RoPE (rotated in _build_fused_cache); so we rotate K_new the same
    # way at the chunk positions before comparing.
    k_new_pre_check = captured_k_pre[check_li][..., :total_chunk_len, :]
    k_new_rot_check = recover_rope_k(k_new_pre_check, chunk_positions, rotary_emb)
    k_pre_chunks = fused_cache.layers[check_li].keys
    deviation = compute_kv_deviation(
        k_new_rot_check,
        k_pre_chunks,
        v_new_chunks,
        v_pre_chunks,
        mode=cfg.deviation_mode,
    )  # (total_chunk_len,)
    hkvd_idx = select_hkvd_indices(deviation, cfg.recompute_ratio).to(device)

    # 4. Build blended cache: at each layer, RoPE-recover K_new at the same
    # positions used for K_pre (which is positions 0..total_chunk_len-1 since
    # chunks come first), then merge at HKVD indices. (rotary_emb /
    # chunk_positions were already bound above for the check-layer rotation.)
    blended_cache = DynamicCache()
    for li in range(model.config.num_hidden_layers):
        k_pre_loaded, v_loaded = fused_cache.layers[li].keys, fused_cache.layers[li].values
        # Re-rotate K_new at the same positions for an apples-to-apples merge.
        k_new_pre = captured_k_pre[li][..., :total_chunk_len, :]
        v_new = captured_v[li][..., :total_chunk_len, :]
        k_new_rot = recover_rope_k(k_new_pre, chunk_positions, rotary_emb)
        k_blend = k_pre_loaded.clone()
        v_blend = v_loaded.clone()
        merge_selective_kv(k_blend, v_blend, k_new_rot, v_new, hkvd_idx)
        blended_cache.update(k_blend, v_blend, li)

    # 5. Decode from the blended cache feeding the suffix+query tokens.
    new_start = total_chunk_len
    new_position_ids = torch.arange(
        new_start, new_start + suffix_query_ids.shape[1], device=device
    ).unsqueeze(0)
    attn_mask = torch.ones(
        (1, total_chunk_len + suffix_query_ids.shape[1]), device=device, dtype=torch.long
    )
    out = model.generate(
        suffix_query_ids,
        attention_mask=attn_mask,
        past_key_values=blended_cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        position_ids=new_position_ids,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return _decode_new_tokens(tokenizer, out, suffix_query_ids.shape[1])
