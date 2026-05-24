"""Baseline generation strategies + the fused-KV-cache builder.

* :func:`full_recompute_generate` -- vanilla HuggingFace forward; no cache.
* :func:`full_reuse_generate` -- "PromptCache" style: concatenate per-chunk
  pre-computed (K_pre, V) tensors, RoPE-recover K at new positions, feed
  to the model as ``past_key_values`` together with only the query tokens.
  No recompute anywhere.

CacheBlend's selective recompute lives in :mod:`cacheblend.single_pass`
(``cacheblend_selective_generate``), a single forward that recomputes only the
HKVD chunk tokens + suffix. That same function is used for BOTH accuracy and
TTFT in the eval harnesses, so the two are read off one implementation.
:func:`_build_fused_cache` here is shared by full_reuse and single_pass.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import DynamicCache

from .blend import recover_rope_k
from .kv_cache import ChunkKVStore
from .precompute import precompute_chunk_kv


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

    # Make sure each chunk is precomputed. The FIRST chunk includes BOS so
    # the cached positional layout matches what full_recompute_generate would
    # feed (which tokenizes the whole prompt with add_special_tokens=True).
    chunk_hashes: List[bytes] = []
    chunk_ids: List[torch.Tensor] = []
    for i, c in enumerate(chunks):
        add_special = (i == 0)
        ids = _tok_ids(tokenizer, c, add_special_tokens=add_special)
        chunk_ids.append(ids)
        chunk_hashes.append(
            precompute_chunk_kv(model, tokenizer, c, store, add_special_tokens=add_special)
        )

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
