"""Offline (warm-up) pass that captures per-chunk pre-RoPE K and V.

For each chunk we run a single forward pass with ``use_cache=True`` at
positions ``[0..L-1]``. Two things are captured per decoder layer:

* The output of ``k_proj`` (reshaped to (batch, num_kv_heads, L, head_dim)).
  This is the *pre-RoPE* K -- exactly the tensor that RoPE will operate on.
* The V tensor from the layer's ``past_key_values`` (V is rotation-free, so
  it is reused verbatim at fusion time).

Both tensors are pushed into the supplied :class:`ChunkKVStore`.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from .kv_cache import ChunkKVStore


def _attention_modules(model) -> List[torch.nn.Module]:
    """Return the list of MistralAttention modules in layer order."""
    return [layer.self_attn for layer in model.model.layers]


@torch.no_grad()
def precompute_chunk_kv(
    model,
    tokenizer,
    chunk_text: str,
    store: ChunkKVStore,
    add_special_tokens: bool = False,
) -> bytes:
    """Run chunk in isolation and store (K_pre_rope, V) per layer.

    ``add_special_tokens=True`` lets the caller fold a leading BOS (Mistral
    id=1) into the first chunk so the cached layout matches what
    full-recompute would feed the model. The resulting chunk hash differs
    from the no-BOS variant; both can coexist in the same store.

    Returns the chunk hash used as the store key.
    """
    enc = tokenizer(chunk_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = enc["input_ids"].to(model.device)
    token_ids = input_ids[0].tolist()
    chunk_hash = ChunkKVStore.hash_chunk(token_ids)

    if store.has(chunk_hash):
        # All layers already present.
        if all(store.fetch(chunk_hash, l) is not None for l in range(store.num_layers)):
            return chunk_hash

    captured_k_pre: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int, attn_module):
        # k_proj output is (batch, seq_len, num_kv_heads * head_dim).
        head_dim = attn_module.head_dim
        num_kv = attn_module.config.num_key_value_heads

        def hook(_module, _inputs, output):
            # output shape: (batch, seq_len, num_kv_heads * head_dim)
            b, s, _ = output.shape
            k_pre = output.view(b, s, num_kv, head_dim).transpose(1, 2).contiguous()
            captured_k_pre[layer_idx] = k_pre

        return hook

    handles = []
    for li, attn in enumerate(_attention_modules(model)):
        handles.append(attn.k_proj.register_forward_hook(make_hook(li, attn)))

    try:
        out = model.model(
            input_ids=input_ids,
            attention_mask=enc.get("attention_mask", None).to(model.device)
            if enc.get("attention_mask", None) is not None
            else None,
            use_cache=True,
        )
    finally:
        for h in handles:
            h.remove()

    past_kv = out.past_key_values  # transformers.Cache (DynamicCache)
    n_layers = store.num_layers
    for li in range(n_layers):
        k_post, v = past_kv.layers[li].keys, past_kv.layers[li].values
        # Fallback for older Cache APIs:
        if k_post is None or v is None:
            k_post, v = past_kv[li]
        k_pre = captured_k_pre[li]
        # We store K_pre (un-rotated) and V (rotation-free).
        store.put(chunk_hash, li, k_pre, v)
    return chunk_hash


@torch.no_grad()
def precompute_all_chunks(
    model, tokenizer, chunks: Iterable[str], store: ChunkKVStore
) -> List[bytes]:
    return [precompute_chunk_kv(model, tokenizer, c, store) for c in chunks]
