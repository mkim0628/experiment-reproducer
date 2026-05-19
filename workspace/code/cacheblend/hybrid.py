"""CacheBlend variants for hybrid Mamba/DeltaNet + Transformer models.

Motivation
----------
The base :mod:`cacheblend.baselines` module assumes a homogeneous transformer
stack where every decoder layer exposes a ``self_attn`` submodule with
``k_proj`` / ``v_proj`` and contributes K, V tensors to the KV cache. Recent
hybrid models (Qwen3.6-27B, Zamba2, Jamba, ...) interleave standard attention
layers with linear-attention or state-space layers (Gated DeltaNet, Mamba(2),
...). Those non-attention layers do not have a KV cache - they carry a
recurrent hidden state that depends sequentially on the entire prior context.

This module relaxes the homogeneity assumption:

* :func:`detect_attention_layers` walks ``model.model.layers`` and returns
  the indices whose layer module exposes a standard attention with K/V
  projections. Everything else is treated as opaque sequence-mixing
  machinery that must be re-run on the full sequence.
* :func:`precompute_chunk_kv_hybrid` is the hybrid analogue of
  :func:`cacheblend.precompute.precompute_chunk_kv`: it only hooks the
  attention layers identified above.
* :func:`cacheblend_generate_hybrid` is the hybrid analogue of
  :func:`cacheblend.baselines.cacheblend_generate`: the fused / blended
  past_key_values cache is keyed only by attention-layer indices and the
  HKVD check layer is the FIRST attention layer (not the model's layer 1,
  which on Qwen3.6 / Zamba2 is typically a Mamba/DeltaNet layer).

Important caveat
----------------
The chunk-isolated KV captured here is, structurally, a worse approximation
to the full-sequence KV than for pure transformers, because the input to each
attention layer comes through several intervening recurrent layers whose
state mixes the full prior sequence. We expect CacheBlend's quality on
hybrid models to degrade more sharply than on Llama / Mistral. The TTFT
benefit is also bounded above by the fraction of prefill time spent in the
attention layers (~25% for Qwen3.6, ~14% for Zamba2), so even if quality
held up the speedup ceiling is much lower than the paper's 6-10x.

This module exists to *measure* that degradation, not to claim CacheBlend
transfers cleanly to hybrid architectures.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from .kv_cache import ChunkKVStore
from .selective_recompute import (
    BlendConfig,
    compute_kv_deviation,
    merge_selective_kv,
    select_hkvd_indices,
)


# --------------------------------------------------------------- layer detection
_ATTN_PROJ_NAMES = ("k_proj", "v_proj")


def _layer_attn_module(layer: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return the attention submodule of ``layer`` iff it has K/V projections.

    Heuristic: look for an attribute named ``self_attn`` (Llama/Mistral/Qwen/
    Qwen3Next convention) or ``attn`` (some Zamba2 variants) that itself
    exposes ``k_proj`` and ``v_proj``. Returns ``None`` for Mamba / DeltaNet
    / SSM layers, which expose ``mixer`` / ``linear_attn`` / ``mamba``
    instead.
    """
    for candidate in ("self_attn", "attn"):
        sub = getattr(layer, candidate, None)
        if sub is None:
            continue
        if all(hasattr(sub, n) for n in _ATTN_PROJ_NAMES):
            return sub
    return None


def _k_capture_module(attn: torch.nn.Module) -> Tuple[torch.nn.Module, bool]:
    """Pick the submodule whose forward hook captures the right K_pre.

    CacheBlend stores K in *pre-RoPE* form so that RoPE can be re-applied at
    new positions during cache fusion. For models that apply an RMSNorm to
    K *after* k_proj but *before* RoPE (Qwen3 / Qwen3-Next), the stored K
    must be the post-norm, pre-RoPE value -- otherwise re-applying RoPE
    on the raw k_proj output produces values that disagree with the model's
    own forward pass.

    Returns ``(module, post_normed)`` -- ``post_normed`` is True when the
    captured tensor has already had k_norm applied.
    """
    k_norm = getattr(attn, "k_norm", None)
    if k_norm is not None:
        return k_norm, True
    return attn.k_proj, False


def detect_attention_layers(model) -> List[int]:
    """Indices of decoder layers that carry a standard K/V projection.

    On a pure transformer this is ``list(range(num_hidden_layers))``. On
    Qwen3.6 with block pattern [DeltaNet, DeltaNet, DeltaNet, Attention] x 16
    this returns ``[3, 7, 11, ..., 63]``. On Zamba2-1.2B (every 6 Mamba2
    blocks share a transformer) this returns the shared-transformer indices.
    """
    layers = model.model.layers
    return [i for i, layer in enumerate(layers) if _layer_attn_module(layer) is not None]


def attention_layer_modules(model) -> List[Tuple[int, torch.nn.Module]]:
    """Pairs of (layer_index, attention_submodule) for every attention layer."""
    out: List[Tuple[int, torch.nn.Module]] = []
    for i, layer in enumerate(model.model.layers):
        attn = _layer_attn_module(layer)
        if attn is not None:
            out.append((i, attn))
    return out


# --------------------------------------------------------- chunk KV precompute
@torch.no_grad()
def precompute_chunk_kv_hybrid(
    model,
    tokenizer,
    chunk_text: str,
    store: ChunkKVStore,
    attn_layer_indices: List[int],
    add_special_tokens: bool = False,
) -> bytes:
    """Capture (K_pre_rope, V) for the *attention layers only*.

    Differences from :func:`cacheblend.precompute.precompute_chunk_kv`:

    * Hooks are registered only on layers in ``attn_layer_indices``.
    * The store is expected to have been constructed with
      ``num_layers = len(attn_layer_indices)``; the i-th store slot
      corresponds to the i-th entry of ``attn_layer_indices``.
    * V is read from ``past_key_values`` at the same layer indices.

    The chunk hash is computed identically to the homogeneous variant so the
    hybrid runner can interoperate with the existing store API.
    """
    if not attn_layer_indices:
        raise ValueError("attn_layer_indices is empty - model has no attention layers")

    enc = tokenizer(chunk_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = enc["input_ids"].to(model.device)
    token_ids = input_ids[0].tolist()
    chunk_hash = ChunkKVStore.hash_chunk(token_ids)

    if store.has(chunk_hash) and all(
        store.fetch(chunk_hash, slot) is not None for slot in range(store.num_layers)
    ):
        return chunk_hash

    captured_k_pre: Dict[int, torch.Tensor] = {}

    def make_hook(model_layer_idx: int, attn_module, post_normed: bool):
        head_dim = attn_module.head_dim
        num_kv = attn_module.config.num_key_value_heads

        def hook(_module, _inputs, output):
            if post_normed:
                # k_norm output: (batch, seq, num_kv, head_dim) already reshaped.
                k_pre = output.transpose(1, 2).contiguous()
            else:
                # k_proj output: (batch, seq, num_kv*head_dim) flat.
                b, s, _ = output.shape
                k_pre = output.view(b, s, num_kv, head_dim).transpose(1, 2).contiguous()
            captured_k_pre[model_layer_idx] = k_pre

        return hook

    handles = []
    for li, attn in attention_layer_modules(model):
        if li not in attn_layer_indices:
            continue
        capture_mod, post_normed = _k_capture_module(attn)
        handles.append(capture_mod.register_forward_hook(make_hook(li, attn, post_normed)))

    try:
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(model.device)
        out = model.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=True,
        )
    finally:
        for h in handles:
            h.remove()

    past_kv = out.past_key_values
    for slot, model_li in enumerate(attn_layer_indices):
        # Older Cache APIs return per-layer (K, V) on attention layers only;
        # newer DynamicCache.layers indexes ALL model layers and yields
        # placeholder objects for non-attention slots, so we read directly
        # at the model layer index.
        try:
            k_post = past_kv.layers[model_li].keys
            v = past_kv.layers[model_li].values
        except (AttributeError, IndexError):
            k_post, v = past_kv[model_li]
        if k_post is None or v is None:
            raise RuntimeError(
                f"past_key_values at model layer {model_li} are None - the "
                "model's cache layout disagrees with our attention-layer "
                "detection. Inspect detect_attention_layers(model) output."
            )
        k_pre = captured_k_pre[model_li]
        store.put(chunk_hash, slot, k_pre, v)
    return chunk_hash


# ---------------------------------------------------------- end-to-end runner
def _tok_ids(tokenizer, text: str, add_special_tokens: bool = False) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)[
        "input_ids"
    ]


def _decode_new_tokens(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)


@torch.no_grad()
def full_recompute_generate_hybrid(
    model, tokenizer, prompt: str, max_new_tokens: int = 32
) -> str:
    """Hybrid-model passthrough wrapper - identical to the homogeneous version.

    Hybrid models do not need any special handling for full recompute, since
    we are not trying to reuse anything. Kept here for API symmetry so the
    Modal entrypoint can dispatch on strategy name.
    """
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


@torch.no_grad()
def cacheblend_generate_hybrid(
    model,
    tokenizer,
    chunks: List[str],
    query: str,
    store: ChunkKVStore,
    cfg: BlendConfig,
    attn_layer_indices: List[int],
    max_new_tokens: int = 32,
) -> str:
    """CacheBlend selective recompute, restricted to attention layers.

    Algorithm (deviations from the homogeneous variant called out inline):

    1. Precompute per-chunk (K_pre, V) at *attention layers only*. SSM /
       DeltaNet layers receive no per-chunk treatment - they will be
       recomputed naturally during the full prefill.
    2. Run one full prefill on (chunks + query) tokens. Capture K_new / V_new
       at the attention layers via hooks.
    3. Check layer = FIRST attention layer in ``attn_layer_indices`` (NOT
       the model-global ``cfg.check_layer``, since that's meaningless on a
       hybrid stack). Compute deviation, pick top-r% HKVD indices.
    4. For each attention layer, blend stored (K_pre, V) with the freshly
       computed K_new / V_new at the HKVD indices.
    5. Decode from the blended cache.

    NOTE: This implementation actually performs the full prefill in step 2,
    so it does NOT realize the TTFT benefit that CacheBlend gets on pure
    transformers. The point of this function is to measure *quality* under
    the selective merge - if quality already collapses there's no need to
    build the TTFT-saving variant. See the module docstring caveat.
    """
    if not attn_layer_indices:
        raise ValueError("attn_layer_indices is empty - model has no attention layers")

    device = model.device
    dtype = next(model.parameters()).dtype

    # ------------------------------------------------------- 1. precompute
    chunk_hashes: List[bytes] = []
    chunk_ids: List[torch.Tensor] = []
    for i, c in enumerate(chunks):
        add_special = (i == 0)
        ids = _tok_ids(tokenizer, c, add_special_tokens=add_special)
        chunk_ids.append(ids)
        chunk_hashes.append(
            precompute_chunk_kv_hybrid(
                model, tokenizer, c, store, attn_layer_indices,
                add_special_tokens=add_special,
            )
        )

    chunk_lengths = [t.shape[1] for t in chunk_ids]
    total_chunk_len = sum(chunk_lengths)

    # ------------------------------------------------------ 2. chunks prefill
    # Run prefill on the chunks-only sequence (no suffix / query yet). This
    # gives us:
    #   - For attention layers: K_new / V_new captured via hooks. These are
    #     the values we'd get if all chunks were prefilled together in one
    #     pass (the "fresh" reference).
    #   - For linear-attention layers: the recurrent state propagated through
    #     all chunks in order, which is what we want for the suffix decode.
    # We DELIBERATELY do not include suffix+query in this prefill so that the
    # downstream generate() call can feed them with past_key_values set to
    # this blended cache - same pattern as the homogeneous cacheblend_generate.
    suffix_query_ids = _tok_ids(tokenizer, query, add_special_tokens=False).to(device)
    chunk_ids_cat = torch.cat([t.to(device) for t in chunk_ids], dim=1)

    captured_k_pre: Dict[int, torch.Tensor] = {}
    captured_v: Dict[int, torch.Tensor] = {}

    def make_k_hook(li: int, attn, post_normed: bool):
        head_dim = attn.head_dim
        num_kv = attn.config.num_key_value_heads

        def hook(_m, _i, out):
            if post_normed:
                captured_k_pre[li] = out.transpose(1, 2).contiguous()
            else:
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
    for li, attn in attention_layer_modules(model):
        capture_mod, post_normed = _k_capture_module(attn)
        handles.append(capture_mod.register_forward_hook(make_k_hook(li, attn, post_normed)))
        handles.append(attn.v_proj.register_forward_hook(make_v_hook(li, attn)))
    try:
        prefill_out = model.model(input_ids=chunk_ids_cat, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    # ------------------------------------------------------- 3. HKVD
    # Check layer = first attention layer in the stack. We rotate K_pre (from
    # the store) at chunk positions to match K_new at the same positions
    # for an apples-to-apples K-deviation comparison.
    check_li = attn_layer_indices[0]
    rotary_emb = _resolve_rotary(model, check_li)
    apply_rotary = _resolve_apply_rotary(model)
    chunk_positions = torch.arange(total_chunk_len, device=device).unsqueeze(0)

    # Stored K_pre (slot 0) at the check layer, RoPE-rotated at chunk positions.
    k_pre_stored_chunks_list = []
    v_pre_stored_chunks_list = []
    for h, L in zip(chunk_hashes, chunk_lengths):
        kv = store.fetch(h, 0)  # slot 0 == check layer
        if kv is None:
            raise KeyError(f"missing chunk {h.hex()} at check layer (slot 0)")
        k_pre_stored_chunks_list.append(kv[0].to(device=device, dtype=dtype))
        v_pre_stored_chunks_list.append(kv[1].to(device=device, dtype=dtype))
    k_pre_stored = torch.cat(k_pre_stored_chunks_list, dim=-2)
    v_pre_stored = torch.cat(v_pre_stored_chunks_list, dim=-2)

    k_pre_rot_check = _maybe_recover_rope(k_pre_stored, chunk_positions, rotary_emb, apply_rotary)
    k_new_pre_check = captured_k_pre[check_li][..., :total_chunk_len, :]
    k_new_rot_check = _maybe_recover_rope(k_new_pre_check, chunk_positions, rotary_emb, apply_rotary)
    v_new_chunks = captured_v[check_li][..., :total_chunk_len, :]

    deviation = compute_kv_deviation(
        k_new_rot_check, k_pre_rot_check,
        v_new_chunks, v_pre_stored,
        mode=cfg.deviation_mode,
    )
    hkvd_idx = select_hkvd_indices(deviation, cfg.recompute_ratio).to(device)

    # ---------------------------------------------------- 4. blended cache
    # Start from the chunks-prefill cache, then at each attention layer
    # overwrite the non-HKVD chunk positions with the *chunk-isolated*
    # stored K_pre / V_pre. HKVD positions retain the fresh K_new / V_new
    # from the all-chunks prefill (= the "important" tokens get the right
    # context-aware KV; the rest get the cheap, chunk-isolated KV).
    # Linear-attention layers' state is left untouched: it's the state after
    # seeing all chunks in order, which is what the suffix decode needs.
    blended_cache: DynamicCache = prefill_out.past_key_values
    inv_idx = _complement_indices(total_chunk_len, hkvd_idx, device)

    for slot, model_li in enumerate(attn_layer_indices):
        k_chunks = []
        v_chunks = []
        for h in chunk_hashes:
            kv = store.fetch(h, slot)
            if kv is None:
                raise KeyError(f"missing chunk {h.hex()} at slot {slot}")
            k_chunks.append(kv[0].to(device=device, dtype=dtype))
            v_chunks.append(kv[1].to(device=device, dtype=dtype))
        k_pre = torch.cat(k_chunks, dim=-2)
        v_pre = torch.cat(v_chunks, dim=-2)
        k_pre_rot = _maybe_recover_rope(
            k_pre, chunk_positions, _resolve_rotary(model, model_li), apply_rotary
        )

        k_cache = blended_cache.layers[model_li].keys
        v_cache = blended_cache.layers[model_li].values
        merge_selective_kv(k_cache, v_cache, k_pre_rot, v_pre, inv_idx)

    # ----------------------------------------------------- 5. decode
    # Feed suffix+query with past_key_values covering the chunks. Mirrors
    # the homogeneous cacheblend_generate end-of-function pattern. We let
    # HF infer position_ids from cache length (DynamicCache reports its
    # own length to the model).
    attn_mask = torch.ones(
        (1, total_chunk_len + suffix_query_ids.shape[1]),
        device=device, dtype=torch.long,
    )
    out = model.generate(
        suffix_query_ids,
        attention_mask=attn_mask,
        past_key_values=blended_cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return _decode_new_tokens(tokenizer, out, suffix_query_ids.shape[1])


# ---------------------------------------------------------------- internals
def _resolve_rotary(model, layer_idx: int):
    """Return a rotary-embedding callable for the given attention layer.

    On Llama / Mistral / Qwen the rotary embedding lives at
    ``model.model.rotary_emb`` and is shared across all attention layers.
    Some hybrid models (Zamba2) attach rotary to the attention module
    itself; we fall back to that if the global rotary is absent.
    """
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is not None:
        return rotary
    attn = _layer_attn_module(model.model.layers[layer_idx])
    if attn is not None and hasattr(attn, "rotary_emb"):
        return attn.rotary_emb
    return None


def _resolve_apply_rotary(model):
    """Return the model-specific ``apply_rotary_pos_emb`` callable.

    The base ``cacheblend.blend.recover_rope_k`` hard-codes Mistral's
    rotation. Hybrid models (Qwen3Next, Falcon-H1, ...) ship their own
    apply_rotary that may use partial-rotation (rotary_dim < head_dim) or
    GLM-style interleaving; using Mistral's on those produces silently
    wrong K's. We sniff the model's modeling module and import its local
    apply_rotary_pos_emb.
    """
    module_name = type(model).__module__  # e.g. transformers.models.qwen3_next.modeling_qwen3_next
    try:
        mod = __import__(module_name, fromlist=["apply_rotary_pos_emb"])
        return getattr(mod, "apply_rotary_pos_emb")
    except (ImportError, AttributeError):
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
        return apply_rotary_pos_emb


def _maybe_recover_rope(
    k_pre: torch.Tensor,
    positions: torch.LongTensor,
    rotary_emb,
    apply_rotary,
) -> torch.Tensor:
    """RoPE-rotate ``k_pre`` if a rotary module is available; pass through otherwise.

    Some hybrid attention variants (e.g. Zamba2's shared transformer) use
    rotary; some Mamba-only stacks don't. We treat absence of a rotary
    callable as "K is already in its final form".
    """
    if rotary_emb is None:
        return k_pre
    if k_pre.dim() != 4:
        raise ValueError(
            f"k_pre must be 4-D (batch, num_kv_heads, seq_len, head_dim); got {tuple(k_pre.shape)}"
        )
    cos, sin = rotary_emb(k_pre, positions)
    fake_q = torch.zeros_like(k_pre)
    _, k_rot = apply_rotary(fake_q, k_pre, cos, sin)
    return k_rot


def _complement_indices(
    n: int, selected: torch.LongTensor, device
) -> torch.LongTensor:
    """Indices in [0, n) NOT in ``selected``, sorted ascending."""
    if selected.numel() == 0:
        return torch.arange(n, device=device, dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    mask[selected] = False
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)
