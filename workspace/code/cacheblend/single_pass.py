"""CacheBlend selective recompute -- one forward for both accuracy and TTFT.

``cacheblend_selective_generate`` is THE cacheblend strategy in this repo: a
**single forward** in which, after a small number of "full" layers, only the
selected HKVD tokens (plus the always-fresh suffix/query) are recomputed and
every other chunk token is served from its precomputed KV cache. The expensive
per-layer work (q/k/v projections, attention, MLP) therefore scales with
``r * chunk_len + suffix_len`` instead of the full sequence, which is where the
paper's multiple-x TTFT reduction comes from. Because it is one forward, the
SAME function is scored for accuracy and timed for TTFT in ``eval.run_eval``,
so the accuracy-drop vs TTFT-saving trade-off is read off one implementation.

Layer schedule (matches the official vllm_blend/llama.py status encoding)::

    li <  check_layer : full forward over all tokens.
    li == check_layer : full forward; compute V/K deviation of fresh vs cached
                        over chunk tokens, select the top-r% HKVD indices.
    li >  check_layer : recompute q/k/v only for HKVD chunk tokens + suffix;
                        attention queries = those active tokens, keys/values =
                        blended (fresh at active positions, cached elsewhere).

Correctness anchor (``check_r1_matches_full_forward``, run on GPU):

* ``r == 1.0``: every chunk token is HKVD -> ``active`` is the whole sequence at
  every layer -> the run is a plain full forward -> first-token logits are
  bit-identical to ``model(full_ids)``. This is the primary correctness test.

Note this does NOT reduce to ``full_reuse_generate`` at ``r == 0``: layers
``0..check_layer`` are always a full forward over all tokens (their fresh chunk
K/V is what the deviation at the check layer is measured against), so r=0 still
recomputes the first ``check_layer+1`` layers for chunk tokens rather than
serving them entirely from cache. That cost is exactly why selective recompute's
TTFT sits above pure reuse but well below full recompute. When recomputing an
HKVD token, non-HKVD chunk tokens are served from cache (the paper's behaviour),
not given fresh context.

NOTE: assumes full (non-sliding-window) causal attention. Mistral-7B-Instruct
-v0.2 sets ``sliding_window=null`` so this holds; a model with an active
sliding window would need the window folded into the attention mask below.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

from .baselines import _build_fused_cache, _tok_ids
from .kv_cache import ChunkKVStore
from .precompute import precompute_chunk_kv
from .selective_recompute import (
    BlendConfig,
    compute_kv_deviation,
    select_hkvd_indices,
)

try:  # HF exposes a module-level repeat_kv; fall back to a local copy.
    from transformers.models.mistral.modeling_mistral import repeat_kv as _hf_repeat_kv
except Exception:  # pragma: no cover
    _hf_repeat_kv = None


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if _hf_repeat_kv is not None:
        return _hf_repeat_kv(x, n_rep)
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


@torch.no_grad()
def _selective_prefill(
    model,
    fused_cache: DynamicCache,
    full_ids: torch.Tensor,
    total_chunk_len: int,
    cfg: BlendConfig,
    build_cache: bool = True,
) -> Tuple[Optional[DynamicCache], torch.Tensor, torch.LongTensor]:
    """Run the single-pass selective-recompute prefill.

    Args:
        model: HF MistralForCausalLM.
        fused_cache: per-layer cached chunk (K post-RoPE, V) for positions
            ``0..C-1`` (built by :func:`cacheblend.baselines._build_fused_cache`).
        full_ids: (1, T) token ids = concatenated chunks then suffix/query.
        total_chunk_len: C, the number of chunk tokens (suffix is ``T-C``).
        cfg: BlendConfig (recompute_ratio, check_layer, deviation_mode).
        build_cache: accumulate the per-layer blended KV over all T positions so
            decoding can continue. Set False for first-token-only uses (TTFT,
            logit checks) to avoid materializing the whole-prompt cache -- a big
            VRAM saving on long prompts.

    Returns:
        (blended_cache, logits_last, hkvd_idx) -- the blended KV cache over all
        T positions (n_kv-head layout, ready to continue decoding) or ``None``
        when ``build_cache`` is False, the logits of the final position
        (1, vocab), and the selected HKVD chunk indices.
    """
    device = model.device
    cfg_model = model.config
    n_heads = cfg_model.num_attention_heads
    n_kv = cfg_model.num_key_value_heads
    n_rep = n_heads // n_kv
    L = cfg_model.num_hidden_layers

    C = int(total_chunk_len)
    T = int(full_ids.shape[1])
    head_dim = getattr(model.model.layers[0].self_attn, "head_dim",
                        cfg_model.hidden_size // n_heads)
    scale = head_dim ** -0.5

    positions = torch.arange(T, device=device)
    key_pos = positions  # (T,)
    suffix_idx = torch.arange(C, T, device=device)

    h = model.model.embed_tokens(full_ids)  # (1, T, D)
    check_layer = cfg.check_layer
    hkvd_idx: Optional[torch.Tensor] = None
    blended_cache = DynamicCache() if build_cache else None

    for li in range(L):
        layer = model.model.layers[li]
        attn = layer.self_attn

        if li <= check_layer:
            active = positions                      # full forward
        else:
            active = torch.cat([hkvd_idx, suffix_idx]).sort().values
        A = int(active.shape[0])
        pos_act = positions.index_select(0, active).unsqueeze(0)  # (1, A)

        # ---- attention block (manual, mirrors HF MistralDecoderLayer) ----
        residual = h
        x = layer.input_layernorm(h)
        x_act = x.index_select(1, active)                          # (1, A, D)

        q = attn.q_proj(x_act).view(1, A, n_heads, head_dim).transpose(1, 2)
        k_act = attn.k_proj(x_act).view(1, A, n_kv, head_dim).transpose(1, 2)
        v_act = attn.v_proj(x_act).view(1, A, n_kv, head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(x_act, pos_act)
        q, k_act = apply_rotary_pos_emb(q, k_act, cos, sin)        # post-RoPE

        # Assemble the full (1, n_kv, T, head_dim) keys/values for this layer:
        # cached chunk KV everywhere, overwritten with fresh KV at active slots.
        if li <= check_layer:
            k_full, v_full = k_act, v_act                          # active == all T
        else:
            # cached chunk KV for positions [0,C); fresh KV scattered into the
            # active slots (hkvd chunk positions + all suffix positions [C,T)).
            k_full = torch.cat(
                [fused_cache.layers[li].keys.to(device),
                 k_act.new_zeros(1, n_kv, T - C, head_dim)], dim=2
            )
            v_full = torch.cat(
                [fused_cache.layers[li].values.to(device),
                 v_act.new_zeros(1, n_kv, T - C, head_dim)], dim=2
            )
            k_full.index_copy_(2, active, k_act)
            v_full.index_copy_(2, active, v_act)

        # ---- HKVD selection at the check layer (over chunk tokens) ----
        if li == check_layer:
            k_fresh_chunk = k_act[:, :, :C, :]
            v_fresh_chunk = v_act[:, :, :C, :]
            k_cached_chunk = fused_cache.layers[li].keys.to(device)
            v_cached_chunk = fused_cache.layers[li].values.to(device)
            deviation = compute_kv_deviation(
                k_fresh_chunk, k_cached_chunk,
                v_fresh_chunk, v_cached_chunk,
                mode=cfg.deviation_mode,
            )  # (C,)
            hkvd_idx = select_hkvd_indices(deviation, cfg.recompute_ratio).to(device)

        # ---- attention: active queries over all T (blended) keys ----
        kf = _repeat_kv(k_full, n_rep)
        vf = _repeat_kv(v_full, n_rep)
        if A == T:
            # active is the whole sequence in order (full-forward layers, and
            # every layer when r=1): plain causal attention. Using is_causal lets
            # SDPA pick the flash/mem-efficient kernel instead of materializing a
            # (T, T) score matrix from an explicit mask -- the difference between
            # fitting on an L4 and OOM on a long prompt.
            attn_out = F.scaled_dot_product_attention(q, kf, vf, is_causal=True, scale=scale)
        else:
            # genuinely selective: A scattered active queries (small) attend to
            # all T keys, allowed iff key_pos <= query_pos.
            mask = (key_pos.unsqueeze(0) <= pos_act.squeeze(0).unsqueeze(1)).view(1, 1, A, T)
            attn_out = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask, scale=scale)
        attn_out = attn_out.transpose(1, 2).reshape(1, A, n_heads * head_dim)
        o = attn.o_proj(attn_out)

        new_h = h.clone()
        new_h.index_copy_(1, active, h.index_select(1, active) + o)
        h = new_h

        # ---- MLP block (active rows only) ----
        h_act = h.index_select(1, active)
        mlp_out = layer.mlp(layer.post_attention_layernorm(h_act))
        h.index_copy_(1, active, h_act + mlp_out)

        if build_cache:
            blended_cache.update(k_full, v_full, li)

    last_hidden = model.model.norm(h[:, -1:, :])
    logits_last = model.lm_head(last_hidden)[:, -1, :]  # (1, vocab)
    return blended_cache, logits_last, hkvd_idx


@torch.no_grad()
def prepare_selective_inputs(model, tokenizer, chunks, store, suffix="", query=""):
    """Precompute chunk KV, build the fused cache, and the full token sequence.

    Shared by :func:`cacheblend_selective_generate` and the validation harness so
    both feed ``_selective_prefill`` from the *same* construction (and so the
    validator can compare against a full forward over the identical ``full_ids``).
    Returns ``(fused_cache, full_ids, total_chunk_len)``.
    """
    device = model.device
    dtype = next(model.parameters()).dtype
    chunk_hashes: List[bytes] = []
    chunk_ids: List[torch.Tensor] = []
    for i, c in enumerate(chunks):
        add_special = (i == 0)
        chunk_ids.append(_tok_ids(tokenizer, c, add_special_tokens=add_special))
        chunk_hashes.append(
            precompute_chunk_kv(model, tokenizer, c, store, add_special_tokens=add_special)
        )
    chunk_lengths = [t.shape[1] for t in chunk_ids]
    total_chunk_len = sum(chunk_lengths)
    offsets: List[int] = []
    running = 0
    for length in chunk_lengths:
        offsets.append(running)
        running += length
    fused_cache, _ = _build_fused_cache(
        model, chunk_hashes, store, offsets, chunk_lengths, device=device, dtype=dtype
    )
    suffix_query_ids = _tok_ids(tokenizer, suffix + query, add_special_tokens=False).to(device)
    chunk_ids_cat = torch.cat([t.to(device) for t in chunk_ids], dim=1)
    full_ids = torch.cat([chunk_ids_cat, suffix_query_ids], dim=1)
    return fused_cache, full_ids, total_chunk_len


@torch.no_grad()
def cacheblend_selective_generate(
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
    """Single-pass selective-recompute generation -- the cacheblend strategy.

    Same signature as the ``full_reuse_generate`` family (chunks + suffix/query,
    a shared chunk-KV store, a BlendConfig). Runs ONE forward, so its first-token
    latency is the algorithm's real TTFT and the same call also produces the text
    that run_eval scores for accuracy.
    """
    if prefix:
        raise NotImplementedError("cacheblend_selective_generate: prefix unsupported")
    device = model.device

    fused_cache, full_ids, total_chunk_len = prepare_selective_inputs(
        model, tokenizer, chunks, store, suffix=suffix, query=query
    )

    blended_cache, logits_last, _ = _selective_prefill(
        model, fused_cache, full_ids, total_chunk_len, cfg,
        build_cache=(max_new_tokens > 1),
    )

    first_id = int(torch.argmax(logits_last, dim=-1).item())
    generated: List[int] = [first_id]
    eos_id = tokenizer.eos_token_id

    # Continue greedy decoding from the blended cache (standard incremental
    # decode, one token per step; the prompt is NOT re-run -- that is the whole
    # point of single-pass). Skipped entirely for TTFT (max_new_tokens == 1).
    cur_pos = int(full_ids.shape[1])  # T; first generated token sits at T
    cur = torch.tensor([[first_id]], device=device)
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and generated[-1] == eos_id:
            break
        cache_position = torch.tensor([cur_pos], device=device)
        out = model(
            input_ids=cur,
            past_key_values=blended_cache,
            use_cache=True,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
        )
        blended_cache = out.past_key_values
        nxt = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        generated.append(nxt)
        cur = torch.tensor([[nxt]], device=device)
        cur_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def check_r1_matches_full_forward(
    model, tokenizer, chunks: List[str], store: ChunkKVStore, cfg: BlendConfig,
    suffix: str = "", query: str = "",
) -> Tuple[bool, float]:
    """Correctness anchor: at r=1 the selective forward must reduce to a full one.

    With recompute_ratio=1.0 every chunk token is HKVD, so ``active`` is the
    whole sequence at every layer and ``_selective_prefill`` becomes a plain full
    forward. We compare its final-position logits against an ordinary
    ``model(full_ids)`` over the identical ids. Returns
    ``(first_token_argmax_matches, max_abs_logit_diff)``; a correct forward gives
    ``(True, ~0.0)``. Independent of tokenization / long-decode drift since it
    only looks at the first-token logits on the same ids.
    """
    cfg_r1 = BlendConfig(recompute_ratio=1.0, check_layer=cfg.check_layer,
                         deviation_mode=cfg.deviation_mode)
    fused_cache, full_ids, total_chunk_len = prepare_selective_inputs(
        model, tokenizer, chunks, store, suffix=suffix, query=query)
    try:
        ref = model(full_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:  # older HF without logits_to_keep
        ref = model(full_ids, use_cache=False).logits[:, -1, :].float()
    _, sp, _ = _selective_prefill(
        model, fused_cache, full_ids, total_chunk_len, cfg_r1, build_cache=False)
    sp = sp.float()
    matches = bool(torch.argmax(ref, dim=-1).item() == torch.argmax(sp, dim=-1).item())
    return matches, float((sp - ref).abs().max().item())
