"""Stage-0 selection-quality analysis: how much headroom is there, and does the
attention-weighted signal capture it?

The accuracy of CacheBlend at a fixed recompute budget ``r`` is decided by *which*
``r%`` of chunk tokens it recomputes. This harness compares selection rules
(:mod:`cacheblend.selection`) through the *same* single-pass forward and scores
each by the real end metric we care about: the deviation of the first-token
logits from a true full prefill (the quantity CacheBlend's "attention deviation"
objective is a proxy for).

Outputs, per ``(rule, r)``:

* ``logit_l2``     -- L2 distance of first-token logits vs full prefill (lower better).
* ``logit_max``    -- max-abs logit difference vs full prefill.
* ``argmax_match`` -- whether the greedy first token matches full prefill.

Comparing rules at equal ``r``:

* ``raw`` is the released algorithm (top-r% by KV deviation).
* ``attn_weighted`` is the cheap online candidate (deviation x attention mass).
* ``oracle`` is an informed upper bound driven by :func:`oracle_score` -- a
  multi-layer importance computed from an instrumented full prefill.
* ``random`` is the lower-bound sanity floor.

``oracle - raw`` is the headroom; ``oracle - attn_weighted`` is what the cheap
candidate leaves on the table. This is GPU-cost-free to *validate* on synthetic
tensors (see tests) and meant to be run on a handful of real examples in one
cheap smoke before committing to a full Stage-1 sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

from .kv_cache import ChunkKVStore
from .selection import SELECTOR_FACTORIES, _repeat_kv
from .selective_recompute import BlendConfig, compute_kv_deviation
from .single_pass import _selective_prefill, prepare_selective_inputs


@dataclass
class LayerRecord:
    """True full-prefill chunk KV (post-RoPE) and suffix attention mass at a layer."""

    k_chunk_true: torch.Tensor   # (1, n_kv, C, head_dim)
    v_chunk_true: torch.Tensor   # (1, n_kv, C, head_dim)
    attn_mass: torch.Tensor      # (C,) suffix/query attention mass on each chunk token


@torch.no_grad()
def instrumented_full_forward(
    model, full_ids: torch.Tensor, total_chunk_len: int,
) -> Tuple[torch.Tensor, List[LayerRecord]]:
    """Full prefill that records per-layer chunk KV (post-RoPE) and attention mass.

    Mirrors the full-forward math in
    :func:`cacheblend.single_pass._selective_prefill` (active == all tokens at
    every layer) so the recorded KV matches what selective recompute compares
    against. Returns ``(logits_last, records)`` where ``logits_last`` is the
    first-token logits (1, vocab) and ``records[li]`` holds layer ``li``'s true
    chunk KV and suffix attention mass.
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
    pos_ids = positions.unsqueeze(0)
    suffix_idx = torch.arange(C, T, device=device)
    causal = (positions.unsqueeze(0) <= positions.unsqueeze(1)).view(1, 1, T, T)

    h = model.model.embed_tokens(full_ids)
    records: List[LayerRecord] = []

    for li in range(L):
        layer = model.model.layers[li]
        attn = layer.self_attn

        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1, T, n_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, T, n_kv, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, T, n_kv, head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(x, pos_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kf = _repeat_kv(k, n_rep)
        vf = _repeat_kv(v, n_rep)
        scores = torch.matmul(q, kf.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1)

        # suffix->chunk attention mass (avg over heads), per chunk token.
        if suffix_idx.numel() > 0:
            mass = probs.index_select(2, suffix_idx).sum(dim=2)  # (1, n_heads, T)
        else:
            mass = probs.sum(dim=2)
        mass = mass.mean(dim=1).squeeze(0)[:C].detach()          # (C,)

        records.append(LayerRecord(
            k_chunk_true=k[:, :, :C, :].detach().clone(),
            v_chunk_true=v[:, :, :C, :].detach().clone(),
            attn_mass=mass,
        ))

        attn_out = torch.matmul(probs.to(vf.dtype), vf)
        attn_out = attn_out.transpose(1, 2).reshape(1, T, n_heads * head_dim)
        h = residual + attn.o_proj(attn_out)
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    last_hidden = model.model.norm(h[:, -1:, :])
    logits_last = model.lm_head(last_hidden)[:, -1, :]
    return logits_last, records


@torch.no_grad()
def oracle_score(
    records: Sequence[LayerRecord],
    fused_cache,
    total_chunk_len: int,
    mode: str = "v",
    mass_source: str = "suffix",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-layer oracle importance of each chunk token.

    ``score[j] = sum_layers attn_mass_L[j] * deviation( cached_KV_L[j], true_KV_L[j] )``

    The deviation is the released per-token squared-L2 (mode ``v``/``k``/``kv``)
    between the *cached* chunk KV (what reuse serves) and the *true full-prefill*
    chunk KV (recorded). Weighting by the true attention mass turns it into each
    token's total contribution to the final attention deviation -- the thing a
    fixed selection should target. ``mass_source`` is kept for symmetry; the
    records already store suffix mass.
    """
    C = int(total_chunk_len)
    device = records[0].k_chunk_true.device
    score = torch.zeros(C, dtype=torch.float32, device=device)
    for li, rec in enumerate(records):
        cached_layer = fused_cache.layers[li]
        k_cached = cached_layer.keys.to(device)
        v_cached = cached_layer.values.to(device)
        dev = compute_kv_deviation(
            rec.k_chunk_true, k_cached, rec.v_chunk_true, v_cached, mode=mode,
        ).float()                                    # (C,)
        score = score + rec.attn_mass.to(device) * (dev + eps)
    return score


@torch.no_grad()
def score_selectors(
    model,
    fused_cache,
    full_ids: torch.Tensor,
    total_chunk_len: int,
    ratios: Sequence[float],
    rules: Sequence[str] = ("raw", "attn_weighted", "oracle", "random"),
    deviation_mode: str = "v",
    check_layer: int = 1,
    mass_source: str = "suffix",
) -> Dict[str, object]:
    """Measurement core: score each (rule, r) given a prebuilt fused cache + ids.

    Separated from :func:`selection_quality` (which does tokenization/precompute)
    so it can be unit-tested with a tiny model and a hand-built cache -- no
    tokenizer needed. One HF full forward gives the ground-truth logits; one
    instrumented full prefill gives the oracle score; then each (rule, r) runs ONE
    selective forward via the injectable selector and is scored against GT.
    """
    C = int(total_chunk_len)

    # Ground truth: HF full forward over the identical ids.
    try:
        gt = model(full_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        gt = model(full_ids, use_cache=False).logits[:, -1, :].float()
    gt_argmax = int(torch.argmax(gt, dim=-1).item())

    # Oracle score from an instrumented full prefill.
    score = None
    if "oracle" in rules:
        _, records = instrumented_full_forward(model, full_ids, C)
        score = oracle_score(records, fused_cache, C, mode=deviation_mode,
                             mass_source=mass_source)

    rows: List[Dict[str, object]] = []
    for rule in rules:
        if rule == "raw":
            selector = SELECTOR_FACTORIES["raw"](mode=deviation_mode)
            extra = None
        elif rule == "attn_weighted":
            selector = SELECTOR_FACTORIES["attn_weighted"](
                mode=deviation_mode, mass_source=mass_source)
            extra = None
        elif rule == "oracle":
            selector = SELECTOR_FACTORIES["oracle"]()
            extra = {"score": score}
        elif rule == "random":
            selector = SELECTOR_FACTORIES["random"](seed=0)
            extra = None
        else:
            raise ValueError(f"unknown rule {rule!r}")

        for r in ratios:
            cfg = BlendConfig(recompute_ratio=r, check_layer=check_layer,
                              deviation_mode=deviation_mode)
            _, logits_last, hkvd = _selective_prefill(
                model, fused_cache, full_ids, C, cfg, build_cache=False,
                selector=selector, selector_extra=extra)
            logits_last = logits_last.float()
            diff = logits_last - gt
            rows.append({
                "rule": rule,
                "ratio": float(r),
                "n_selected": int(hkvd.numel()),
                "logit_l2": float(diff.norm().item()),
                "logit_max": float(diff.abs().max().item()),
                "argmax_match": bool(torch.argmax(logits_last, dim=-1).item() == gt_argmax),
            })

    return {
        "chunk_len": C,
        "seq_len": int(full_ids.shape[1]),
        "gt_argmax": gt_argmax,
        "deviation_mode": deviation_mode,
        "check_layer": check_layer,
        "rows": rows,
    }


@torch.no_grad()
def selection_quality(
    model, tokenizer,
    chunks: List[str],
    suffix: str,
    store: ChunkKVStore,
    ratios: Sequence[float],
    rules: Sequence[str] = ("raw", "attn_weighted", "oracle", "random"),
    deviation_mode: str = "v",
    check_layer: int = 1,
    mass_source: str = "suffix",
) -> Dict[str, object]:
    """Tokenize + precompute chunk KV, then score selection rules on one example.

    Thin wrapper over :func:`score_selectors`: builds the fused cache and full ids
    from text chunks the same way ``cacheblend_selective_generate`` does, so the
    measurement uses the identical construction. Designed to run on a few examples
    in a cheap smoke.
    """
    fused_cache, full_ids, C = prepare_selective_inputs(
        model, tokenizer, chunks, store, suffix=suffix, query="")
    return score_selectors(
        model, fused_cache, full_ids, C, ratios, rules=rules,
        deviation_mode=deviation_mode, check_layer=check_layer,
        mass_source=mass_source)
