"""Pluggable HKVD selection rules + the Stage-0 oracle analysis primitives.

CacheBlend's accuracy at a fixed recompute budget ``r`` is decided entirely by
*which* ``r%`` of chunk tokens it recomputes. The released algorithm ranks
tokens by raw per-token KV deviation at a single check layer
(:func:`cacheblend.selective_recompute.compute_kv_deviation` +
:func:`select_hkvd_indices`) and freezes that set for all later layers.

This module makes the ranking *pluggable* so different rules can be compared
through the **same** selective forward (:func:`cacheblend.single_pass._selective_prefill`
accepts a ``selector``). A selector is::

    Callable[[SelectionContext, float], torch.LongTensor]

returning the chunk-token indices to recompute (a subset of ``[0, C)``), given
everything available at the check layer and the target ratio ``r``.

Three rules ship here:

* :func:`raw_deviation_selector` -- parity with the released algorithm
  (top-r% by K/V/KV squared-L2 deviation). Used as the baseline and to assert
  the injectable path reproduces the internal one bit-for-bit.
* :func:`attention_weighted_selector` -- the candidate signal: weight each
  token's KV deviation by the attention mass the query/suffix actually places
  on it, so budget goes to tokens that both *moved* and are *attended to*.
* :func:`precomputed_score_selector` -- selects top-r% by a score injected via
  ``ctx.extra['score']``. The Stage-0 harness uses this to drive the **oracle**:
  a multi-layer importance score (sum over layers of true-full-vs-cached KV
  deviation weighted by the true-full attention mass) computed from an
  instrumented full prefill -- the best a fixed check-layer selection could do
  if it knew each token's total contribution to the final output. The gap
  (oracle - raw) is the available headroom; (oracle - attention_weighted) is how
  much of it the cheap online candidate captures.

  A per-layer oracle is needed because at the check layer the fresh chunk KV
  already equals the true full-prefill KV (layers 0..check_layer are full
  forwards), so any check-layer-only "oracle" degenerates to the candidate.

All deviation comparisons are between post-RoPE chunk K (or V) at the SAME
absolute positions, matching the convention in ``_selective_prefill``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .selective_recompute import compute_kv_deviation, select_hkvd_indices


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (b, n_kv, s, d) -> (b, n_kv*n_rep, s, d) for GQA, like HF repeat_kv."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


@dataclass
class SelectionContext:
    """Everything a selector may read at the check layer.

    Tensors follow the (1, n_kv_or_heads, seq, head_dim) layout used by
    ``_selective_prefill``. K tensors are post-RoPE at their absolute positions.

    * ``k_fresh_chunk`` / ``v_fresh_chunk`` -- freshly computed chunk KV at the
      check layer, shape (1, n_kv, C, head_dim).
    * ``k_cached_chunk`` / ``v_cached_chunk`` -- the loaded (precomputed) chunk
      KV, same shape -- what reuse would serve.
    * ``q_all`` -- post-RoPE queries for ALL T positions, (1, n_heads, T, head_dim).
    * ``k_full`` / ``v_full`` -- blended keys/values used for attention at this
      layer, (1, n_kv, T, head_dim). At the check layer this equals the fresh
      full KV (the check layer is a full forward).
    * ``positions`` -- (T,) absolute positions.
    * ``suffix_idx`` -- (T-C,) the suffix/query positions, always recomputed.
    * ``scale`` -- attention softmax scale (head_dim ** -0.5).
    * ``n_rep`` -- GQA repeat factor (n_heads // n_kv).
    * ``extra`` -- optional injected data, e.g. ``k_full_true`` / ``v_full_true``
      (true full-prefill chunk KV at this layer) for the oracle.
    """

    k_fresh_chunk: torch.Tensor
    v_fresh_chunk: torch.Tensor
    k_cached_chunk: torch.Tensor
    v_cached_chunk: torch.Tensor
    q_all: torch.Tensor
    k_full: torch.Tensor
    v_full: torch.Tensor
    positions: torch.Tensor
    suffix_idx: torch.Tensor
    scale: float
    n_rep: int
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def C(self) -> int:
        return int(self.k_fresh_chunk.shape[2])

    @property
    def T(self) -> int:
        return int(self.q_all.shape[2])


Selector = Callable[[SelectionContext, float], torch.LongTensor]


# ------------------------------------------------------------ attention mass
def chunk_attention_mass(ctx: SelectionContext, mass_source: str = "suffix") -> torch.Tensor:
    """Per-chunk-token attention mass: how much the queries attend to each token.

    Computes attention probabilities at the check layer for the chosen query rows
    only and sums, for each chunk key ``j in [0, C)``, the probability mass they
    place on it, averaged over heads.

    Only the *query rows* are reduced (``mass_source="suffix"`` -> the T-C
    suffix/query rows; ``"all"`` -> all T rows); the score matrix is therefore
    ``(rows, T)`` rather than ``(T, T)``. With the suffix being short this avoids
    the full (T, T) materialization the single-pass path deliberately keeps off
    the L4 -- so this is safe to call inside production prefill, not just offline.
    The softmax denominator still spans all causally-visible keys (correct
    normalization); we keep only the chunk columns of the result.

    Returns a 1-D tensor of length C, non-negative.
    """
    if mass_source == "suffix":
        rows = ctx.suffix_idx
    elif mass_source == "all":
        rows = ctx.positions
    else:
        raise ValueError(f"mass_source must be 'suffix' or 'all'; got {mass_source!r}")
    if rows.numel() == 0:                            # no suffix -> fall back to all
        rows = ctx.positions

    q = ctx.q_all.index_select(2, rows)              # (1, n_heads, R, d)
    k = _repeat_kv(ctx.k_full, ctx.n_rep)            # (1, n_heads, T, d)
    scores = torch.matmul(q, k.transpose(-1, -2)) * ctx.scale   # (1, n_heads, R, T)
    qpos = ctx.positions.index_select(0, rows)       # (R,)
    causal = (ctx.positions.unsqueeze(0) <= qpos.unsqueeze(1))  # (R, T): key <= query
    scores = scores.masked_fill(~causal.view(1, 1, rows.shape[0], ctx.T), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1)    # (1, n_heads, R, T)

    mass = probs.sum(dim=2)                           # (1, n_heads, T)
    mass = mass.mean(dim=1).squeeze(0)                # (T,)
    return mass[: ctx.C]


# ------------------------------------------------------------------ selectors
def raw_deviation_selector(mode: str = "v") -> Selector:
    """Released CacheBlend rule: top-r% by raw K/V/KV squared-L2 deviation.

    Reproduces the internal selection in ``_selective_prefill`` exactly, so the
    injectable path can be asserted bit-for-bit against it.
    """

    def _select(ctx: SelectionContext, r: float) -> torch.LongTensor:
        deviation = compute_kv_deviation(
            ctx.k_fresh_chunk, ctx.k_cached_chunk,
            ctx.v_fresh_chunk, ctx.v_cached_chunk, mode=mode,
        )
        return select_hkvd_indices(deviation, r)

    return _select


def attention_weighted_selector(mode: str = "v", mass_source: str = "suffix",
                                eps: float = 1e-6) -> Selector:
    """Candidate rule: rank by (KV deviation) x (attention mass).

    A token is recomputed when it both *moved* (high reuse-vs-fresh KV deviation)
    and *matters* (the query/suffix attends to it). Tokens with large KV drift but
    near-zero attention weight no longer waste budget; moderately-drifted but
    heavily-attended tokens are no longer starved.
    """

    def _select(ctx: SelectionContext, r: float) -> torch.LongTensor:
        deviation = compute_kv_deviation(
            ctx.k_fresh_chunk, ctx.k_cached_chunk,
            ctx.v_fresh_chunk, ctx.v_cached_chunk, mode=mode,
        ).float()
        mass = chunk_attention_mass(ctx, mass_source=mass_source).to(deviation.device)
        importance = deviation * (mass + eps)
        return select_hkvd_indices(importance, r)

    return _select


def precomputed_score_selector() -> Selector:
    """Select top-r% by an externally injected per-chunk-token score.

    Reads ``ctx.extra['score']`` (a length-C tensor). The Stage-0 harness uses
    this to drive the oracle: it precomputes a multi-layer importance score from
    an instrumented full prefill (see :func:`cacheblend.analysis.oracle_score`)
    and injects it via ``selector_extra``.
    """

    def _select(ctx: SelectionContext, r: float) -> torch.LongTensor:
        if "score" not in ctx.extra:
            raise KeyError("precomputed_score_selector needs ctx.extra['score'] (length C)")
        score = ctx.extra["score"].float().to(ctx.q_all.device)
        if score.shape[0] != ctx.C:
            raise ValueError(f"score length {score.shape[0]} != C {ctx.C}")
        return select_hkvd_indices(score, r)

    return _select


def random_selector(seed: int = 0) -> Selector:
    """Lower-bound sanity rule: recompute a random ceil(r*C) chunk tokens."""

    def _select(ctx: SelectionContext, r: float) -> torch.LongTensor:
        import math

        C = ctx.C
        if C == 0 or r <= 0.0:
            return torch.empty(0, dtype=torch.long, device=ctx.q_all.device)
        k = min(C, max(1, math.ceil(r * C)))
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(C, generator=g)[:k]
        return torch.sort(perm).values.to(ctx.q_all.device)

    return _select


SELECTOR_FACTORIES: Dict[str, Callable[..., Selector]] = {
    "raw": raw_deviation_selector,
    "attn_weighted": attention_weighted_selector,
    "oracle": precomputed_score_selector,
    "random": random_selector,
}
