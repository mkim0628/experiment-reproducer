"""Core CacheBlend algorithm: HKVD selection and selective KV recompute.

Algorithmic decisions (pinned in workspace/spec/ambiguity_log.json):

* HKVD deviation = per-token squared L2 of (V_new - V_pre); summed across
  head and head_dim. Code reference: vllm_blend/vllm/attention/backends/
  xformers.py:210-211. The official release uses V only; this module also
  exposes K-only and combined K+V deviation modes (``BlendConfig.deviation_mode``)
  as ablations -- not what the official numbers were measured with.
* Selection is performed at a SINGLE check layer (decoder index 1) and the
  resulting index set is reused on all subsequent layers. Reference:
  vllm_blend/.../models/llama.py:300 (``check_layers:[1]``).
* In-place fancy-indexed assignment writes recomputed K/V back into the loaded
  tensors: ``K_old[imp_indices] = K_new[imp_indices]``. Reference:
  vllm_blend/.../xformers.py:240-245.
* RoPE recovery is applied per layer to the *stored* (un-rotated) K before
  the in-place merge; see :mod:`cacheblend.blend`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# ----------------------------------------------------------------------- config
DEVIATION_MODES = ("v", "k", "kv")
SELECTION_MODES = ("raw", "attn_weighted")
MASS_SOURCES = ("suffix", "all")
BUDGET_MODES = ("ratio", "threshold")


@dataclass
class BlendConfig:
    recompute_ratio: float = 0.15
    check_layer: int = 1
    schedule: str = "single_check"  # or "every_layer"
    # Which tensor(s) to use when computing per-token deviation at the check
    # layer. The released vllm_blend implementation uses "v" only (HIGH-conf
    # resolution in ambiguity_log.json); we default to "k" per user request
    # for this reproduction -- this is an ablation choice, NOT the paper's
    # measured setting. Set to "v" to match the official numbers.
    deviation_mode: str = "k"
    # HKVD ranking rule. "raw" = released CacheBlend (top-r% by raw KV deviation).
    # "attn_weighted" = Stage-1 candidate: weight each token's deviation by the
    # attention mass the query/suffix places on it, so budget goes to tokens that
    # both moved AND are attended to. "raw" reproduces the official numbers.
    selection: str = "raw"
    # For selection="attn_weighted": which query rows define the attention mass.
    # "suffix" (default) = the suffix/query rows whose generation we care about
    # (cheap: R x T scores, R = suffix length); "all" = every query row.
    mass_source: str = "suffix"
    # Stage-2 adaptive budget. "ratio" = fixed top-(recompute_ratio) tokens (the
    # released behaviour). "threshold" = recompute every chunk token whose
    # importance score (under `selection`) is >= `threshold` * per-example max,
    # so easy examples recompute fewer tokens (lower average TTFT) and hard ones
    # more, at matched accuracy. `min_frac`/`max_frac` clamp the realized budget.
    budget_mode: str = "ratio"
    threshold: float = 0.5
    min_frac: float = 0.0
    max_frac: float = 1.0

    def __post_init__(self) -> None:
        if self.deviation_mode not in DEVIATION_MODES:
            raise ValueError(
                f"deviation_mode must be one of {DEVIATION_MODES}; "
                f"got {self.deviation_mode!r}"
            )
        if self.selection not in SELECTION_MODES:
            raise ValueError(
                f"selection must be one of {SELECTION_MODES}; got {self.selection!r}"
            )
        if self.mass_source not in MASS_SOURCES:
            raise ValueError(
                f"mass_source must be one of {MASS_SOURCES}; got {self.mass_source!r}"
            )
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(
                f"budget_mode must be one of {BUDGET_MODES}; got {self.budget_mode!r}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {self.threshold}")
        if not 0.0 <= self.min_frac <= self.max_frac <= 1.0:
            raise ValueError(
                f"need 0 <= min_frac <= max_frac <= 1; got "
                f"min_frac={self.min_frac}, max_frac={self.max_frac}"
            )


# ------------------------------------------------------------ per-token reduce
def _per_token_sq_l2(new: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Per-token squared-L2 reduction over (head, head_dim) axes.

    Accepts tensors of shape:
      * (batch, num_kv_heads, seq_len, head_dim) -- HF layout (batch squeezed)
      * (num_kv_heads, seq_len, head_dim)
      * (seq_len, head_dim) -- single-head fallback for tests
    """
    if new.shape != pre.shape:
        raise ValueError(f"shape mismatch: {tuple(new.shape)} vs {tuple(pre.shape)}")
    diff_sq = (new - pre) ** 2
    if diff_sq.dim() == 4:
        diff_sq = diff_sq.squeeze(0)
    if diff_sq.dim() == 3:
        return diff_sq.sum(dim=(0, 2))
    if diff_sq.dim() == 2:
        return diff_sq.sum(dim=-1)
    raise ValueError(f"unsupported tensor rank {diff_sq.dim()}")


# ----------------------------------------------------------------- V-deviation
def compute_v_deviation(v_new: torch.Tensor, v_pre: torch.Tensor) -> torch.Tensor:
    """Per-token squared-L2 deviation of V_new vs V_pre.

    Matches xformers.py:210-211: ``sum((V_new - V_pre)**2, dim=[heads, head_dim])``.
    Returns a 1-D tensor of length ``seq_len``.
    """
    return _per_token_sq_l2(v_new, v_pre)


def compute_k_deviation(k_new: torch.Tensor, k_pre: torch.Tensor) -> torch.Tensor:
    """Per-token squared-L2 deviation of K_new vs K_pre (ablation only).

    Same reduction as :func:`compute_v_deviation`. The caller must make sure
    K_new and K_pre are at the SAME RoPE state (both pre-rotation, or both
    post-rotation) -- comparing one pre and one post would just measure RoPE
    rather than cross-attention drift.
    """
    return _per_token_sq_l2(k_new, k_pre)


def compute_kv_deviation(
    k_new: torch.Tensor,
    k_pre: torch.Tensor,
    v_new: torch.Tensor,
    v_pre: torch.Tensor,
    mode: str = "v",
) -> torch.Tensor:
    """Per-token deviation under one of ``("v", "k", "kv")``.

    "kv" returns the sum of the K and V squared-L2 reductions. There is no
    cross-axis normalization because both K and V live in (num_kv_heads,
    head_dim) of equal size for Mistral/Llama-family models.
    """
    if mode == "v":
        return compute_v_deviation(v_new, v_pre)
    if mode == "k":
        return compute_k_deviation(k_new, k_pre)
    if mode == "kv":
        return compute_k_deviation(k_new, k_pre) + compute_v_deviation(v_new, v_pre)
    raise ValueError(f"mode must be one of {DEVIATION_MODES}; got {mode!r}")


def select_hkvd_indices(deviation: torch.Tensor, r: float) -> torch.LongTensor:
    """Select the top ceil(r * N) tokens by V-deviation.

    Args:
        deviation: 1-D tensor, length N (one scalar per token).
        r: recompute ratio in (0, 1].
    """
    if deviation.dim() != 1:
        raise ValueError(f"deviation must be 1-D, got shape {tuple(deviation.shape)}")
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"r must be in [0, 1]; got {r}")
    n = deviation.shape[0]
    if n == 0 or r == 0.0:
        return torch.empty(0, dtype=torch.long, device=deviation.device)
    k = min(n, max(1, math.ceil(r * n)))
    top = torch.topk(deviation, k=k).indices
    # Sort ascending so downstream slice writes are deterministic.
    return torch.sort(top).values


def select_by_threshold(
    score: torch.Tensor,
    tau: float,
    min_frac: float = 0.0,
    max_frac: float = 1.0,
) -> torch.LongTensor:
    """Adaptive-budget selection: recompute every token with score >= tau*max(score).

    The threshold is *normalized* to the per-example peak, so it is scale-free
    across examples (absolute deviation magnitudes vary): an easy example whose
    importance is concentrated in a few tokens selects few; a hard, diffuse one
    selects many. The realized count is clamped to
    ``[ceil(min_frac*N), max(1, ceil(max_frac*N))]`` so a degenerate example can
    neither recompute nothing (which would collapse to pure reuse) nor more than
    intended. ``tau=0`` selects all (subject to max_frac); larger ``tau`` selects
    fewer. Returns ascending indices, matching :func:`select_hkvd_indices`.

    Args:
        score: 1-D non-negative importance per token, length N.
        tau: normalized threshold in [0, 1].
        min_frac / max_frac: floor / ceiling on the recomputed fraction.
    """
    if score.dim() != 1:
        raise ValueError(f"score must be 1-D, got shape {tuple(score.shape)}")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1]; got {tau}")
    n = score.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    kmin = min(n, max(1, math.ceil(min_frac * n)))
    kmax = min(n, max(kmin, math.ceil(max_frac * n)))
    smax = score.max()
    if smax <= 0:  # no positive signal -> recompute the floor budget, deterministic
        sel = torch.topk(score, k=kmin).indices
        return torch.sort(sel).values
    k = int((score >= tau * smax).sum().item())
    k = max(kmin, min(k, kmax))
    # topk by score == exactly the >= tau*max set when k matches the mask count,
    # and the natural extension/restriction once clamped.
    sel = torch.topk(score, k=k).indices
    return torch.sort(sel).values


# ---------------------------------------------------------- in-place merge
def merge_selective_kv(
    k_old: torch.Tensor,
    v_old: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    imp_indices: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write freshly recomputed K/V at ``imp_indices`` into the loaded tensors.

    Layout expected: (batch, num_kv_heads, seq_len, head_dim). Operates
    in-place on ``k_old``/``v_old`` and returns them for chaining.
    Matches ``key_old[imp_indices] = key`` from xformers.py:240.
    """
    if imp_indices.numel() == 0:
        return k_old, v_old
    # k_old shape: (batch, num_kv_heads, seq_len, head_dim) -- index along dim=2.
    k_old.index_copy_(2, imp_indices, k_new.index_select(2, imp_indices))
    v_old.index_copy_(2, imp_indices, v_new.index_select(2, imp_indices))
    return k_old, v_old
