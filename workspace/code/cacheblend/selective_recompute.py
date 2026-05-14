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


@dataclass
class BlendConfig:
    recompute_ratio: float = 0.15
    check_layer: int = 1
    schedule: str = "single_check"  # or "every_layer"
    # Which tensor(s) to use when computing per-token deviation at the check
    # layer. "v" matches the released vllm_blend implementation (HIGH-conf
    # resolution in ambiguity_log.json). "k" and "kv" are ablations exposed
    # by user request -- not part of the paper's measured numbers.
    deviation_mode: str = "v"

    def __post_init__(self) -> None:
        if self.deviation_mode not in DEVIATION_MODES:
            raise ValueError(
                f"deviation_mode must be one of {DEVIATION_MODES}; "
                f"got {self.deviation_mode!r}"
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


# ----------------------------------------------------- selective layer forward
def selective_layer_forward(
    decoder_layer,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    pre_kv_layer: Tuple[torch.Tensor, torch.Tensor],
    hkvd_idx: Optional[torch.LongTensor],
    status: int,
    rotary_emb,
    cfg: BlendConfig,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.LongTensor]]:
    """Run a single transformer block in CacheBlend selective-recompute mode.

    Status encoding (matches vllm_blend/llama.py:350-356):
        0 -- full prefill (no cache; normal HF forward)
        1 -- check layer: prefill full sequence, compute V deviation against
             stored V_pre, update ``hkvd_idx`` to top-r%.
        2 -- post-check layer: only Q/K/V for tokens in ``hkvd_idx`` are recomputed;
             non-selected tokens keep their loaded K/V (RoPE recovered).

    This implementation focuses on the *KV side* of the algorithm. Because
    softmax denominator must include all keys, Q is always computed for every
    position; only the K/V tensors used as keys/values for attention are
    selectively swapped in. (See ambiguity log: "Standard softmax denominator
    over ALL keys".)

    Returns:
        (layer_output, (K_layer, V_layer), updated_hkvd_idx)
    """
    if status == 0:
        # Standard HF forward; cache will be re-populated from scratch.
        out = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=True,
        )
        layer_out = out[0]
        # HF Mistral returns hidden_states, (k, v) when use_cache=True; the
        # K/V here are post-RoPE. Caller is responsible for any storage
        # conversion. We don't have the cache object exposed here in modern HF
        # so fall back to running the layer's self_attn projections directly.
        return layer_out, pre_kv_layer, hkvd_idx

    raise NotImplementedError(
        "Full HF-integration of selective_layer_forward is not wired in the "
        "quality-only reproduction. The selective merge is exercised through "
        ":func:`merge_selective_kv` and the end-to-end driver lives in "
        "cacheblend.baselines.cacheblend_generate, which directly assembles "
        "K_pre / V_pre into a single forward pass with `past_key_values`."
    )
