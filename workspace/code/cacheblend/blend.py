"""RoPE re-rotation helper.

The CacheBlend convention stores K in *pre-rotation* form (i.e., the rotary
embedding has not yet been applied). When the chunk is concatenated into a
new request at new absolute positions, we re-apply rotary embedding on the
fly. This mirrors the call in the official fork:

    _, old_kv[0] = self.rotary_emb(cache_fuse_metadata['org_pos'],
                                   cache_fuse_metadata['fake_q'],
                                   old_kv[0])

(see vllm_blend/vllm/model_executor/models/llama.py:174-179).

In HuggingFace transformers' Mistral implementation, the rotary embedding is
implemented as a (cos, sin) pair followed by ``apply_rotary_pos_emb``. We mimic
that here while accepting a generic ``rotary_emb`` callable that takes the
hidden states (or any tensor with the right last dim) plus ``position_ids`` and
returns ``(cos, sin)``.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb


RotaryFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def recover_rope_k(
    k_pre: torch.Tensor,
    new_positions: torch.LongTensor,
    rotary_emb: RotaryFn,
) -> torch.Tensor:
    """Apply rotary embedding to a stored pre-rotation K at new positions.

    Args:
        k_pre: shape (batch, num_kv_heads, seq_len, head_dim) -- K stored
            without rotary applied.
        new_positions: shape (batch, seq_len) -- absolute positions to apply.
        rotary_emb: callable returning ``(cos, sin)`` given (x, position_ids).

    Returns:
        K with rotary embedding applied at ``new_positions``.
    """
    if k_pre.dim() != 4:
        raise ValueError(
            f"k_pre must be 4-D (batch, num_kv_heads, seq_len, head_dim); got {tuple(k_pre.shape)}"
        )
    if new_positions.dim() != 2:
        raise ValueError(
            f"new_positions must be 2-D (batch, seq_len); got {tuple(new_positions.shape)}"
        )
    cos, sin = rotary_emb(k_pre, new_positions)
    # apply_rotary_pos_emb rotates both q and k; we pass a dummy q (fake_q).
    fake_q = torch.zeros_like(k_pre)
    _, k_rot = apply_rotary_pos_emb(fake_q, k_pre, cos, sin)
    return k_rot


def concat_chunk_kvs(
    per_chunk_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    new_positions: torch.LongTensor,
    rotary_emb: RotaryFn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-chunk (K_pre, V) along the sequence axis and recover RoPE.

    Args:
        per_chunk_kv: list of (K_pre, V) tensors of shape
            (batch, num_kv_heads, chunk_len, head_dim) each.
        new_positions: (batch, total_len) absolute positions for the entire
            concatenated sequence.
        rotary_emb: rotary embedding module (callable as in
            :func:`recover_rope_k`).

    Returns:
        ``(K, V)`` of shape (batch, num_kv_heads, total_len, head_dim) ready
        for full-sequence attention.
    """
    if not per_chunk_kv:
        raise ValueError("per_chunk_kv must contain at least one chunk")
    k_cat = torch.cat([kv[0] for kv in per_chunk_kv], dim=-2)
    v_cat = torch.cat([kv[1] for kv in per_chunk_kv], dim=-2)
    if k_cat.shape[-2] != new_positions.shape[-1]:
        raise ValueError(
            f"total chunk length {k_cat.shape[-2]} != new_positions length "
            f"{new_positions.shape[-1]}"
        )
    k_rot = recover_rope_k(k_cat, new_positions, rotary_emb)
    return k_rot, v_cat
