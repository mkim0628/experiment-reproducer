"""Tests for cacheblend.blend.recover_rope_k.

The store-then-recover-RoPE convention must be equivalent to applying
rotary embedding directly at the new absolute positions: re-rotating a stored
pre-rotation K at positions ``p_new`` should be bit-equivalent to producing
the K via the standard Mistral forward at those same positions.
"""
from __future__ import annotations

import torch

from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralRotaryEmbedding,
    apply_rotary_pos_emb,
)

from cacheblend.blend import concat_chunk_kvs, recover_rope_k


def _tiny_config() -> MistralConfig:
    return MistralConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=512,
        vocab_size=100,
        head_dim=16,
    )


def test_rope_recovery_matches_direct_forward() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    rotary = MistralRotaryEmbedding(cfg)

    seq_len = 12
    head_dim = cfg.head_dim
    num_kv = cfg.num_key_value_heads
    k_pre = torch.randn(1, num_kv, seq_len, head_dim, dtype=torch.float32)
    new_positions = torch.arange(7, 7 + seq_len, dtype=torch.long).unsqueeze(0)

    # Path 1: our helper.
    k_recovered = recover_rope_k(k_pre, new_positions, rotary)

    # Path 2: direct apply_rotary_pos_emb at the same positions.
    cos, sin = rotary(k_pre, new_positions)
    _, k_direct = apply_rotary_pos_emb(torch.zeros_like(k_pre), k_pre, cos, sin)

    assert torch.allclose(k_recovered, k_direct, atol=1e-6)
    # Max abs diff well under the 1e-4 tolerance required by the spec.
    assert (k_recovered - k_direct).abs().max().item() < 1e-5


def test_rope_recovery_shape_preserved() -> None:
    cfg = _tiny_config()
    rotary = MistralRotaryEmbedding(cfg)
    k_pre = torch.randn(2, cfg.num_key_value_heads, 5, cfg.head_dim)
    new_positions = torch.arange(5).unsqueeze(0).expand(2, -1).contiguous()
    out = recover_rope_k(k_pre, new_positions, rotary)
    assert out.shape == k_pre.shape


def test_concat_chunk_kvs_round_trip() -> None:
    cfg = _tiny_config()
    rotary = MistralRotaryEmbedding(cfg)
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    chunks = [
        (torch.randn(1, num_kv, 4, head_dim), torch.randn(1, num_kv, 4, head_dim)),
        (torch.randn(1, num_kv, 3, head_dim), torch.randn(1, num_kv, 3, head_dim)),
    ]
    total_len = sum(k.shape[-2] for k, _ in chunks)
    positions = torch.arange(total_len).unsqueeze(0)
    k_cat, v_cat = concat_chunk_kvs(chunks, positions, rotary)
    assert k_cat.shape == (1, num_kv, total_len, head_dim)
    assert v_cat.shape == (1, num_kv, total_len, head_dim)
