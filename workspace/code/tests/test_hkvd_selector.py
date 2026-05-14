"""Tests for V-deviation reduction and top-r% HKVD selection."""
from __future__ import annotations

import math

import torch

from cacheblend.selective_recompute import (
    compute_v_deviation,
    select_hkvd_indices,
)


def test_v_deviation_is_per_token_squared_l2() -> None:
    # 1 batch, 2 kv heads, 5 tokens, 4-dim heads. Construct V_new so that
    # only token 2 differs from V_pre by exactly 1 unit in one channel.
    v_pre = torch.zeros(1, 2, 5, 4)
    v_new = v_pre.clone()
    v_new[0, 0, 2, 1] = 1.0  # diff 1 -> squared 1
    dev = compute_v_deviation(v_new, v_pre)
    assert dev.shape == (5,)
    expected = torch.zeros(5)
    expected[2] = 1.0
    assert torch.allclose(dev, expected)


def test_v_deviation_matches_official_layout() -> None:
    # vllm_blend layout: (seq_len, num_kv_heads, head_dim). We reshape to the
    # (num_kv_heads, seq_len, head_dim) layout used here and confirm the same
    # per-token reduction.
    torch.manual_seed(0)
    seq_len, num_kv, hd = 7, 3, 8
    a = torch.randn(seq_len, num_kv, hd)
    b = torch.randn(seq_len, num_kv, hd)
    # Official: torch.sum((a-b)**2, dim=[1,2]) -> per-token.
    expected = torch.sum((a - b) ** 2, dim=[1, 2])
    # Ours expects (num_kv_heads, seq_len, head_dim).
    our = compute_v_deviation(a.permute(1, 0, 2), b.permute(1, 0, 2))
    assert torch.allclose(our, expected, atol=1e-5)


def test_topk_selection_with_known_values() -> None:
    dev = torch.tensor([0.1, 0.9, 0.5, 0.05, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6])
    # r = 0.3 -> ceil(0.3 * 10) = 3 indices -> values 0.9, 0.8, 0.7
    # -> positions 1, 4, 6.
    idx = select_hkvd_indices(dev, r=0.3)
    assert idx.tolist() == [1, 4, 6]


def test_topk_size_ceil_rule() -> None:
    dev = torch.arange(20, dtype=torch.float32)
    # r = 0.15 -> ceil(0.15 * 20) = 3
    assert len(select_hkvd_indices(dev, 0.15)) == 3
    # r = 0.18 -> ceil(0.18 * 20) = 4
    assert len(select_hkvd_indices(dev, 0.18)) == 4
    # r = 0.05 -> ceil(0.05 * 20) = 1
    assert len(select_hkvd_indices(dev, 0.05)) == 1
    # r = 0.0 -> empty
    assert len(select_hkvd_indices(dev, 0.0)) == 0
    # r = 1.0 -> all
    assert len(select_hkvd_indices(dev, 1.0)) == 20


def test_topk_ceil_for_arbitrary_n() -> None:
    # 7-token sequence with r = 0.18 -> ceil(0.18*7) = 2
    dev = torch.tensor([0.5, 0.1, 0.9, 0.3, 0.4, 0.7, 0.2])
    idx = select_hkvd_indices(dev, 0.18)
    assert math.ceil(0.18 * 7) == 2
    assert set(idx.tolist()) == {2, 5}  # top-2 by value
