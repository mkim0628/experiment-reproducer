"""Tests for V-deviation reduction and top-r% HKVD selection."""
from __future__ import annotations

import math

import torch

import pytest

from cacheblend.selective_recompute import (
    BlendConfig,
    compute_k_deviation,
    compute_kv_deviation,
    compute_v_deviation,
    select_by_threshold,
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


# --------------------------------------------------- K-deviation (ablation)
def test_k_deviation_is_per_token_squared_l2() -> None:
    # Mirror of test_v_deviation_is_per_token_squared_l2 but on K.
    k_pre = torch.zeros(1, 2, 5, 4)
    k_new = k_pre.clone()
    k_new[0, 1, 3, 2] = 2.0  # diff 2 -> squared 4 on token 3
    dev = compute_k_deviation(k_new, k_pre)
    assert dev.shape == (5,)
    expected = torch.zeros(5)
    expected[3] = 4.0
    assert torch.allclose(dev, expected)


def test_kv_deviation_modes_match_underlying_reductions() -> None:
    torch.manual_seed(1)
    k_new = torch.randn(1, 2, 6, 4)
    k_pre = torch.randn(1, 2, 6, 4)
    v_new = torch.randn(1, 2, 6, 4)
    v_pre = torch.randn(1, 2, 6, 4)

    v_only = compute_kv_deviation(k_new, k_pre, v_new, v_pre, mode="v")
    k_only = compute_kv_deviation(k_new, k_pre, v_new, v_pre, mode="k")
    kv = compute_kv_deviation(k_new, k_pre, v_new, v_pre, mode="kv")

    assert torch.allclose(v_only, compute_v_deviation(v_new, v_pre))
    assert torch.allclose(k_only, compute_k_deviation(k_new, k_pre))
    assert torch.allclose(kv, k_only + v_only)


def test_kv_deviation_rejects_unknown_mode() -> None:
    t = torch.zeros(1, 1, 3, 2)
    with pytest.raises(ValueError):
        compute_kv_deviation(t, t, t, t, mode="qk")


# ----------------------------------------------- Stage-2 threshold budget
def test_threshold_selects_relative_to_max() -> None:
    score = torch.tensor([0.1, 1.0, 0.5, 0.9, 0.05])
    # tau=0.5 -> keep score >= 0.5*max(1.0)=0.5 -> tokens {1,2,3}, sorted.
    assert select_by_threshold(score, 0.5).tolist() == [1, 2, 3]
    # tau=0.95 -> only the peak (token 1).
    assert select_by_threshold(score, 0.95).tolist() == [1]
    # tau=0.0 -> everything (subject to default max_frac=1.0).
    assert select_by_threshold(score, 0.0).tolist() == [0, 1, 2, 3, 4]


def test_threshold_is_monotonic_in_tau() -> None:
    torch.manual_seed(0)
    score = torch.rand(20)
    counts = [select_by_threshold(score, t).numel() for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert counts == sorted(counts, reverse=True)  # higher tau -> fewer (or equal)


def test_threshold_min_frac_floor() -> None:
    # Only token 0 crosses tau, but min_frac forces a 3-token floor.
    score = torch.zeros(10)
    score[0] = 1.0
    out = select_by_threshold(score, 0.5, min_frac=0.3)  # ceil(0.3*10)=3
    assert out.numel() == 3
    assert 0 in out.tolist()  # the real peak is always included


def test_threshold_max_frac_cap() -> None:
    score = torch.ones(10)  # all equal -> all cross any tau<=1
    out = select_by_threshold(score, 0.5, max_frac=0.2)  # ceil(0.2*10)=2
    assert out.numel() == 2


def test_threshold_zero_signal_recomputes_floor_not_nothing() -> None:
    score = torch.zeros(8)
    # No positive signal: must not collapse to 0 tokens (that would be pure reuse).
    assert select_by_threshold(score, 0.5).numel() == 1               # default floor
    assert select_by_threshold(score, 0.5, min_frac=0.25).numel() == 2  # ceil(0.25*8)


def test_threshold_returns_sorted_and_validates() -> None:
    score = torch.tensor([0.2, 0.9, 0.4, 1.0])
    out = select_by_threshold(score, 0.3)
    assert torch.equal(out, torch.sort(out).values)
    with pytest.raises(ValueError):
        select_by_threshold(score, 1.5)
    with pytest.raises(ValueError):
        select_by_threshold(score.unsqueeze(0), 0.5)  # not 1-D


def test_blendconfig_validates_budget_mode_and_threshold() -> None:
    assert BlendConfig().budget_mode == "ratio"          # released default
    BlendConfig(budget_mode="threshold", threshold=0.5, min_frac=0.05, max_frac=0.8)
    with pytest.raises(ValueError):
        BlendConfig(budget_mode="adaptive")              # unknown mode
    with pytest.raises(ValueError):
        BlendConfig(threshold=1.5)
    with pytest.raises(ValueError):
        BlendConfig(min_frac=0.6, max_frac=0.4)          # min > max


def test_blendconfig_validates_deviation_mode() -> None:
    BlendConfig(deviation_mode="v")
    BlendConfig(deviation_mode="k")
    BlendConfig(deviation_mode="kv")
    with pytest.raises(ValueError):
        BlendConfig(deviation_mode="kq")


def test_blendconfig_selection_defaults_and_validation() -> None:
    cfg = BlendConfig()
    assert cfg.selection == "raw"           # released algorithm by default
    assert cfg.mass_source == "suffix"
    BlendConfig(selection="attn_weighted")
    with pytest.raises(ValueError):
        BlendConfig(selection="magic")
    with pytest.raises(ValueError):
        BlendConfig(mass_source="prefix")


def test_k_vs_v_deviation_can_pick_different_indices() -> None:
    # Construct a case where K diff is large at token 0 and V diff is large
    # at token 4 -> top-1 selection should differ between modes.
    seq = 5
    k_pre = torch.zeros(1, 1, seq, 4)
    v_pre = torch.zeros(1, 1, seq, 4)
    k_new = k_pre.clone()
    v_new = v_pre.clone()
    k_new[0, 0, 0, 0] = 10.0  # K-diff concentrated at token 0
    v_new[0, 0, 4, 0] = 10.0  # V-diff concentrated at token 4
    v_idx = select_hkvd_indices(compute_v_deviation(v_new, v_pre), r=0.2)
    k_idx = select_hkvd_indices(compute_k_deviation(k_new, k_pre), r=0.2)
    assert v_idx.tolist() == [4]
    assert k_idx.tolist() == [0]
    # "kv" mode aggregates and ties broken by topk -> picks one of {0,4}
    kv_idx = select_hkvd_indices(
        compute_kv_deviation(k_new, k_pre, v_new, v_pre, mode="kv"), r=0.2
    )
    assert kv_idx.tolist()[0] in (0, 4)
