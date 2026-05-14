"""Tests for the in-place selective KV merge.

The merge must (a) overwrite K/V at the selected indices with the newly
computed values and (b) leave all other positions bit-identical.
"""
from __future__ import annotations

import torch

from cacheblend.selective_recompute import (
    BlendConfig,
    merge_selective_kv,
    select_hkvd_indices,
)


def _make_kv(seq_len: int = 8, num_kv: int = 2, head_dim: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, num_kv, seq_len, head_dim, generator=g)
    v = torch.randn(1, num_kv, seq_len, head_dim, generator=g)
    return k, v


def test_in_place_index_assignment_preserves_unselected() -> None:
    k_old, v_old = _make_kv(seed=1)
    k_new, v_new = _make_kv(seed=2)
    k_pre_snapshot = k_old.clone()
    v_pre_snapshot = v_old.clone()
    idx = torch.tensor([1, 4, 7], dtype=torch.long)
    merge_selective_kv(k_old, v_old, k_new, v_new, idx)
    selected_mask = torch.zeros(k_old.shape[2], dtype=torch.bool)
    selected_mask[idx] = True
    # Unselected positions must equal pre-merge values exactly.
    assert torch.equal(k_old[:, :, ~selected_mask, :], k_pre_snapshot[:, :, ~selected_mask, :])
    assert torch.equal(v_old[:, :, ~selected_mask, :], v_pre_snapshot[:, :, ~selected_mask, :])
    # Selected positions must equal k_new at those slots.
    assert torch.equal(k_old[:, :, idx, :], k_new[:, :, idx, :])
    assert torch.equal(v_old[:, :, idx, :], v_new[:, :, idx, :])


def test_zero_recompute_matches_pure_reuse() -> None:
    # r = 0 -> empty index set -> merge is a no-op -> K/V unchanged.
    k_old, v_old = _make_kv(seed=3)
    k_new, v_new = _make_kv(seed=4)
    k_ref = k_old.clone()
    v_ref = v_old.clone()
    idx = select_hkvd_indices(torch.zeros(k_old.shape[2]), r=0.0)
    merge_selective_kv(k_old, v_old, k_new, v_new, idx)
    assert torch.equal(k_old, k_ref)
    assert torch.equal(v_old, v_ref)


def test_full_recompute_matches_hf_forward() -> None:
    # r = 1 -> every token selected -> output equals k_new / v_new entirely.
    k_old, v_old = _make_kv(seed=5)
    k_new, v_new = _make_kv(seed=6)
    dev = torch.arange(k_old.shape[2], dtype=torch.float32)
    idx = select_hkvd_indices(dev, r=1.0)
    merge_selective_kv(k_old, v_old, k_new, v_new, idx)
    assert torch.equal(k_old, k_new)
    assert torch.equal(v_old, v_new)


def test_blend_config_defaults() -> None:
    cfg = BlendConfig()
    assert cfg.recompute_ratio == 0.15
    assert cfg.check_layer == 1
    assert cfg.schedule == "single_check"
