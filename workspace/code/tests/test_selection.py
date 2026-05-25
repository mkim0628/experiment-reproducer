"""Tests for the pluggable selection rules (no model required).

These exercise the selector logic with synthetic SelectionContexts on CPU, so
they are GPU-cost-free and run in CI. Model-level parity (injected raw selector
== built-in path) lives in test_selection_analysis.py.
"""
from __future__ import annotations

import torch

from cacheblend.selection import (
    SelectionContext,
    attention_weighted_selector,
    chunk_attention_mass,
    importance_score,
    precomputed_score_selector,
    raw_deviation_selector,
    random_selector,
    threshold_selector,
)
from cacheblend.selective_recompute import (
    compute_kv_deviation,
    select_by_threshold,
    select_hkvd_indices,
)


def _ctx(C: int = 6, suffix_len: int = 2, n_kv: int = 2, n_rep: int = 2,
         head_dim: int = 4, seed: int = 0) -> SelectionContext:
    """Build a synthetic check-layer context with C chunk + suffix_len tokens."""
    g = torch.Generator().manual_seed(seed)
    T = C + suffix_len
    n_heads = n_kv * n_rep
    k_fresh = torch.randn(1, n_kv, C, head_dim, generator=g)
    v_fresh = torch.randn(1, n_kv, C, head_dim, generator=g)
    k_cached = torch.randn(1, n_kv, C, head_dim, generator=g)
    v_cached = torch.randn(1, n_kv, C, head_dim, generator=g)
    q_all = torch.randn(1, n_heads, T, head_dim, generator=g)
    k_full = torch.randn(1, n_kv, T, head_dim, generator=g)
    v_full = torch.randn(1, n_kv, T, head_dim, generator=g)
    return SelectionContext(
        k_fresh_chunk=k_fresh, v_fresh_chunk=v_fresh,
        k_cached_chunk=k_cached, v_cached_chunk=v_cached,
        q_all=q_all, k_full=k_full, v_full=v_full,
        positions=torch.arange(T), suffix_idx=torch.arange(C, T),
        scale=head_dim ** -0.5, n_rep=n_rep,
    )


def test_raw_selector_matches_builtin_topk() -> None:
    ctx = _ctx(seed=1)
    for mode in ("v", "k", "kv"):
        dev = compute_kv_deviation(ctx.k_fresh_chunk, ctx.k_cached_chunk,
                                   ctx.v_fresh_chunk, ctx.v_cached_chunk, mode=mode)
        expected = select_hkvd_indices(dev, 0.5)
        got = raw_deviation_selector(mode=mode)(ctx, 0.5)
        assert torch.equal(got, expected)


def test_attention_mass_shape_and_nonneg() -> None:
    ctx = _ctx(C=6, suffix_len=2, seed=2)
    mass = chunk_attention_mass(ctx, mass_source="suffix")
    assert mass.shape == (6,)
    assert torch.all(mass >= 0)
    # Suffix has 2 query rows; each row's probs sum to 1 over all keys, so the
    # total mass over all keys (chunk+suffix) per row is 1 -> chunk mass <= n_rows.
    assert mass.sum() <= 2.0 + 1e-4


def test_attention_weighted_differs_from_raw_when_mass_concentrates() -> None:
    # Construct a context where raw deviation peaks at token 0 but attention mass
    # is forced onto token 5, so the weighted rule should prefer token 5.
    C, suffix_len, head_dim = 6, 1, 4
    ctx = _ctx(C=C, suffix_len=suffix_len, n_kv=1, n_rep=1, head_dim=head_dim, seed=3)
    # Make V deviation largest at token 0.
    ctx.v_cached_chunk.zero_()
    ctx.v_fresh_chunk.zero_()
    ctx.v_fresh_chunk[0, 0, 0, 0] = 10.0   # huge raw deviation at token 0
    ctx.v_fresh_chunk[0, 0, 5, 0] = 1.0    # small raw deviation at token 5
    # Force the suffix query to attend almost entirely to token 5: make key 5
    # align with the query, others orthogonal/negative.
    ctx.q_all.zero_()
    ctx.k_full.zero_()
    ctx.q_all[0, 0, -1, 0] = 10.0          # suffix query points along dim 0
    ctx.k_full[0, 0, 5, 0] = 10.0          # only key 5 has positive dim-0 -> wins softmax
    raw = raw_deviation_selector(mode="v")(ctx, 1.0 / C)   # top-1
    aw = attention_weighted_selector(mode="v", mass_source="suffix")(ctx, 1.0 / C)
    assert raw.tolist() == [0]
    assert aw.tolist() == [5]


def test_precomputed_score_selector_uses_injected_score() -> None:
    ctx = _ctx(C=6, seed=4)
    score = torch.tensor([0.1, 0.0, 0.9, 0.2, 0.8, 0.05])
    ctx.extra["score"] = score
    got = precomputed_score_selector()(ctx, 2.0 / 6)  # top-2 -> tokens 2, 4
    assert got.tolist() == [2, 4]


def test_importance_score_matches_selector_ranking() -> None:
    # The shared score must rank the same tokens the fixed-budget selectors pick.
    ctx = _ctx(C=6, suffix_len=2, seed=6)
    for sel in ("raw", "attn_weighted"):
        s = importance_score(ctx, sel, mode="v", mass_source="suffix")
        top = select_hkvd_indices(s, 0.5)
        if sel == "raw":
            got = raw_deviation_selector(mode="v")(ctx, 0.5)
        else:
            got = attention_weighted_selector(mode="v", mass_source="suffix")(ctx, 0.5)
        assert torch.equal(top, got)


def test_threshold_selector_matches_threshold_on_score() -> None:
    # threshold_selector == select_by_threshold applied to the shared score.
    ctx = _ctx(C=8, suffix_len=2, seed=7)
    for sel in ("raw", "attn_weighted"):
        s = importance_score(ctx, sel, mode="v", mass_source="suffix")
        expected = select_by_threshold(s, 0.4, min_frac=0.1, max_frac=0.9)
        got = threshold_selector(selection=sel, mode="v", mass_source="suffix",
                                 tau=0.4, min_frac=0.1, max_frac=0.9)(ctx, 0.0)
        assert torch.equal(got, expected)


def test_threshold_selector_composes_with_mass() -> None:
    # Same construction as the raw-vs-weighted test: raw peaks at token 0, mass on
    # token 5. A high tau (top-1 budget) must pick 0 for raw, 5 for attn_weighted.
    C, suffix_len, head_dim = 6, 1, 4
    ctx = _ctx(C=C, suffix_len=suffix_len, n_kv=1, n_rep=1, head_dim=head_dim, seed=3)
    ctx.v_cached_chunk.zero_(); ctx.v_fresh_chunk.zero_()
    ctx.v_fresh_chunk[0, 0, 0, 0] = 10.0
    ctx.v_fresh_chunk[0, 0, 5, 0] = 1.0
    ctx.q_all.zero_(); ctx.k_full.zero_()
    ctx.q_all[0, 0, -1, 0] = 10.0
    ctx.k_full[0, 0, 5, 0] = 10.0
    raw = threshold_selector(selection="raw", mode="v", tau=0.99)(ctx, 0.0)
    aw = threshold_selector(selection="attn_weighted", mode="v",
                            mass_source="suffix", tau=0.99)(ctx, 0.0)
    assert raw.tolist() == [0]
    assert aw.tolist() == [5]


def test_random_selector_count_and_determinism() -> None:
    ctx = _ctx(C=10, seed=5)
    a = random_selector(seed=7)(ctx, 0.3)   # ceil(0.3*10)=3
    b = random_selector(seed=7)(ctx, 0.3)
    assert a.numel() == 3
    assert torch.equal(a, b)                # same seed -> deterministic
    assert torch.equal(a, torch.sort(a).values)
