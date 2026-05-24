"""Model-level tests for the Stage-0 analysis harness.

Uses a tiny RANDOM-weight Mistral built on CPU (no network, no download). Random
weights are fine here: every assertion is about *correctness/parity* of the
machinery (injection path == built-in path; instrumented forward == HF forward;
oracle score formula), not about whether attention is "good".
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from cacheblend.analysis import (
    LayerRecord,
    instrumented_full_forward,
    oracle_score,
    score_selectors,
)
from cacheblend.selective_recompute import BlendConfig, compute_kv_deviation
from cacheblend.single_pass import _selective_prefill


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = MistralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, sliding_window=None,
        rms_norm_eps=1e-5,
    )
    model = MistralForCausalLM(cfg).eval()
    return model


def _manual_fused_cache(model, C: int, seed: int = 1):
    """A DynamicCache of arbitrary (but fixed) per-layer chunk KV, shape-correct."""
    from transformers import DynamicCache

    g = torch.Generator().manual_seed(seed)
    n_kv = model.config.num_key_value_heads
    hd = getattr(model.model.layers[0].self_attn, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    cache = DynamicCache()
    for li in range(model.config.num_hidden_layers):
        k = torch.randn(1, n_kv, C, hd, generator=g)
        v = torch.randn(1, n_kv, C, hd, generator=g)
        cache.update(k, v, li)
    return cache


def test_instrumented_forward_matches_hf(tiny_model):
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    C = 8
    logits_inst, records = instrumented_full_forward(model, full_ids, C)
    logits_hf = model(full_ids, use_cache=False).logits[:, -1, :]
    assert torch.allclose(logits_inst.float(), logits_hf.float(), atol=1e-4, rtol=1e-3)
    # One record per layer; chunk KV sliced to C; mass length C and non-negative.
    assert len(records) == model.config.num_hidden_layers
    for rec in records:
        assert rec.k_chunk_true.shape[2] == C
        assert rec.attn_mass.shape == (C,)
        assert torch.all(rec.attn_mass >= 0)


def test_injection_parity_raw_equals_builtin(tiny_model):
    """selector=raw(mode) must reproduce the built-in (selector=None) path exactly."""
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    C = 8
    from cacheblend.selection import raw_deviation_selector

    for mode in ("v", "k", "kv"):
        cache_a = _manual_fused_cache(model, C, seed=2)
        cache_b = _manual_fused_cache(model, C, seed=2)
        cfg = BlendConfig(recompute_ratio=0.25, check_layer=1, deviation_mode=mode)
        _, logits_builtin, hkvd_builtin = _selective_prefill(
            model, cache_a, full_ids, C, cfg, build_cache=False)
        _, logits_inj, hkvd_inj = _selective_prefill(
            model, cache_b, full_ids, C, cfg, build_cache=False,
            selector=raw_deviation_selector(mode=mode))
        assert torch.equal(hkvd_builtin, hkvd_inj)
        assert torch.allclose(logits_builtin, logits_inj, atol=0, rtol=0)


def test_oracle_score_formula(tiny_model):
    """oracle_score == sum_layers mass * (deviation + eps), checked manually."""
    model = tiny_model
    C = 5
    L = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    hd = getattr(model.model.layers[0].self_attn, "head_dim", 8)
    fused = _manual_fused_cache(model, C, seed=3)

    g = torch.Generator().manual_seed(9)
    records = []
    for _ in range(L):
        records.append(LayerRecord(
            k_chunk_true=torch.randn(1, n_kv, C, hd, generator=g),
            v_chunk_true=torch.randn(1, n_kv, C, hd, generator=g),
            attn_mass=torch.rand(C, generator=g),
        ))
    eps = 1e-6
    score = oracle_score(records, fused, C, mode="v", eps=eps)

    expected = torch.zeros(C)
    for li, rec in enumerate(records):
        dev = compute_kv_deviation(
            rec.k_chunk_true, fused.layers[li].keys,
            rec.v_chunk_true, fused.layers[li].values, mode="v").float()
        expected += rec.attn_mass * (dev + eps)
    assert torch.allclose(score, expected, atol=1e-5)


def test_score_selectors_r1_matches_full_for_all_rules(tiny_model):
    """At r=1 every chunk token is recomputed -> full forward -> 0 deviation."""
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    C = 8
    fused = _manual_fused_cache(model, C, seed=4)
    out = score_selectors(
        model, fused, full_ids, C, ratios=[1.0],
        rules=("raw", "attn_weighted", "oracle", "random"),
        deviation_mode="v", check_layer=1)
    rows = out["rows"]
    assert len(rows) == 4
    for row in rows:
        assert row["ratio"] == 1.0
        assert row["n_selected"] == C
        assert row["argmax_match"] is True
        assert row["logit_l2"] < 1e-3


def test_cfg_attn_weighted_matches_explicit_selector(tiny_model):
    """cfg.selection='attn_weighted' must equal passing the selector explicitly."""
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    C = 8
    from cacheblend.selection import attention_weighted_selector

    cache_a = _manual_fused_cache(model, C, seed=6)
    cache_b = _manual_fused_cache(model, C, seed=6)
    cfg = BlendConfig(recompute_ratio=0.25, check_layer=1, deviation_mode="v",
                      selection="attn_weighted", mass_source="suffix")
    _, logits_cfg, hkvd_cfg = _selective_prefill(
        model, cache_a, full_ids, C, cfg, build_cache=False)
    cfg_raw = BlendConfig(recompute_ratio=0.25, check_layer=1, deviation_mode="v")
    _, logits_inj, hkvd_inj = _selective_prefill(
        model, cache_b, full_ids, C, cfg_raw, build_cache=False,
        selector=attention_weighted_selector(mode="v", mass_source="suffix"))
    assert torch.equal(hkvd_cfg, hkvd_inj)
    assert torch.allclose(logits_cfg, logits_inj, atol=0, rtol=0)


def test_cfg_attn_weighted_r1_matches_full(tiny_model):
    """At r=1 the attn_weighted cfg path is still a full forward -> matches GT."""
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    C = 8
    cache = _manual_fused_cache(model, C, seed=7)
    cfg = BlendConfig(recompute_ratio=1.0, check_layer=1, deviation_mode="v",
                      selection="attn_weighted")
    _, logits, hkvd = _selective_prefill(model, cache, full_ids, C, cfg, build_cache=False)
    gt = model(full_ids, use_cache=False).logits[:, -1, :].float()
    assert hkvd.numel() == C
    assert torch.argmax(logits.float(), -1).item() == torch.argmax(gt, -1).item()
    assert (logits.float() - gt).norm().item() < 1e-3


def test_score_selectors_rows_wellformed(tiny_model):
    model = tiny_model
    full_ids = torch.randint(0, model.config.vocab_size, (1, 14))
    C = 10
    fused = _manual_fused_cache(model, C, seed=5)
    out = score_selectors(
        model, fused, full_ids, C, ratios=[0.1, 0.3],
        rules=("raw", "attn_weighted", "oracle"), deviation_mode="v")
    assert out["chunk_len"] == C
    assert out["seq_len"] == 14
    assert len(out["rows"]) == 6  # 3 rules x 2 ratios
    for row in out["rows"]:
        assert 0 <= row["n_selected"] <= C
        assert row["logit_l2"] >= 0
