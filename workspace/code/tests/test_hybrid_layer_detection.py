"""CPU-only tests for cacheblend.hybrid layer-detection logic.

These tests intentionally do NOT load any real HuggingFace model. They mock
the ``model.model.layers`` structure so we can verify that
``detect_attention_layers`` correctly distinguishes layers that carry K/V
projections from layers that don't (Mamba / DeltaNet / SSM placeholders).

The end-to-end ``cacheblend_generate_hybrid`` path requires a real model and
is exercised on Modal; here we only test the cheap parts that constrain the
overall correctness.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from cacheblend.hybrid import (
    _layer_attn_module,
    attention_layer_modules,
    detect_attention_layers,
)


def _make_attn(head_dim: int = 8, num_kv_heads: int = 2) -> nn.Module:
    """Minimal attention-like module that satisfies the K/V projection probe."""
    cfg = SimpleNamespace(num_key_value_heads=num_kv_heads)
    attn = nn.Module()
    attn.k_proj = nn.Linear(head_dim * num_kv_heads, head_dim * num_kv_heads)
    attn.v_proj = nn.Linear(head_dim * num_kv_heads, head_dim * num_kv_heads)
    attn.head_dim = head_dim
    attn.config = cfg
    return attn


def _make_mamba_like() -> nn.Module:
    """Mamba/DeltaNet-style layer module: has a mixer but no k_proj/v_proj."""
    mixer = nn.Module()
    mixer.in_proj = nn.Linear(16, 16)
    mixer.dt_proj = nn.Linear(8, 16)
    layer = nn.Module()
    layer.mixer = mixer  # name used by Mamba models in transformers
    return layer


def _layer_with_self_attn(attn: nn.Module) -> nn.Module:
    """Standard Llama/Mistral/Qwen style: layer.self_attn = ..."""
    layer = nn.Module()
    layer.self_attn = attn
    return layer


def _layer_with_attn(attn: nn.Module) -> nn.Module:
    """Zamba2-shared-transformer style: layer.attn = ..."""
    layer = nn.Module()
    layer.attn = attn
    return layer


def _make_model_with_layers(layers):
    inner = nn.Module()
    inner.layers = nn.ModuleList(layers)
    outer = nn.Module()
    outer.model = inner
    return outer


# ----------------------------------------------------------------- tests
def test_homogeneous_transformer_all_layers_detected():
    layers = [_layer_with_self_attn(_make_attn()) for _ in range(4)]
    model = _make_model_with_layers(layers)
    assert detect_attention_layers(model) == [0, 1, 2, 3]


def test_pure_mamba_no_attention_layers():
    layers = [_make_mamba_like() for _ in range(4)]
    model = _make_model_with_layers(layers)
    assert detect_attention_layers(model) == []


def test_qwen36_pattern_three_deltanet_one_attention():
    # 16 blocks of [D, D, D, A] = 64 layers. Attention at 3, 7, ..., 63.
    layers = []
    for _ in range(16):
        for _ in range(3):
            layers.append(_make_mamba_like())
        layers.append(_layer_with_self_attn(_make_attn()))
    model = _make_model_with_layers(layers)
    expected = [4 * b + 3 for b in range(16)]
    assert detect_attention_layers(model) == expected


def test_zamba2_pattern_alternate_naming():
    # Zamba2 uses layer.attn (not self_attn) on its shared transformer.
    # We mix both naming conventions to confirm the probe handles both.
    layers = []
    # 6 Mamba, then 1 attn-with-attn-name, repeated 3 times.
    for _ in range(3):
        for _ in range(6):
            layers.append(_make_mamba_like())
        layers.append(_layer_with_attn(_make_attn()))
    model = _make_model_with_layers(layers)
    assert detect_attention_layers(model) == [6, 13, 20]


def test_layer_attn_module_returns_the_module_or_none():
    attn = _make_attn()
    layer = _layer_with_self_attn(attn)
    assert _layer_attn_module(layer) is attn
    assert _layer_attn_module(_make_mamba_like()) is None


def test_attention_layer_modules_returns_index_module_pairs():
    layers = [
        _make_mamba_like(),
        _layer_with_self_attn(_make_attn()),
        _make_mamba_like(),
        _layer_with_attn(_make_attn()),
    ]
    model = _make_model_with_layers(layers)
    pairs = attention_layer_modules(model)
    assert [i for i, _ in pairs] == [1, 3]
    # Returned modules must be the same objects, not copies.
    assert pairs[0][1] is layers[1].self_attn
    assert pairs[1][1] is layers[3].attn


def test_layer_without_kproj_is_rejected_even_if_self_attn_exists():
    # An attention-shaped module that lacks k_proj must NOT be detected as
    # attention. This catches the failure mode where a hybrid model names
    # its SSM mixer "self_attn" out of convention.
    incomplete = nn.Module()
    incomplete.q_proj = nn.Linear(4, 4)  # no k_proj, no v_proj
    layer = nn.Module()
    layer.self_attn = incomplete
    model = _make_model_with_layers([layer])
    assert detect_attention_layers(model) == []


def test_complement_indices_basic():
    from cacheblend.hybrid import _complement_indices

    sel = torch.tensor([1, 3], dtype=torch.long)
    comp = _complement_indices(5, sel, device=torch.device("cpu"))
    assert comp.tolist() == [0, 2, 4]


def test_complement_indices_empty_selection_returns_full_range():
    from cacheblend.hybrid import _complement_indices

    sel = torch.empty(0, dtype=torch.long)
    comp = _complement_indices(4, sel, device=torch.device("cpu"))
    assert comp.tolist() == [0, 1, 2, 3]
