"""The TTFT/accuracy join must not collapse different selection modes.

Stage-1 evaluates ``raw`` and ``attn_weighted`` cacheblend at the same ratios in
one run, so ``_merge`` keys on (dataset, strategy, ratio, selection); without the
selection component the two modes' rows at the same ratio would overwrite.
"""
from __future__ import annotations

from eval.run_eval import _merge


def test_merge_separates_selection_modes() -> None:
    acc_rows = [
        {"dataset": "d", "strategy": "full_recompute", "ratio": None, "mean": 0.5, "n": 3},
        {"dataset": "d", "strategy": "cacheblend", "ratio": 0.15,
         "selection": "raw", "mean": 0.40, "n": 3},
        {"dataset": "d", "strategy": "cacheblend", "ratio": 0.15,
         "selection": "attn_weighted", "mean": 0.46, "n": 3},
    ]
    ttft_rows = [
        {"dataset": "d", "strategy": "cacheblend", "ratio": 0.15, "selection": "raw",
         "ttft_ms_median": 100.0, "speedup_vs_recompute": 2.0, "n": 3},
        {"dataset": "d", "strategy": "cacheblend", "ratio": 0.15,
         "selection": "attn_weighted",
         "ttft_ms_median": 105.0, "speedup_vs_recompute": 1.9, "n": 3},
    ]
    merged = _merge(ttft_rows, acc_rows)
    cb = [r for r in merged if r["strategy"] == "cacheblend"]
    assert len(cb) == 2
    raw = next(r for r in cb if r["selection"] == "raw")
    aw = next(r for r in cb if r["selection"] == "attn_weighted")
    # Accuracy and TTFT joined onto the correct row, no overwrite.
    assert raw["mean"] == 0.40 and raw["ttft_ms_median"] == 100.0
    assert aw["mean"] == 0.46 and aw["ttft_ms_median"] == 105.0


def test_merge_keys_threshold_cells_by_tau() -> None:
    # Stage-2 threshold mode: ratio is None and the budget point is `threshold`.
    # Two tau cells for the same selection must NOT collapse, and mean_budget +
    # ttft must join onto the right cell.
    acc_rows = [
        {"dataset": "d", "strategy": "cacheblend", "ratio": None, "threshold": 0.3,
         "budget_mode": "threshold", "selection": "attn_weighted",
         "mean": 0.44, "mean_budget": 0.22, "n": 3},
        {"dataset": "d", "strategy": "cacheblend", "ratio": None, "threshold": 0.6,
         "budget_mode": "threshold", "selection": "attn_weighted",
         "mean": 0.40, "mean_budget": 0.09, "n": 3},
    ]
    ttft_rows = [
        {"dataset": "d", "strategy": "cacheblend", "ratio": None, "threshold": 0.3,
         "selection": "attn_weighted", "ttft_ms_median": 800.0,
         "speedup_vs_recompute": 3.0, "n": 3},
        {"dataset": "d", "strategy": "cacheblend", "ratio": None, "threshold": 0.6,
         "selection": "attn_weighted", "ttft_ms_median": 500.0,
         "speedup_vs_recompute": 4.8, "n": 3},
    ]
    merged = _merge(ttft_rows, acc_rows)
    cb = [r for r in merged if r["strategy"] == "cacheblend"]
    assert len(cb) == 2
    lo = next(r for r in cb if r["threshold"] == 0.3)
    hi = next(r for r in cb if r["threshold"] == 0.6)
    assert lo["mean"] == 0.44 and lo["mean_budget"] == 0.22 and lo["ttft_ms_median"] == 800.0
    assert hi["mean"] == 0.40 and hi["mean_budget"] == 0.09 and hi["ttft_ms_median"] == 500.0
    # Higher tau -> smaller realized budget -> bigger speedup.
    assert hi["speedup_vs_recompute"] > lo["speedup_vs_recompute"]
