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
