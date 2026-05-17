# Latency benchmark results

This directory holds JSON dumps produced by `run_latency_modal.py`.
Each file is one run: prefill TTFT (mean / p50 / p95 ms) for
`full_recompute`, `full_reuse`, and `cacheblend(r)` across every
dataset listed in the config.

Eval-quality results (F1 / Rouge-L) live under `workspace/results/`
and are gitignored.
