"""Cerebrium entrypoint for the CacheBlend reproduction.

This is the Cerebrium analogue of ``run_eval_modal.py``. Where Modal uses
``modal run run_eval_modal.py --mode smoke`` to spin up an ephemeral GPU
container, run a function and stream the result back, Cerebrium offers the same
ergonomics with ``cerebrium run``:

    # ephemeral one-off run (like `modal run`): packages this directory, runs on
    # the GPU set in cerebrium.toml (ADA_L4), streams logs back, then tears down.
    cerebrium run main.py::run_eval_cerebrium --mode smoke
    cerebrium run main.py::run_eval_cerebrium --mode full --n 50
    cerebrium run main.py::run_eval_cerebrium --mode full --deviation_mode v

    # OR deploy once as a persistent REST endpoint and POST to it:
    cerebrium deploy
    curl -X POST \
      https://api.aws.us-east-1.cerebrium.ai/v4/<PROJECT-ID>/cacheblend-eval/run_eval_cerebrium \
      -H 'Authorization: Bearer <JWT_TOKEN>' \
      -H 'Content-Type: application/json' \
      --data '{"mode": "smoke"}'

Results
-------
The full per-run JSON is written to the 50 GB persistent volume at
``/persistent-storage/cacheblend-results/<run_id>.json`` (survives container
restarts and is shared across runs). The function ALSO returns a compact summary
(and prints a table) so you see the headline numbers in the run logs / HTTP
response without downloading anything. Pull the full JSON afterwards with:

    cerebrium ls cacheblend-results
    cerebrium download cacheblend-results/<run_id>.json

Cost choices (see CLAUDE.md "Modal GPU cost rules" -- same logic on Cerebrium)
-----------------------------------------------------------------------------
* GPU class (ADA_L4) is set in cerebrium.toml, NOT per call -- unlike Modal,
  Cerebrium has no per-invocation GPU override. To change GPU, edit the toml.
* HF weights cached under /persistent-storage so re-runs skip the ~14 GB
  Mistral download.
* min_replicas=0 + cooldown=30 in the toml -> no idle GPU billing after a run.
* The model is loaded once per call and the whole (dataset x strategy x ratio)
  grid runs in that single call -- no per-cell cold start.

Required setup the caller must provide
--------------------------------------
* ``cerebrium login`` (interactive, once).
* A Cerebrium plan with L4 access (Hobby+).
* A HuggingFace token with access to the gated ``mistralai/Mistral-7B-Instruct-v0.2``
  repo, stored as a Cerebrium secret named ``HF_TOKEN``:
      cerebrium secrets set HF_TOKEN hf_xxx
  Cerebrium injects secrets as environment variables, which this module reads.

NOTE: do NOT add ``from __future__ import annotations`` here. Cerebrium binds
request params by introspecting this function's signature and calling
``isinstance(value, annotation)``; PEP 563 would turn the annotations into
strings and break that with "isinstance() arg 2 must be a type".
"""
import json
import os
import pathlib
import sys

CODE_DIR = pathlib.Path(__file__).resolve().parent
PERSIST = "/persistent-storage"
HF_CACHE_DIR = os.path.join(PERSIST, "hf-cache")
RESULTS_DIR = os.path.join(PERSIST, "cacheblend-results")
# Datasets live on the volume, NOT in the `cerebrium run` tar: the bundled
# JSONs total ~26 MB and `cerebrium run` caps the upload tar at 4 MB. Upload
# them once with `cerebrium cp data/<f>.json cacheblend-data/<f>.json`.
DATA_DIR = os.path.join(PERSIST, "cacheblend-data")


def _deep_merge(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _build_config_text(mode: str, n: int, deviation_mode: str, config: str) -> str:
    """Read configs/default.yaml and apply the smoke/full overrides.

    Returns the resolved YAML *text* (mirrors run_eval_modal.main's logic so the
    two backends stay in lock-step).
    """
    import yaml

    config_path = (CODE_DIR / config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if mode == "smoke":
        # wikimqa only, tiny n, single recompute ratio.
        cfg["datasets"] = {
            "wikimqa": {
                "path": "workspace/code/data/wikimqa_s.json",
                "n": n or 3,
                "metric": "f1",
            }
        }
        cfg.setdefault("strategy", {})["recompute_ratios"] = [0.15]
        if deviation_mode:
            cfg["strategy"]["deviation_mode"] = deviation_mode
    elif mode == "full":
        if n:
            for ds_cfg in cfg.get("datasets", {}).values():
                ds_cfg["n"] = n
        if deviation_mode:
            cfg.setdefault("strategy", {})["deviation_mode"] = deviation_mode
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'smoke' or 'full')")

    return yaml.safe_dump(cfg)


def _prepare_container(deviation_mode: str) -> None:
    """Shared per-call container setup for both eval and TTFT entrypoints.

    Points HF cache + results at the persistent volume, surfaces the HF token
    Cerebrium injects as a secret, makes the repo importable, and fails fast if
    there is no GPU (so we never silently bill for a CPU box).
    """
    if deviation_mode and deviation_mode not in ("v", "k", "kv"):
        raise ValueError(f"unknown deviation_mode {deviation_mode!r} (use 'v', 'k' or 'kv')")

    # Reduce CUDA caching-allocator fragmentation. The single-pass validation
    # cycles through several full-prompt KV caches per example; without this the
    # allocator can hold ~enough freed-but-non-contiguous blocks to fail a small
    # alloc on a 24 GB L4. Must be set before torch initializes CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available inside the Cerebrium container; aborting before "
            "wasting GPU billing time. Check compute=ADA_L4 / gpu_count=1 in cerebrium.toml."
        )
    print(
        f"[cerebrium] device={torch.cuda.get_device_name(0)} "
        f"cuda={torch.version.cuda} torch={torch.__version__}"
    )


def _resolve_config(mode: str, n: int, deviation_mode: str, config: str, tmp_path: str) -> str:
    """Build the smoke/full config and rewrite paths onto the volume.

    Writes the resolved YAML to ``tmp_path`` (the path ``run_eval`` / ``run_ttft``
    consume) and returns it. Dataset paths in the YAML are repo-root relative
    ("workspace/code/data/<f>.json"); prefer the copy uploaded to the volume's
    DATA_DIR, falling back to a bundled copy under CODE_DIR if present.
    """
    import yaml

    cfg = yaml.safe_load(_build_config_text(mode, n, deviation_mode, config))
    cfg.setdefault("output", {})["results_dir"] = RESULTS_DIR
    for ds in cfg.get("datasets", {}).values():
        p = ds.get("path")
        if not p or os.path.isabs(p):
            continue
        rel = p.replace("workspace/code/", "", 1)
        vol_path = os.path.join(DATA_DIR, os.path.basename(rel))
        bundled_path = os.path.join(str(CODE_DIR), rel)
        ds["path"] = vol_path if os.path.exists(vol_path) else bundled_path

    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return tmp_path


def run_eval_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
):
    """Run the CacheBlend quality eval grid on a Cerebrium GPU container.

    Parameters map directly to ``cerebrium run main.py::run_eval_cerebrium --<key> <value>``
    flags (and to JSON body keys when called as a deployed endpoint):

    * ``mode``           -- ``smoke`` (wikimqa, n=3, ratio 0.15) or ``full``.
    * ``n``              -- override examples-per-dataset (0 = use the YAML).
    * ``deviation_mode`` -- HKVD selector: ``v`` (paper), ``k`` (ablation default
                            in the YAML) or ``kv``. Empty = use the YAML.
    * ``config``         -- config path relative to this dir.
    """
    _prepare_container(deviation_mode)
    tmp_path = _resolve_config(mode, n, deviation_mode, config,
                               "/tmp/cerebrium_run_eval_config.yaml")

    from eval.run_eval import run_eval

    summary = run_eval(tmp_path)

    # Compact table in the logs / response.
    print("\n=== results ===")
    for row in summary.get("results", []):
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        print(
            f"  {row['dataset']:<14} {row['strategy']:<16} {ratio_s:<8} "
            f"mean={row['mean']:.3f} n={row['n']}"
        )

    # Return summary stats only (the full JSON already lives on the volume) so the
    # HTTP/response payload stays small.
    return {
        "mode": mode,
        "results": summary.get("results", []),
        "config": summary.get("config", {}),
        "results_dir": RESULTS_DIR,
    }


def run_ttft_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
    repeats: int = 3,
    warmup: int = 1,
):
    """Measure time-to-first-token (TTFT) on a Cerebrium GPU container.

    The latency analogue of ``run_eval_cerebrium``: same smoke/full config
    handling, but runs ``eval.run_ttft`` and returns per-(dataset,strategy,ratio)
    TTFT in milliseconds plus speedup-vs-recompute.

    * ``mode`` / ``n`` / ``deviation_mode`` / ``config`` -- as run_eval_cerebrium.
    * ``repeats`` -- timed iterations per example (median taken).
    * ``warmup``  -- untimed warmup iterations (also pre-warms the chunk store).

    NOTE: this reproduction's ``cacheblend`` is a two-pass implementation, so its
    TTFT is an upper bound, NOT the paper's single-pass selective-recompute
    latency. The full_recompute vs full_reuse comparison is faithful.
    """
    _prepare_container(deviation_mode)
    tmp_path = _resolve_config(mode, n, deviation_mode, config,
                               "/tmp/cerebrium_run_ttft_config.yaml")

    from eval.run_ttft import run_ttft

    summary = run_ttft(tmp_path, repeats=repeats, warmup=warmup)

    print("\n=== TTFT (ms) ===")
    for row in summary.get("results", []):
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        print(
            f"  {row['dataset']:<14} {row['strategy']:<16} {ratio_s:<8} "
            f"median={row['ttft_ms_median']:.1f}ms p90={row['ttft_ms_p90']:.1f}ms "
            f"speedup={row['speedup_vs_recompute']:.2f}x n={row['n']}"
        )

    return {
        "mode": mode,
        "ttft": summary.get("ttft", {}),
        "results": summary.get("results", []),
        "config": summary.get("config", {}),
        "results_dir": RESULTS_DIR,
    }


def run_combined_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
    repeats: int = 3,
    warmup: int = 1,
):
    """Measure accuracy AND TTFT in one Cerebrium GPU call, phases isolated.

    Runs ``eval.run_combined``, which loads the model once (per CLAUDE.md cost
    rules) and measures TTFT FIRST on a clean device, THEN accuracy -- so the
    accuracy phase's full-length generation cannot perturb the TTFT numbers.
    Returns merged per-(dataset,strategy,ratio) rows carrying both ``mean``
    (accuracy) and ``ttft_ms_*`` / ``speedup_vs_recompute``.

    * ``mode`` / ``n`` / ``deviation_mode`` / ``config`` -- as run_eval_cerebrium.
    * ``repeats`` / ``warmup`` -- TTFT timed / warmup iterations, as run_ttft_cerebrium.

    NOTE: cacheblend's TTFT here is a two-pass upper bound, not the paper's
    single-pass selective-recompute latency. Accuracy is faithful for all three.
    """
    _prepare_container(deviation_mode)
    tmp_path = _resolve_config(mode, n, deviation_mode, config,
                               "/tmp/cerebrium_run_combined_config.yaml")

    from eval.run_combined import run_combined

    summary = run_combined(tmp_path, repeats=repeats, warmup=warmup)

    print("\n=== accuracy + TTFT ===")
    for row in summary.get("results", []):
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        mean = row.get("mean")
        mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else "n/a"
        ttft = row.get("ttft_ms_median")
        ttft_s = f"{ttft:.1f}ms" if isinstance(ttft, (int, float)) else "n/a"
        spd = row.get("speedup_vs_recompute")
        spd_s = f"{spd:.2f}x" if isinstance(spd, (int, float)) else "n/a"
        print(
            f"  {row['dataset']:<14} {row['strategy']:<16} {ratio_s:<8} "
            f"acc={mean_s} ttft={ttft_s} ({spd_s}) n={row.get('n')}"
        )

    return {
        "mode": mode,
        "ttft": summary.get("ttft", {}),
        "phase_order": summary.get("phase_order", []),
        "isolation": summary.get("isolation", ""),
        "results": summary.get("results", []),
        "config": summary.get("config", {}),
        "results_dir": RESULTS_DIR,
    }


def run_singlepass_validate_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
    repeats: int = 3,
    warmup: int = 1,
    max_new_tokens: int = 32,
    compare_twopass: bool = False,
):
    """Validate the TRUE single-pass selective recompute and report its TTFT.

    Runs ``eval.validate_singlepass``, which (1) asserts single-pass at r=1.0
    reproduces full_recompute token-for-token, (2) cross-checks single-pass vs
    two-pass F1 at the config ratios, and (3) reports single-pass TTFT and its
    speedup over full_recompute -- the paper-style selective-recompute latency
    the two-pass path could not measure.

    * ``mode`` / ``n`` / ``deviation_mode`` / ``config`` -- as run_eval_cerebrium.
    * ``repeats`` / ``warmup`` -- TTFT timed / warmup iterations.
    * ``max_new_tokens`` -- decode length for the r=1 exact-match + F1 checks.
    """
    _prepare_container(deviation_mode)
    tmp_path = _resolve_config(mode, n, deviation_mode, config,
                               "/tmp/cerebrium_singlepass_validate_config.yaml")

    import json as _json
    import time as _time

    import yaml as _yaml

    from eval.run_eval import _load_model, _set_seed
    from eval.validate_singlepass import validate_singlepass

    cfg = _yaml.safe_load(open(tmp_path, "r", encoding="utf-8"))
    _set_seed(cfg.get("seed", 42))
    model, tokenizer, device, dtype = _load_model(cfg)
    summary = validate_singlepass(model, tokenizer, dtype, cfg,
                                  repeats=repeats, warmup=warmup,
                                  max_new_tokens=max_new_tokens,
                                  compare_twopass=compare_twopass)

    run_id = _time.strftime("%Y%m%d_%H%M%S") + "_singlepass_validate"
    out_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2)
    print(f"[cerebrium] wrote {out_path}")

    print("\n=== single-pass selective recompute ===")
    for ds in summary.get("results", []):
        print(f"  [{ds['dataset']}] r1 first-token match: "
              f"{ds['r1_first_token_matches_full_forward']}  "
              f"max|logit diff|={ds['r1_max_logit_diff_max']:.4f}  "
              f"full_recompute TTFT={ds['ttft_full_recompute_median_ms']:.1f}ms")
        for r, row in ds.get("cacheblend_selective", {}).items():
            f2 = row.get("f1_twopass")
            f2s = f"{f2:.3f}" if isinstance(f2, (int, float)) else "n/a"
            print(f"    r={r}: TTFT={row['ttft_ms_median']:.1f}ms "
                  f"({row['speedup_vs_recompute']:.2f}x)  "
                  f"f1_single={row['f1_singlepass']:.3f} f1_two={f2s}")

    return {
        "mode": mode,
        "results": summary.get("results", []),
        "config": summary.get("config", {}),
        "results_dir": RESULTS_DIR,
    }
