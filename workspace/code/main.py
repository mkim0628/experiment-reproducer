"""Cerebrium entrypoint for the CacheBlend reproduction.

This is the Cerebrium analogue of ``run_eval_modal.py``. Where Modal uses
``modal run run_eval_modal.py --mode smoke`` to spin up an ephemeral GPU
container, run a function and stream the result back, Cerebrium offers the same
ergonomics with ``cerebrium run``:

    # ephemeral one-off run (like `modal run`): packages this directory, runs on
    # the GPU set in cerebrium.toml (ADA_L4), streams logs back, then tears down.
    cerebrium run main.py::run_cerebrium --mode smoke
    cerebrium run main.py::run_cerebrium --mode full --n 50
    cerebrium run main.py::run_cerebrium --mode full --deviation_mode v

    # OR deploy once as a persistent REST endpoint and POST to it:
    cerebrium deploy
    curl -X POST \
      https://api.aws.us-east-1.cerebrium.ai/v4/<PROJECT-ID>/cacheblend-eval/run_cerebrium \
      -H 'Authorization: Bearer <JWT_TOKEN>' \
      -H 'Content-Type: application/json' \
      --data '{"mode": "smoke"}'

``run_cerebrium`` measures BOTH accuracy and TTFT in one call (single-pass
cacheblend, one model load). The headline numbers per (dataset, strategy, ratio)
carry ``mean`` (accuracy) and ``ttft_ms_*`` / ``speedup_vs_recompute`` side by
side. See ``eval/run_eval.py``.

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


def _build_config_text(mode: str, n: int, deviation_mode: str, config: str,
                       ratios: str = "", budget_mode: str = "", thresholds: str = "",
                       min_frac: float = 0.0, max_frac: float = 1.0) -> str:
    """Read configs/default.yaml and apply the smoke/full overrides.

    Returns the resolved YAML *text* (mirrors run_eval_modal.main's logic so the
    two backends stay in lock-step). ``ratios`` (comma-separated, e.g.
    "0.1,0.15,0.2") overrides strategy.recompute_ratios for either mode.
    ``budget_mode="threshold"`` switches the sweep to the Stage-2 adaptive budget:
    ``thresholds`` (comma-separated tau, e.g. "0.3,0.5,0.7") + ``min_frac``/
    ``max_frac`` clamps then drive the per-example recompute budget.
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

    if ratios:
        parsed = [float(x) for x in str(ratios).split(",") if x.strip() != ""]
        if parsed:
            cfg.setdefault("strategy", {})["recompute_ratios"] = parsed

    if budget_mode == "threshold":
        strat = cfg.setdefault("strategy", {})
        strat["budget_mode"] = "threshold"
        taus = [float(x) for x in str(thresholds).split(",") if x.strip() != ""]
        strat["thresholds"] = taus or [0.5]
        strat["min_frac"] = float(min_frac)
        strat["max_frac"] = float(max_frac)
    elif budget_mode and budget_mode != "ratio":
        raise ValueError(f"unknown budget_mode {budget_mode!r} (use 'ratio' or 'threshold')")

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


def _resolve_config(mode: str, n: int, deviation_mode: str, config: str, tmp_path: str,
                    ratios: str = "", budget_mode: str = "", thresholds: str = "",
                    min_frac: float = 0.0, max_frac: float = 1.0) -> str:
    """Build the smoke/full config and rewrite paths onto the volume.

    Writes the resolved YAML to ``tmp_path`` (the path ``eval.run_eval``
    consumes) and returns it. Dataset paths in the YAML are repo-root relative
    ("workspace/code/data/<f>.json"); prefer the copy uploaded to the volume's
    DATA_DIR, falling back to a bundled copy under CODE_DIR if present.
    ``ratios`` overrides strategy.recompute_ratios (comma-separated);
    ``budget_mode``/``thresholds``/``min_frac``/``max_frac`` drive the Stage-2
    adaptive budget (see ``_build_config_text``).
    """
    import yaml

    cfg = yaml.safe_load(_build_config_text(
        mode, n, deviation_mode, config, ratios,
        budget_mode=budget_mode, thresholds=thresholds,
        min_frac=min_frac, max_frac=max_frac))
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


def run_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
    repeats: int = 3,
    warmup: int = 1,
    check_correctness: bool = False,
    ratios: str = "",
    ttft_only: bool = False,
    budget_mode: str = "",
    thresholds: str = "",
    min_frac: float = 0.0,
    max_frac: float = 1.0,
):
    """Run the CacheBlend eval (accuracy AND TTFT) on a Cerebrium GPU container.

    One entry point for the whole evaluation. ``eval.run_eval.run_combined``
    loads the model once (per CLAUDE.md cost rules) and measures TTFT FIRST on a
    clean device, THEN accuracy -- so the accuracy phase's full-length generation
    cannot perturb the TTFT numbers. Returns merged per-(dataset,strategy,ratio)
    rows carrying both ``mean`` (accuracy) and ``ttft_ms_*`` / ``speedup_vs_recompute``.

    ``cacheblend`` is the single-pass selective recompute: the SAME function is
    timed (TTFT) and scored (accuracy), so each row's acc/ttft come from one
    inference path -- the accuracy-drop vs TTFT-saving trade-off the paper
    reports, read off one implementation.

    * ``mode``           -- ``smoke`` (wikimqa, n=3) or ``full``.
    * ``n``              -- override examples-per-dataset (0 = use the YAML).
    * ``deviation_mode`` -- HKVD selector: ``v`` (paper), ``k`` (YAML default) or
                            ``kv``. Empty = use the YAML.
    * ``config``         -- config path relative to this dir.
    * ``repeats`` / ``warmup`` -- TTFT timed / warmup iterations per example.
    * ``check_correctness`` -- first verify single-pass r=1 reproduces a full
      forward (bit-exact first-token logits) before the measured run.
    * ``ratios``         -- comma-separated recompute ratios to sweep, e.g.
                            "0.1,0.15,0.2,0.4,0.6,0.8" (overrides the YAML grid).
    * ``ttft_only``      -- time TTFT only (skip the accuracy phase).
    * ``budget_mode``    -- "ratio" (default, fixed top-r%) or "threshold"
                            (Stage-2 adaptive: recompute tokens with importance
                            >= tau * per-example max).
    * ``thresholds``     -- comma-separated tau to sweep when budget_mode=threshold,
                            e.g. "0.3,0.5,0.7". ``min_frac``/``max_frac`` clamp the
                            realized per-example recompute fraction.

    The returned dict (and the JSON on the volume) includes an ``environment``
    block: GPU, CUDA/cuDNN, torch/transformers versions, model and dtype.
    """
    _prepare_container(deviation_mode)
    tmp_path = _resolve_config(mode, n, deviation_mode, config,
                               "/tmp/cerebrium_run_config.yaml", ratios=ratios,
                               budget_mode=budget_mode, thresholds=thresholds,
                               min_frac=min_frac, max_frac=max_frac)

    if ttft_only:
        from eval.run_eval import run_ttft_only

        summary = run_ttft_only(tmp_path, repeats=repeats, warmup=warmup)
    else:
        from eval.run_eval import run_combined

        summary = run_combined(tmp_path, repeats=repeats, warmup=warmup,
                               check_correctness=check_correctness)

    return {
        "mode": mode,
        "ttft_only": ttft_only,
        "environment": summary.get("environment", {}),
        "ttft": summary.get("ttft", {}),
        "phase_order": summary.get("phase_order", []),
        "isolation": summary.get("isolation", ""),
        "correctness": summary.get("correctness"),
        "results": summary.get("results", []),
        "config": summary.get("config", {}),
        "results_dir": RESULTS_DIR,
    }
