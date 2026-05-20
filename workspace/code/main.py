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
"""
from __future__ import annotations

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


def run_eval_cerebrium(
    mode: str = "smoke",
    n: int = 0,
    deviation_mode: str = "",
    config: str = "configs/default.yaml",
):
    """Run the CacheBlend eval grid on a Cerebrium GPU container.

    Parameters map directly to ``cerebrium run main.py::run_eval_cerebrium --<key> <value>``
    flags (and to JSON body keys when called as a deployed endpoint):

    * ``mode``           -- ``smoke`` (wikimqa, n=3, ratio 0.15) or ``full``.
    * ``n``              -- override examples-per-dataset (0 = use the YAML).
    * ``deviation_mode`` -- HKVD selector: ``v`` (paper), ``k`` (ablation default
                            in the YAML) or ``kv``. Empty = use the YAML.
    * ``config``         -- config path relative to this dir.
    """
    if deviation_mode and deviation_mode not in ("v", "k", "kv"):
        raise ValueError(f"unknown deviation_mode {deviation_mode!r} (use 'v', 'k' or 'kv')")

    # --- HF cache + results onto the persistent volume ---------------------
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cerebrium injects secrets as env vars. Accept either common HF var name.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    # Make the repo importable and run from its root (paths in the YAML are
    # relative to it).
    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)

    # GPU sanity check -- fail fast so we never silently bill for a CPU box.
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

    import yaml

    cfg_text = _build_config_text(mode, n, deviation_mode, config)
    cfg = yaml.safe_load(cfg_text)

    # Force results onto the persistent volume.
    cfg.setdefault("output", {})["results_dir"] = RESULTS_DIR
    # Resolve each dataset path: YAML paths are repo-root relative
    # ("workspace/code/data/<f>.json"). Prefer the copy uploaded to the volume's
    # DATA_DIR (the data files are too big to ship in the run tar); fall back to
    # a bundled copy under CODE_DIR if one happens to be present.
    for ds in cfg.get("datasets", {}).values():
        p = ds.get("path")
        if not p or os.path.isabs(p):
            continue
        rel = p.replace("workspace/code/", "", 1)
        vol_path = os.path.join(DATA_DIR, os.path.basename(rel))
        bundled_path = os.path.join(str(CODE_DIR), rel)
        ds["path"] = vol_path if os.path.exists(vol_path) else bundled_path

    tmp_path = "/tmp/cerebrium_run_eval_config.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

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
