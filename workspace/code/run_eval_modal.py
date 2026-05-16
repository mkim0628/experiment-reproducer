"""Modal entrypoint for the CacheBlend quality eval.

Runs `eval.run_eval.run_eval(...)` on a Modal GPU container, with HF weights
and per-run outputs persisted on Modal Volumes so re-runs don't re-download.

Usage (from `workspace/code/`):

    # smoke (3 examples of wikimqa, single ratio, ~couple of minutes on L4)
    modal run run_eval_modal.py --mode smoke

    # full grid from configs/default.yaml
    modal run run_eval_modal.py --mode full

    # override GPU / ratios / dataset size
    modal run run_eval_modal.py --mode full --gpu A10G --n 50
    modal run run_eval_modal.py --mode smoke --gpu L4 --n 2

Cost-minimization choices baked in (see CLAUDE.md "Modal GPU cost rules"):
- Defaults to L4 (cheapest GPU that fits Mistral-7B fp16 with 24 GB headroom).
- HF weights cached on a Modal Volume so the second run skips ~14 GB download.
- One container loads the model once and runs the whole grid (no per-cell
  cold start).
- Container shuts down immediately after the entrypoint returns
  (`scaledown_window=60`, no `min_containers`).
- `enable_memory_snapshot=True` so post-import CPU state is restored from
  snapshot on cold start.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import modal


CODE_DIR = pathlib.Path(__file__).resolve().parent
REMOTE_CODE_DIR = "/root/code"
HF_CACHE_DIR = "/root/.cache/huggingface"
RESULTS_DIR = "/root/results"

# ---------------------------------------------------------------- image
# Pinning torch to the CUDA wheel index so the GPU container gets a CUDA build
# (debian_slim doesn't carry CUDA itself; Modal supplies the runtime).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.44,<5",
        "accelerate>=0.30",
        "rouge_score>=0.1.2",
        "numpy<2",
        "tqdm",
        "pyyaml",
        "cbor2",
        "sentencepiece",
        "protobuf",
    )
    # Make local repo importable inside the container.
    .add_local_dir(str(CODE_DIR), REMOTE_CODE_DIR)
)

# Persist HF model weights and per-run results across container restarts.
hf_cache_vol = modal.Volume.from_name("cacheblend-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cacheblend-results", create_if_missing=True)

app = modal.App("cacheblend-eval")


# --------------------------------------------------------------- GPU function
@app.function(
    image=image,
    gpu="L4",  # overridden by local_entrypoint via app.function.with_options(...)
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        RESULTS_DIR: results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 3,        # 3 h hard cap for the full grid
    scaledown_window=60,        # shut down 60 s after the call returns
    enable_memory_snapshot=True,
)
def run_eval_remote(config_yaml: str, override: dict | None = None) -> dict:
    """Execute the existing `eval.run_eval.run_eval` on this GPU container.

    `config_yaml` is the *contents* (not path) of the config so we don't have
    to ship config files separately. `override` is merged on top.
    """
    import yaml

    sys.path.insert(0, REMOTE_CODE_DIR)
    os.chdir(REMOTE_CODE_DIR)

    # HuggingFace cache → Volume.
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

    # GPU sanity check + log so we never silently bill for a CPU container.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available inside Modal container; aborting before "
            "we waste GPU billing time."
        )
    print(f"[modal] device={torch.cuda.get_device_name(0)} "
          f"cuda={torch.version.cuda} torch={torch.__version__}")

    cfg = yaml.safe_load(config_yaml)
    if override:
        _deep_merge(cfg, override)

    # Force the output directory onto the results volume.
    cfg.setdefault("output", {})["results_dir"] = RESULTS_DIR
    # Make sure data paths are absolute (they're relative in the YAML).
    for ds in cfg.get("datasets", {}).values():
        p = ds.get("path")
        if p and not os.path.isabs(p):
            # YAML paths are relative to repo root ("workspace/code/data/...").
            # Inside the container we mounted only `workspace/code/` at REMOTE_CODE_DIR,
            # so strip that prefix.
            ds["path"] = os.path.join(
                REMOTE_CODE_DIR,
                p.replace("workspace/code/", "", 1),
            )

    # Write the resolved config to a tmp path that eval.run_eval expects.
    tmp_path = "/tmp/run_eval_modal_config.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    from eval.run_eval import run_eval

    summary = run_eval(tmp_path)
    results_vol.commit()
    return summary


def _deep_merge(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if (
            isinstance(v, dict)
            and isinstance(base.get(k), dict)
        ):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# --------------------------------------------------------------- local entry
@app.local_entrypoint()
def main(
    mode: str = "smoke",
    gpu: str = "L4",
    config: str = "configs/default.yaml",
    n: int = 0,
    download_results: bool = True,
):
    """Local driver: build override, invoke the GPU function, download JSON.

    `mode`:
        smoke -> wikimqa only, 1 ratio, n=3
        full  -> exactly what's in the YAML
    `gpu` :
        Any Modal GPU spec ("T4", "L4", "A10G", "A100", "H100", "L40S",
        "A100-80GB"). L4 default = cheapest fp16-7B-friendly GPU.
    `n`   :
        Override examples-per-dataset across the whole grid (0 = use YAML).
    """
    config_path = (CODE_DIR / config).resolve()
    cfg_text = config_path.read_text(encoding="utf-8")

    override: dict = {}
    if mode == "smoke":
        override = {
            "datasets": {
                # Only wikimqa, tiny n
                "wikimqa": {
                    "path": "workspace/code/data/wikimqa_s.json",
                    "n": n or 3,
                    "metric": "f1",
                },
            },
            "strategy": {
                "recompute_ratios": [0.15],
            },
        }
        # Drop other datasets by replacing the whole dict.
        import yaml
        cfg_dict = yaml.safe_load(cfg_text)
        cfg_dict["datasets"] = override["datasets"]
        cfg_dict.setdefault("strategy", {}).update(override["strategy"])
        cfg_text = yaml.safe_dump(cfg_dict)
        override = {}
    elif mode == "full":
        if n:
            override = {"datasets": {}}
            import yaml
            for ds_name, ds_cfg in yaml.safe_load(cfg_text).get("datasets", {}).items():
                ds_cfg = dict(ds_cfg)
                ds_cfg["n"] = n
                override["datasets"][ds_name] = ds_cfg
    else:
        raise SystemExit(f"unknown --mode {mode!r} (use 'smoke' or 'full')")

    print(f"[local] launching run_eval on Modal GPU={gpu} mode={mode}")
    summary = run_eval_remote.with_options(gpu=gpu).remote(cfg_text, override)

    # Show a compact result table in the local terminal.
    print("\n=== results ===")
    for row in summary.get("results", []):
        ratio = row.get("ratio")
        ratio_s = f"r={ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        print(f"  {row['dataset']:<14} {row['strategy']:<16} {ratio_s:<8} "
              f"mean={row['mean']:.3f} n={row['n']}")

    if download_results:
        # Copy the latest JSON in the volume back to the local workspace.
        local_results = CODE_DIR.parent / "results"
        local_results.mkdir(parents=True, exist_ok=True)
        latest = _download_latest_result(local_results)
        if latest is not None:
            print(f"\n[local] downloaded -> {latest}")


def _download_latest_result(local_dir: pathlib.Path) -> pathlib.Path | None:
    """Pull the newest *.json from the results volume into `local_dir`."""
    listing = list(results_vol.iterdir("/"))
    if not listing:
        return None
    listing = [e for e in listing if e.path.endswith(".json")]
    if not listing:
        return None
    latest = max(listing, key=lambda e: getattr(e, "mtime", 0) or 0)
    out = local_dir / pathlib.Path(latest.path).name
    with out.open("wb") as f:
        for chunk in results_vol.read_file(latest.path):
            f.write(chunk)
    # Also dump a copy of the summary metadata for quick inspection.
    try:
        summary = json.loads(out.read_text())
        (local_dir / "LATEST.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return out
