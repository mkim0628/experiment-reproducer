"""Modal entrypoint for the TTFT (prefill latency) benchmark.

Self-contained Modal app that mirrors ``run_eval_modal.py`` (same image,
volumes, secret) but invokes ``scripts.run_latency`` instead of
``eval.run_eval``. We duplicate the image/volume definitions rather than
import from run_eval_modal because Modal places this script at /root/
while the rest of the repo is mounted at /root/code/, so a module-level
import would race the sys.path setup.

Usage (from ``workspace/code/``)::

    # smoke: nq_dpr only, 3 examples, 1 ratio (~couple of minutes on L4)
    modal run run_latency_modal.py --mode smoke

    # full: every dataset in configs/default.yaml that exists
    modal run run_latency_modal.py --mode full --n 10
    modal run run_latency_modal.py --mode full --gpu A10G --n 20
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
# Kept in sync with run_eval_modal.py so both apps share the same image
# cache (same torch + transformers pins).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==5.8.1",
        "accelerate>=0.30",
        "rouge_score>=0.1.2",
        "numpy<2",
        "tqdm",
        "pyyaml",
        "cbor2",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(
        str(CODE_DIR),
        REMOTE_CODE_DIR,
        ignore=["*.log", "__pycache__", "*.pyc", "results/", ".pytest_cache"],
    )
)

hf_cache_vol = modal.Volume.from_name("cacheblend-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cacheblend-results", create_if_missing=True)

app = modal.App("cacheblend-latency")


def _deep_merge(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# --------------------------------------------------------------- GPU class
@app.cls(
    image=image,
    gpu="L4",
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        RESULTS_DIR: results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 2,
    scaledown_window=60,
    enable_memory_snapshot=True,
)
class LatencyRunner:
    @modal.method()
    def run_latency(
        self,
        config_yaml: str,
        n_examples: int,
        warmup: int,
        repeats: int,
        override: dict | None = None,
    ) -> dict:
        import yaml

        sys.path.insert(0, REMOTE_CODE_DIR)
        os.chdir(REMOTE_CODE_DIR)

        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; aborting before billing.")
        print(
            f"[modal-latency] device={torch.cuda.get_device_name(0)} "
            f"cuda={torch.version.cuda} torch={torch.__version__}"
        )

        cfg = yaml.safe_load(config_yaml)
        if override:
            _deep_merge(cfg, override)
        cfg.setdefault("output", {})["results_dir"] = RESULTS_DIR
        for ds in cfg.get("datasets", {}).values():
            p = ds.get("path")
            if p and not os.path.isabs(p):
                ds["path"] = os.path.join(
                    REMOTE_CODE_DIR,
                    p.replace("workspace/code/", "", 1),
                )

        tmp_cfg = "/tmp/run_latency_modal_config.yaml"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        from scripts.run_latency import run_latency

        summary = run_latency(tmp_cfg, n_examples, warmup, repeats)

        import time as _t
        run_id = _t.strftime("%Y%m%d_%H%M%S")
        out_path = pathlib.Path(RESULTS_DIR) / f"latency_{run_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results_vol.commit()
        summary["_remote_path"] = str(out_path)
        return summary


# --------------------------------------------------------------- local entry
@app.local_entrypoint()
def main(
    mode: str = "smoke",
    gpu: str = "L4",
    config: str = "configs/default.yaml",
    n: int = 0,
    warmup: int = 2,
    repeats: int = 5,
    download_results: bool = True,
):
    """`mode` choices:

        smoke -> nq_dpr only, n=3, 1 ratio
        full  -> every dataset in the YAML that exists on disk
    """
    import yaml

    config_path = (CODE_DIR / config).resolve()
    cfg_text = config_path.read_text(encoding="utf-8")
    cfg_dict = yaml.safe_load(cfg_text)

    if mode == "smoke":
        cfg_dict["datasets"] = {
            "nq_dpr": {
                "path": "workspace/code/data/nq_dpr.json",
                "n": n or 3,
                "metric": "f1",
            }
        }
        cfg_dict.setdefault("strategy", {})["recompute_ratios"] = [0.15]
        cfg_text = yaml.safe_dump(cfg_dict)
        n_examples = n or 3
    elif mode == "full":
        n_examples = n or 10
        if n:
            for ds_cfg in cfg_dict.get("datasets", {}).values():
                ds_cfg["n"] = n
            cfg_text = yaml.safe_dump(cfg_dict)
    else:
        raise SystemExit(f"unknown --mode {mode!r} (use 'smoke' or 'full')")

    print(
        f"[local-latency] mode={mode} gpu={gpu} n_examples={n_examples} "
        f"warmup={warmup} repeats={repeats}"
    )
    Runner = LatencyRunner.with_options(gpu=gpu)
    summary = Runner().run_latency.remote(cfg_text, n_examples, warmup, repeats, None)

    # ----- Local pretty-print
    print("\n=== prefill TTFT (mean ms; speedup vs full_recompute) ===")
    for ds_name, ds in summary["datasets"].items():
        sz = ds["sizes"]
        print(
            f"\n[{ds_name}] avg_full_tokens={sz['avg_full_prompt_tokens']} "
            f"avg_chunk_tokens={sz['avg_total_chunk_tokens']} "
            f"avg_suffix={sz['avg_suffix_tokens']} "
            f"avg_chunks={sz['avg_num_chunks']} n={sz['n_examples_measured']}"
        )
        rows = []
        order = (
            ["full_recompute", "full_reuse"]
            + sorted(k for k in ds["strategies"] if k.startswith("cacheblend_"))
        )
        for k in order:
            v = ds["strategies"].get(k)
            if not v:
                continue
            speedup = v.get("speedup_over_full_recompute")
            sp_str = (
                f"{speedup:>5.2f}x"
                if speedup
                else ("  1.00x" if k == "full_recompute" else "    --")
            )
            rows.append(
                f"  {k:<18s} mean={v['mean_ms']:>7.1f}  "
                f"p50={v['p50_ms']:>7.1f}  p95={v['p95_ms']:>7.1f}  speedup={sp_str}"
            )
        print("\n".join(rows))

    if download_results:
        remote_path = summary.get("_remote_path", "")
        if remote_path:
            local_dir = CODE_DIR.parent / "results"
            local_dir.mkdir(parents=True, exist_ok=True)
            fname = pathlib.Path(remote_path).name
            local_path = local_dir / fname
            try:
                with local_path.open("wb") as f:
                    for chunk in results_vol.read_file(
                        remote_path.replace(RESULTS_DIR, "").lstrip("/")
                    ):
                        f.write(chunk)
                print(f"\n[local] downloaded -> {local_path}")
            except Exception as e:
                print(f"\n[local] download failed: {e}")
