# Running the CacheBlend eval on a Cerebrium GPU

This is the operations guide for running the CacheBlend reproduction on a
**Cerebrium serverless GPU**. It documents the exact, verified workflow used to
get a smoke run returning real F1 numbers, the cost controls, and how to pick
the GPU.

> **Scope note.** Only this guide currently lives under `cerebrium/`. The actual
> deploy artifacts still live one level up in `workspace/code/` (a hard
> requirement of Cerebrium's default runtime — see *Why these files can't move
> yet* at the bottom):
>
> | File | Role |
> |---|---|
> | `workspace/code/main.py` | Cerebrium entrypoint (`run_eval_cerebrium`) |
> | `workspace/code/cerebrium.toml` | App config: GPU, scaling, deps |
> | `workspace/code/scripts/run_cerebrium.sh` | Async invoke wrapper (webhook-aware) |

---

## TL;DR

```bash
cd workspace/code

# one-time
pip install cerebrium
cerebrium secrets add "HF_TOKEN=$HF_TOKEN"          # gated Mistral repo
cerebrium cp data/wikimqa_s.json cacheblend-data/wikimqa_s.json   # + the other 3
cerebrium deploy -y                                  # build + deploy on the GPU in the toml

# run (async) + get results
scripts/run_cerebrium.sh --mode smoke                # or: --mode full --n 200
cerebrium ls cacheblend-results/
cerebrium download cacheblend-results/<id>.json /tmp/r.json
```

The deployed app, project and endpoint for this repo:

| | |
|---|---|
| Project | `p-238b3475` |
| App | `cacheblend-eval` |
| Function | `run_eval_cerebrium` |
| Endpoint | `https://api.aws.us-east-1.cerebrium.ai/v4/p-238b3475/cacheblend-eval/run_eval_cerebrium` |

---

## Why deploy + async (and NOT `cerebrium run`)

`cerebrium run main.py::run_eval_cerebrium --mode smoke` is the obvious
ephemeral path, but it **does not work for this workload**:

1. **4 MB tar limit.** `cerebrium run` caps the uploaded code+data tar at 4 MB.
   The bundled datasets are ~26 MB. (Fixed by moving datasets to the volume —
   see *Datasets*.)
2. **No persistent volume.** Ephemeral runs do **not** mount the project volume,
   so the ~14 GB Mistral download never persists. Every run is cold and
   re-downloads.
3. **Fixed polling timeout.** The CLI stops polling after ~5–7 min (not
   configurable), which is shorter than cold-start + 14 GB download + eval, so
   it returns `⚠️ Polling timeout reached` with no result.

`cerebrium deploy` avoids all three: it creates a **persistent app that mounts
the 50 GB volume**, so the model is cached once in `hf-cache/`, and you invoke it
**async** (fire-and-forget) and read results off the volume. This is the
supported path for long GPU jobs.

---

## One-time setup

### 1. Install + authenticate

```bash
pip install cerebrium
```

Auth (pick one):

- **Managed/web env:** `CEREBRIUM_SERVICE_ACCOUNT_TOKEN` and
  `CEREBRIUM_PROJECT_ID` are already set as env vars; the CLI and the curl
  examples use them automatically.
- **Local machine:** `cerebrium login` (interactive), or pass
  `--service-account-token <token>` / set the env var.

### 2. HuggingFace token (gated Mistral repo)

`mistralai/Mistral-7B-Instruct-v0.2` is gated, so the container needs a token
with access. Store it as a project secret (Cerebrium injects secrets as env
vars; `main.py` reads `HF_TOKEN`):

```bash
cerebrium secrets add "HF_TOKEN=$HF_TOKEN"
cerebrium secrets list                # verify (HF_TOKEN listed)
```

### 3. Upload datasets to the volume

Datasets are **not** bundled in the deploy/upload (keeps the tar small and works
around the `cerebrium run` 4 MB cap). `main.py` reads them from
`/persistent-storage/cacheblend-data/`, so upload them once:

```bash
cd workspace/code
cerebrium cp data/wikimqa_s.json cacheblend-data/wikimqa_s.json   # smoke needs only this
cerebrium cp data/musique_s.json cacheblend-data/musique_s.json   # full grid
cerebrium cp data/samsum.json    cacheblend-data/samsum.json
cerebrium cp data/nq_dpr.json    cacheblend-data/nq_dpr.json
cerebrium ls cacheblend-data/
```

`cerebrium cp` has no 4 MB limit (that cap is only on the `cerebrium run` tar).

### 4. Deploy

```bash
cd workspace/code
cerebrium deploy -y
```

This builds the image (deps from `cerebrium.toml`), deploys the app, and prints
the endpoint. The GPU is **not** billed until you invoke (`min_replicas = 0`).

---

## Choosing the GPU

The GPU is set **at deploy time** in `cerebrium.toml`, not per invocation —
Cerebrium has **no per-call GPU override**. To change GPU you edit the toml and
**redeploy**:

```toml
[cerebrium.hardware]
cpu = 4
memory = 16.0
compute = "ADA_L4"     # <-- change this, then `cerebrium deploy -y`
gpu_count = 1
provider = "aws"
region = "us-east-1"
```

### Cost rule (from repo `CLAUDE.md`)

> Never default to A100/H100. Start at **L4** and only upgrade after a confirmed
> OOM. Each tier up is ~2–3× the price.

| Workload | Cheapest acceptable `compute` |
|---|---|
| Mistral-7B fp16 (this repo) | `ADA_L4` (24 GB) |
| Mistral-7B very long ctx (>8k) | `AMPERE_A10` |
| ≥30B params | `AMPERE_A100_40GB` |
| ≥70B params | `AMPERE_A100_80GB` / `HOPPER_H100` |

Common `compute` values: `ADA_L4`, `ADA_L40`, `AMPERE_A10`, `TURING_T4`,
`AMPERE_A100_40GB`, `AMPERE_A100_80GB`, `HOPPER_H100`, `CPU`. Exact availability
depends on your plan — confirm against the current Cerebrium GPU docs
(<https://cerebrium.ai/docs/cerebrium/hardware/using-gpus>) before using a tier
you haven't run before.

---

## Running the eval

### Option A — wrapper script (recommended)

`scripts/run_cerebrium.sh` issues the async call and, when
`CEREBRIUM_WEBHOOK_URL` is set, appends a url-encoded `&webhookEndpoint` so the
run reports completion instead of sitting at `processing`.

```bash
cd workspace/code
export CEREBRIUM_WEBHOOK_URL="https://<your-receiver-url>"   # optional, set once

scripts/run_cerebrium.sh                          # smoke (async)
scripts/run_cerebrium.sh --mode full --n 200      # full grid
scripts/run_cerebrium.sh --mode full --deviation-mode v   # paper-comparison HKVD
SYNC=1 scripts/run_cerebrium.sh                   # wait for the JSON response inline
DRY_RUN=1 scripts/run_cerebrium.sh --mode full    # print the request, send nothing (no GPU)
```

### Option B — raw curl

```bash
# async: returns {"run_id": "..."} (202) immediately
curl -X POST "https://api.aws.us-east-1.cerebrium.ai/v4/p-238b3475/cacheblend-eval/run_eval_cerebrium?async=true" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CEREBRIUM_SERVICE_ACCOUNT_TOKEN" \
  --data '{"mode":"smoke"}'

# sync: drop ?async=true to wait and get the summary JSON back inline
```

### Parameters (JSON body / wrapper flags)

| Body key | Wrapper flag | Values | Meaning |
|---|---|---|---|
| `mode` | `--mode` | `smoke` \| `full` | smoke = wikimqa, n=3, ratio 0.15 |
| `n` | `--n` | int | examples per dataset (0 = use YAML) |
| `deviation_mode` | `--deviation-mode` | `v` \| `k` \| `kv` | HKVD selector (`v` = paper) |
| `config` | `--config` | path | config relative to `workspace/code/` |

---

## Getting results

`main.py` writes the full per-run JSON to the volume at
`/persistent-storage/cacheblend-results/<timestamp>.json` (a new file per call —
nothing is overwritten). The sync response / wrapper return also contains a
compact summary.

```bash
cerebrium ls cacheblend-results/                          # list runs
cerebrium download cacheblend-results/<id>.json /tmp/r.json
cat /tmp/r.json
```

Reference smoke output (wikimqa, n=3 — noisy, just a pipeline sanity check):

| strategy | ratio | F1 |
|---|---|---|
| full_recompute | – | 0.111 |
| full_reuse | – | 0.058 |
| cacheblend | 0.15 (dev=k) | 0.078 |

### Watching progress

```bash
cerebrium runs list cacheblend-eval     # async run status
cerebrium logs cacheblend-eval          # live logs (model load, per-batch F1)
```

Note: async runs without a `webhookEndpoint` can stay labelled `processing` in
the dashboard even after they finish — the completion callback is what flips the
label. The reliable completion signal is the results JSON appearing on the
volume.

---

## Cost & lifecycle

GPU billing is per-second of **running container** time, governed by the toml:

- `min_replicas = 0` → no idle GPU after a run (never set >0 for benchmarks).
- `cooldown = 30` → container scales down 30 s after the call returns.
- `response_grace_period = 10800` (3 h) → max single-run wall-clock (full grid
  is ~2.5 h).

Check whether anything is billing right now (empty = no GPU cost):

```bash
cerebrium containers list cacheblend-eval     # no containers => $0 GPU
```

What still costs money after a run: only the **volume storage** (~14 GB model
cache + datasets + results) — small, and far cheaper than re-downloading the
model. Clear the model cache (forces a re-download next run) with:

```bash
cerebrium rm hf-cache/ -r       # usually NOT worth it
```

Approx L4 cost: cold start (cached weights) ~30 s; smoke run ~$0.03; full 6×
grid ~$2. See `CLAUDE.md` for the full table.

---

## Persistent volume layout

Everything under `/persistent-storage` survives container restarts and new
sessions (it's tied to the **project/region**, not your shell session):

```
/persistent-storage/
├── hf-cache/            # HF_HOME — Mistral weights cached here (downloaded once)
├── cacheblend-data/     # datasets uploaded via `cerebrium cp`
└── cacheblend-results/  # per-run result JSONs
```

A new terminal/session re-downloads **nothing** as long as you call the deployed
app in the same project/region — the volume is re-mounted and the cache hits.

---

## Why these files can't move yet

Cerebrium's **default runtime** discovers the callable functions from a
`main.py` at the **deploy root**, and `cerebrium deploy` packages that directory.
`main.py` imports the shared `eval/` and `cacheblend/` packages (also used by the
Modal path, `run_eval_modal.py`) and reads `configs/` + `data/`. Moving
`main.py`/`cerebrium.toml` into this folder would break entrypoint discovery and
shared-code bundling unless we also relocate the shared core (and fix the Modal
imports) or switch to a custom runtime. That reorg is deferred; for now only this
guide and the invoke wrapper are grouped under the Cerebrium umbrella.
