# Claude operating notes for this repo

## Modal GPU cost rules (read before touching `run_eval_modal.py`)

Modal bills per second of GPU wall-clock per container. The full CacheBlend
grid runs ≥ 1 GPU-hour, so trivial mistakes (wrong GPU class, re-download
on every run, accidental warm pool) waste real money. Follow these rules
for any Modal change in this repo.

### Choosing GPU class

| Workload | Cheapest acceptable GPU | Reason |
|---|---|---|
| Mistral-7B fp16 inference (this repo) | **L4** (24 GB, ~$0.80/h) | T4 (16 GB) OOMs on long prompts + KV cache; L4 has 24 GB and is the lowest tier with bf16/fp16 tensor cores |
| Llama-13B / Mistral-7B with very long ctx (>8k) | A10G | More headroom only if L4 OOMs |
| ≥ 30B parameters | A100-40GB | Anything smaller will spill / require quantization |
| ≥ 70B parameters | A100-80GB / H100 | Only if explicitly required |

Never default to A100 / H100. Each tier is 2-3× the price of the previous
one. Always start at L4 and only upgrade after a confirmed OOM.

### Cache aggressively

1. **HF weights → Modal Volume.** Every `@app.function` that touches a HF
   model MUST mount the `cacheblend-hf-cache` Volume at `$HF_HOME` (i.e.
   `/root/.cache/huggingface`). A 14 GB Mistral re-download on every cold
   start costs ~1 minute of GPU time = $0.013 per launch and gets worse on
   bigger models.
2. **Tokenizer / dataset preprocessing → CPU container.** If a future task
   adds an expensive CPU-side pre-step (chunking, embedding, building
   indexes), put it on a separate `@app.function()` with `gpu=None` and
   write the artifact to a Volume. Don't run CPU work on a GPU container.
3. **Results → Volume, not return value.** Large outputs (KV traces, full
   prediction dumps) go to the results Volume; the function should return
   only summary stats so wire transfer is cheap.

### Container lifecycle

1. **One container per grid, not per cell.** A `(dataset × strategy × ratio)`
   sweep should run inside a single `@app.function` invocation that loops,
   not via `.map()` over many small calls. Loading Mistral-7B into VRAM
   takes ~30-60 s; doing that 24 times for a 6×4 grid wastes ~15 min of
   GPU billing.
2. **`scaledown_window=60`** (or shorter) on the eval function. Modal's
   default keeps the container warm for a few minutes after the call
   returns, which is helpful for interactive dev but pure waste for a
   one-shot eval. Set explicitly.
3. **Never set `min_containers > 0`** for eval functions. That keeps a GPU
   warm 24/7 — only acceptable for production serving, never for benchmarks.
4. **Set a hard `timeout`.** Eval bugs (infinite loop in generate) will
   otherwise burn the full default 24 h timeout. Cap at the realistic
   wall-clock you expect × 2.
5. **`enable_memory_snapshot=True`** for any function that imports torch /
   transformers. The snapshot replays after-import CPU memory so cold
   starts skip ~5-10 s of Python imports.

### Smoke before sweep

Before any change to the full grid (new dataset, new strategy, new model):

1. Run `modal run run_eval_modal.py --mode smoke` (n=3, single ratio, one
   dataset). Costs ≤ $0.05.
2. Inspect the JSON in `workspace/results/` — sanity-check that F1 is
   non-zero and matches the expected ballpark.
3. Only then run `--mode full`. If the smoke run blows up after model
   load (~$0.02), that's cheap; if a 6-hour full run blows up at hour 5,
   that's $5+ of wasted budget.

### Token / secret handling

- HuggingFace credentials live in the Modal secret named `huggingface`
  (keys: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`). Reference via
  `modal.Secret.from_name("huggingface")`; never inline the token in code
  or YAML.
- If a new token is provided, rotate via
  `modal secret create huggingface HF_TOKEN=... HUGGING_FACE_HUB_TOKEN=... --force`.

### Things to NEVER do

- `modal run --detach` for an untested change. If the bug only manifests
  after model load, the detached run keeps billing until you remember to
  kill it.
- Allocate a GPU function and then `time.sleep()` waiting for an external
  event — the container holds the GPU the entire time.
- Load the model inside a loop or per-example function.
- Skip the volume mount because "it's just one run" — the next person to
  run the eval will pay for the re-download.

### Cost ballpark (L4, May 2026)

| Action | Approx GPU time | Approx cost |
|---|---|---|
| First cold start (weights download + load) | 90 s | $0.02 |
| Warm cold start (weights cached) | 30 s | $0.007 |
| One example (Mistral-7B, prompt ~2k, max_new=32) | 1-2 s | $0.0003 |
| Full grid: 6 datasets × 200 ex × (full+reuse+4 ratios) | ~2.5 h | ~$2.00 |
| Smoke (wikimqa, n=3, r=0.15 only) | ~2 min | ~$0.03 |

Update this table if Modal pricing or the workload changes.

## Cerebrium GPU cost rules (read before touching `main.py` / `cerebrium.toml`)

Cerebrium bills per second of GPU wall-clock while a replica is up. The same
discipline as the Modal rules above applies — the knobs just live in
`cerebrium.toml` and `main.py` instead of `@app.function(...)` decorators.
Every Modal rule maps to a Cerebrium equivalent:

### Modal rule → Cerebrium knob

| Modal rule | Cerebrium equivalent |
|---|---|
| `gpu="L4"` per call | `[cerebrium.hardware] compute = "ADA_L4"` — fixed at **deploy** time; there is **no per-call GPU override**. To change GPU, edit the toml and `cerebrium deploy` again. |
| HF weights → `modal.Volume` at `$HF_HOME` | 50 GB persistent volume at `/persistent-storage`; `_prepare_container` sets `HF_HOME`/`TRANSFORMERS_CACHE` there so the 14 GB Mistral download is cached across runs. |
| Results → Volume, return summary only | written to `/persistent-storage/cacheblend-results/<run_id>*.json`; the entrypoint returns only a compact summary so the HTTP payload stays small. |
| One container per grid (loop, not `.map`) | `eval.run_eval.run_combined` loads the model once and loops the whole `(dataset × strategy × ratio)` grid in one call. Never one invocation per cell. |
| `scaledown_window=60` | `[cerebrium.scaling] cooldown = 30` — scale the replica down 30 s after the run ends. |
| Never `min_containers > 0` | `[cerebrium.scaling] min_replicas = 0` (never keep a GPU warm 24/7); `max_replicas = 1` + `replica_concurrency = 1` prevent fan-out billing. |
| Hard `timeout` | `[cerebrium.scaling] response_grace_period` — default 3600 s; bumped to 10800 for the full grid (hard cap 12 h). |
| `enable_memory_snapshot=True` | no direct analog; the volume-cached weights cover the bulk of cold-start cost. |
| secret `huggingface` | `cerebrium secrets set HF_TOKEN <hf_xxx>`; the managed web env injects `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`. Never inline the token in code/toml. |

### Choosing GPU class (Cerebrium `compute` names)

Same ladder as Modal — start at the cheapest that fits, upgrade only after a
confirmed OOM. Names that are authoritative in this repo's toml:

| Workload | `compute` | Reason |
|---|---|---|
| Mistral-7B fp16 (this repo) | **`ADA_L4`** (L4, 24 GB) | a 16 GB tier OOMs on long prompts + KV cache; L4 is the cheapest 24 GB option |
| Mistral-7B long ctx (>8k) / ~13B | `AMPERE_A10` (24 GB) | more headroom only if L4 OOMs |
| ≥ 30B params | `AMPERE_A100_40GB` | smaller will spill / need quantization |
| ≥ 70B params | `AMPERE_A100_80GB` / `HOPPER_H100` | only if explicitly required |

Never default to A100 / H100.

### Run it the cheap way: deploy + async, not `cerebrium run`

- The ephemeral `cerebrium run` path cannot carry this workload: 4 MB upload cap
  (datasets total ~26 MB), no persistent volume (the 14 GB model re-downloads),
  and a ~5–7 min poll timeout shorter than cold-start + eval. **Deploy once and
  invoke the endpoint async**, then pull the result JSON off the volume.
- Smoke before sweep: `scripts/run_cerebrium.sh --mode smoke` (wikimqa, n=3, one
  ratio, ≤ ~$0.05). Inspect the `_combined.json` on the volume (F1 non-zero,
  TTFT sane) before `--mode full`. Optionally `--check-correctness` first.

### Cerebrium-specific things to NEVER do

- **Don't retry a `SYNC=1` call in a loop.** A SYNC request that fails on the
  *response* (e.g. transient TLS clock-skew) has usually already RUN on the GPU;
  retrying re-runs the whole sweep and bills twice. Prefer async + pull from the
  volume, or check whether the prior run's JSON already landed before retrying.
- Don't deploy with `min_replicas > 0` (idle GPU billed 24/7).
- Don't block a GPU replica on an external event (`time.sleep`, polling) — the
  replica holds the GPU the entire time.
- Don't load the model per example, and don't skip the `/persistent-storage` HF
  cache "because it's just one run" — the next run pays for the re-download.

### Cost ballpark (Cerebrium L4, May 2026; verify current pricing)

| Action | Approx GPU time | Approx cost |
|---|---|---|
| Cold start (weights cached on volume) + model load | 30–45 s | ~$0.01 |
| First-ever cold start (14 GB download) | ~90 s | ~$0.02 |
| Smoke (wikimqa, n=3, 1 ratio, accuracy + TTFT) | ~3.5 min | ~$0.05 |
| One TTFT example (full_recompute + reuse + N ratios, repeats=3) | ~30–45 s | ~$0.01 |
| Full grid (~6 datasets × 200 ex, accuracy + TTFT) | ~2.5–3 h | ~$2–2.5 |

Update this table if Cerebrium pricing or the workload changes.
