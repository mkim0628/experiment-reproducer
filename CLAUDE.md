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
