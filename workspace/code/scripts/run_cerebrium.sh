#!/usr/bin/env bash
#
# Async-invoke the deployed CacheBlend evaluation (accuracy + TTFT) on Cerebrium.
#
# Targets the single run_cerebrium function, which loads the model once and
# measures TTFT FIRST (on a clean device) then accuracy -- so the accuracy
# phase's full-length generation cannot perturb the TTFT numbers. The webhook
# completion callback is enabled BY DEFAULT whenever CEREBRIUM_WEBHOOK_URL is set.
#
# cacheblend is the single-pass selective recompute -- the SAME function is timed
# (TTFT) and scored (accuracy), so each row's acc/ttft come from one inference
# path (the paper's accuracy-vs-TTFT trade-off, one implementation). Pass
# --check-correctness to first verify r=1 reproduces a full forward.
# See eval/run_eval.py.
#
# Usage:
#   ./run_cerebrium.sh                              # mode=smoke (async)
#   ./run_cerebrium.sh --mode full --n 50
#   ./run_cerebrium.sh --mode smoke --repeats 5 --warmup 2
#   ./run_cerebrium.sh --mode smoke --check-correctness
#   ./run_cerebrium.sh --mode full --n 10 --ratios 0.1,0.15,0.2,0.4,0.6,0.8 --ttft-only
#   SYNC=1 ./run_cerebrium.sh                       # wait for the JSON response inline
#   DRY_RUN=1 ./run_cerebrium.sh --mode full        # print the request, don't send it
#
# Required env:
#   CEREBRIUM_SERVICE_ACCOUNT_TOKEN   bearer token (already set in the managed env)
#   CEREBRIUM_PROJECT_ID              e.g. p-238b3475 (already set in the managed env)
# Optional env:
#   CEREBRIUM_WEBHOOK_URL             completion callback target
#   CEREBRIUM_APP                     default: cacheblend-eval
#   CEREBRIUM_REGION                  default: aws.us-east-1
#   CEREBRIUM_FUNCTION                default: run_cerebrium
#
# After an async run, fetch the JSON (file name ends in _combined.json):
#   cerebrium ls cacheblend-results/
#   cerebrium download cacheblend-results/<id>_combined.json
set -euo pipefail

MODE=smoke
N=""
DEV=""
CONFIG=""
REPEATS=""
WARMUP=""
CHECK=""
RATIOS=""
TTFT_ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)              MODE="$2";    shift 2;;
    --n)                 N="$2";       shift 2;;
    --deviation-mode)    DEV="$2";     shift 2;;
    --config)            CONFIG="$2";  shift 2;;
    --repeats)           REPEATS="$2"; shift 2;;
    --warmup)            WARMUP="$2";  shift 2;;
    --check-correctness) CHECK="1";    shift 1;;
    --ratios)            RATIOS="$2";  shift 2;;
    --ttft-only)         TTFT_ONLY="1"; shift 1;;
    -h|--help)           sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

: "${CEREBRIUM_SERVICE_ACCOUNT_TOKEN:?set CEREBRIUM_SERVICE_ACCOUNT_TOKEN}"
: "${CEREBRIUM_PROJECT_ID:?set CEREBRIUM_PROJECT_ID}"
APP="${CEREBRIUM_APP:-cacheblend-eval}"
REGION="${CEREBRIUM_REGION:-aws.us-east-1}"
FUNC="${CEREBRIUM_FUNCTION:-run_cerebrium}"

# Build the JSON body from the flags that were actually provided.
body="{\"mode\":\"${MODE}\""
[[ -n "$N" ]]       && body="${body},\"n\":${N}"
[[ -n "$DEV" ]]     && body="${body},\"deviation_mode\":\"${DEV}\""
[[ -n "$CONFIG" ]]  && body="${body},\"config\":\"${CONFIG}\""
[[ -n "$REPEATS" ]]   && body="${body},\"repeats\":${REPEATS}"
[[ -n "$WARMUP" ]]    && body="${body},\"warmup\":${WARMUP}"
[[ -n "$CHECK" ]]     && body="${body},\"check_correctness\":true"
[[ -n "$RATIOS" ]]    && body="${body},\"ratios\":\"${RATIOS}\""
[[ -n "$TTFT_ONLY" ]] && body="${body},\"ttft_only\":true"
body="${body}}"

base="https://api.${REGION}.cerebrium.ai/v4/${CEREBRIUM_PROJECT_ID}/${APP}/${FUNC}"

if [[ "${SYNC:-}" == "1" ]]; then
  url="$base"
  echo "[run_cerebrium] SYNC mode: waiting for the response (no webhook)" >&2
else
  url="${base}?async=true"
  if [[ -n "${CEREBRIUM_WEBHOOK_URL:-}" ]]; then
    enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$CEREBRIUM_WEBHOOK_URL")
    url="${url}&webhookEndpoint=${enc}"
    echo "[run_cerebrium] async + webhook -> ${CEREBRIUM_WEBHOOK_URL}" >&2
  else
    echo "[run_cerebrium] async (no CEREBRIUM_WEBHOOK_URL set -> run stays 'processing' until you pull the result JSON)" >&2
  fi
fi

echo "[run_cerebrium] POST ${url}" >&2
echo "[run_cerebrium] body ${body}" >&2

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "[run_cerebrium] DRY_RUN=1 -> not sending" >&2
  exit 0
fi

curl -sS -w '\n' -X POST "${url}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${CEREBRIUM_SERVICE_ACCOUNT_TOKEN}" \
  --data "${body}"
