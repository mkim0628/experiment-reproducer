#!/usr/bin/env bash
#
# Async-invoke the deployed CacheBlend *TTFT* measurement on Cerebrium.
#
# Latency analogue of run_cerebrium.sh: it targets the run_ttft_cerebrium
# function instead of run_eval_cerebrium and adds --repeats / --warmup. As with
# the eval wrapper, the webhook completion callback is enabled BY DEFAULT
# whenever CEREBRIUM_WEBHOOK_URL is set.
#
# NOTE: this reproduction's cacheblend path is a two-pass implementation, so its
# measured TTFT is an upper bound, NOT the paper's single-pass
# selective-recompute latency. The full_recompute vs full_reuse comparison is
# faithful. See eval/run_ttft.py.
#
# Usage:
#   ./run_ttft_cerebrium.sh                              # mode=smoke (async)
#   ./run_ttft_cerebrium.sh --mode full --n 50
#   ./run_ttft_cerebrium.sh --mode smoke --repeats 5 --warmup 2
#   SYNC=1 ./run_ttft_cerebrium.sh                       # wait for the JSON response inline
#   DRY_RUN=1 ./run_ttft_cerebrium.sh --mode full        # print the request, don't send it
#
# Required env:
#   CEREBRIUM_SERVICE_ACCOUNT_TOKEN   bearer token (already set in the managed env)
#   CEREBRIUM_PROJECT_ID              e.g. p-238b3475 (already set in the managed env)
# Optional env:
#   CEREBRIUM_WEBHOOK_URL             completion callback target
#   CEREBRIUM_APP                     default: cacheblend-eval
#   CEREBRIUM_REGION                  default: aws.us-east-1
#   CEREBRIUM_FUNCTION                default: run_ttft_cerebrium
#
# After an async run, fetch the TTFT JSON (file name ends in _ttft.json):
#   cerebrium ls cacheblend-results/
#   cerebrium download cacheblend-results/<id>_ttft.json
set -euo pipefail

MODE=smoke
N=""
DEV=""
CONFIG=""
REPEATS=""
WARMUP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)            MODE="$2";    shift 2;;
    --n)               N="$2";       shift 2;;
    --deviation-mode)  DEV="$2";     shift 2;;
    --config)          CONFIG="$2";  shift 2;;
    --repeats)         REPEATS="$2"; shift 2;;
    --warmup)          WARMUP="$2";  shift 2;;
    -h|--help)         sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

: "${CEREBRIUM_SERVICE_ACCOUNT_TOKEN:?set CEREBRIUM_SERVICE_ACCOUNT_TOKEN}"
: "${CEREBRIUM_PROJECT_ID:?set CEREBRIUM_PROJECT_ID}"
APP="${CEREBRIUM_APP:-cacheblend-eval}"
REGION="${CEREBRIUM_REGION:-aws.us-east-1}"
FUNC="${CEREBRIUM_FUNCTION:-run_ttft_cerebrium}"

# Build the JSON body from the flags that were actually provided.
body="{\"mode\":\"${MODE}\""
[[ -n "$N" ]]       && body="${body},\"n\":${N}"
[[ -n "$DEV" ]]     && body="${body},\"deviation_mode\":\"${DEV}\""
[[ -n "$CONFIG" ]]  && body="${body},\"config\":\"${CONFIG}\""
[[ -n "$REPEATS" ]] && body="${body},\"repeats\":${REPEATS}"
[[ -n "$WARMUP" ]]  && body="${body},\"warmup\":${WARMUP}"
body="${body}}"

base="https://api.${REGION}.cerebrium.ai/v4/${CEREBRIUM_PROJECT_ID}/${APP}/${FUNC}"

if [[ "${SYNC:-}" == "1" ]]; then
  url="$base"
  echo "[run_ttft_cerebrium] SYNC mode: waiting for the response (no webhook)" >&2
else
  url="${base}?async=true"
  if [[ -n "${CEREBRIUM_WEBHOOK_URL:-}" ]]; then
    enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$CEREBRIUM_WEBHOOK_URL")
    url="${url}&webhookEndpoint=${enc}"
    echo "[run_ttft_cerebrium] async + webhook -> ${CEREBRIUM_WEBHOOK_URL}" >&2
  else
    echo "[run_ttft_cerebrium] async (no CEREBRIUM_WEBHOOK_URL set -> run stays 'processing' until you pull the result JSON)" >&2
  fi
fi

echo "[run_ttft_cerebrium] POST ${url}" >&2
echo "[run_ttft_cerebrium] body ${body}" >&2

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "[run_ttft_cerebrium] DRY_RUN=1 -> not sending" >&2
  exit 0
fi

curl -sS -w '\n' -X POST "${url}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${CEREBRIUM_SERVICE_ACCOUNT_TOKEN}" \
  --data "${body}"
