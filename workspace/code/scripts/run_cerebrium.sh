#!/usr/bin/env bash
#
# Async-invoke the deployed CacheBlend eval on Cerebrium.
#
# The webhook completion callback is enabled BY DEFAULT whenever the
# CEREBRIUM_WEBHOOK_URL env var is set: the script appends
# "&webhookEndpoint=<url-encoded>" so Cerebrium POSTs the function's response
# there on completion (otherwise the async run sits at "processing" in the
# dashboard until you pull the result JSON off the volume).
#
# Usage:
#   ./run_cerebrium.sh                          # mode=smoke (async)
#   ./run_cerebrium.sh --mode full --n 200
#   ./run_cerebrium.sh --mode full --deviation-mode v
#   SYNC=1 ./run_cerebrium.sh                    # wait for the JSON response (no async/webhook)
#   DRY_RUN=1 ./run_cerebrium.sh --mode full --n 50   # print the request, don't send it
#
# Required env:
#   CEREBRIUM_SERVICE_ACCOUNT_TOKEN   bearer token (already set in the managed env)
#   CEREBRIUM_PROJECT_ID              e.g. p-238b3475 (already set in the managed env)
# Optional env:
#   CEREBRIUM_WEBHOOK_URL             completion callback target; set once to make webhooks the default
#   CEREBRIUM_APP                     default: cacheblend-eval
#   CEREBRIUM_REGION                  default: aws.us-east-1
#   CEREBRIUM_FUNCTION                default: run_eval_cerebrium
#
# After an async run, fetch results from the volume:
#   cerebrium ls cacheblend-results/
#   cerebrium download cacheblend-results/<id>.json
set -euo pipefail

MODE=smoke
N=""
DEV=""
CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)            MODE="$2";   shift 2;;
    --n)               N="$2";      shift 2;;
    --deviation-mode)  DEV="$2";    shift 2;;
    --config)          CONFIG="$2"; shift 2;;
    -h|--help)         sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

: "${CEREBRIUM_SERVICE_ACCOUNT_TOKEN:?set CEREBRIUM_SERVICE_ACCOUNT_TOKEN}"
: "${CEREBRIUM_PROJECT_ID:?set CEREBRIUM_PROJECT_ID}"
APP="${CEREBRIUM_APP:-cacheblend-eval}"
REGION="${CEREBRIUM_REGION:-aws.us-east-1}"
FUNC="${CEREBRIUM_FUNCTION:-run_eval_cerebrium}"

# Build the JSON body from the flags that were actually provided.
body="{\"mode\":\"${MODE}\""
[[ -n "$N" ]]      && body="${body},\"n\":${N}"
[[ -n "$DEV" ]]    && body="${body},\"deviation_mode\":\"${DEV}\""
[[ -n "$CONFIG" ]] && body="${body},\"config\":\"${CONFIG}\""
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
