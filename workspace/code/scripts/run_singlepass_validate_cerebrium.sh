#!/usr/bin/env bash
#
# Invoke the single-pass selective-recompute VALIDATION on Cerebrium.
#
# Targets run_singlepass_validate_cerebrium, which (1) asserts single-pass at
# r=1.0 reproduces full_recompute token-for-token, (2) cross-checks single-pass
# vs two-pass F1, and (3) reports single-pass TTFT + speedup over full_recompute
# (the paper-style selective-recompute latency). See eval/single_pass.py +
# eval/validate_singlepass.py.
#
# Usage:
#   SYNC=1 ./run_singlepass_validate_cerebrium.sh                  # smoke, wait inline
#   ./run_singlepass_validate_cerebrium.sh --mode full --n 20
#   ./run_singlepass_validate_cerebrium.sh --max-new-tokens 16 --repeats 5
#   DRY_RUN=1 ./run_singlepass_validate_cerebrium.sh
#
# Required env: CEREBRIUM_SERVICE_ACCOUNT_TOKEN, CEREBRIUM_PROJECT_ID (set in env).
# Optional env: CEREBRIUM_WEBHOOK_URL, CEREBRIUM_APP, CEREBRIUM_REGION, CEREBRIUM_FUNCTION.
#
# After an async run, fetch the JSON (file name ends in _singlepass_validate.json):
#   cerebrium ls cacheblend-results/
#   cerebrium download cacheblend-results/<id>_singlepass_validate.json
set -euo pipefail

MODE=smoke
N=""
DEV=""
CONFIG=""
REPEATS=""
WARMUP=""
MAXNEW=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)             MODE="$2";    shift 2;;
    --n)                N="$2";       shift 2;;
    --deviation-mode)   DEV="$2";     shift 2;;
    --config)           CONFIG="$2";  shift 2;;
    --repeats)          REPEATS="$2"; shift 2;;
    --warmup)           WARMUP="$2";  shift 2;;
    --max-new-tokens)   MAXNEW="$2";  shift 2;;
    -h|--help)          sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

: "${CEREBRIUM_SERVICE_ACCOUNT_TOKEN:?set CEREBRIUM_SERVICE_ACCOUNT_TOKEN}"
: "${CEREBRIUM_PROJECT_ID:?set CEREBRIUM_PROJECT_ID}"
APP="${CEREBRIUM_APP:-cacheblend-eval}"
REGION="${CEREBRIUM_REGION:-aws.us-east-1}"
FUNC="${CEREBRIUM_FUNCTION:-run_singlepass_validate_cerebrium}"

body="{\"mode\":\"${MODE}\""
[[ -n "$N" ]]       && body="${body},\"n\":${N}"
[[ -n "$DEV" ]]     && body="${body},\"deviation_mode\":\"${DEV}\""
[[ -n "$CONFIG" ]]  && body="${body},\"config\":\"${CONFIG}\""
[[ -n "$REPEATS" ]] && body="${body},\"repeats\":${REPEATS}"
[[ -n "$WARMUP" ]]  && body="${body},\"warmup\":${WARMUP}"
[[ -n "$MAXNEW" ]]  && body="${body},\"max_new_tokens\":${MAXNEW}"
body="${body}}"

base="https://api.${REGION}.cerebrium.ai/v4/${CEREBRIUM_PROJECT_ID}/${APP}/${FUNC}"

if [[ "${SYNC:-}" == "1" ]]; then
  url="$base"
  echo "[singlepass_validate] SYNC mode: waiting for the response (no webhook)" >&2
else
  url="${base}?async=true"
  if [[ -n "${CEREBRIUM_WEBHOOK_URL:-}" ]]; then
    enc=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$CEREBRIUM_WEBHOOK_URL")
    url="${url}&webhookEndpoint=${enc}"
    echo "[singlepass_validate] async + webhook -> ${CEREBRIUM_WEBHOOK_URL}" >&2
  else
    echo "[singlepass_validate] async (no CEREBRIUM_WEBHOOK_URL -> pull the result JSON when done)" >&2
  fi
fi

echo "[singlepass_validate] POST ${url}" >&2
echo "[singlepass_validate] body ${body}" >&2

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "[singlepass_validate] DRY_RUN=1 -> not sending" >&2
  exit 0
fi

curl -sS -w '\n' -X POST "${url}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${CEREBRIUM_SERVICE_ACCOUNT_TOKEN}" \
  --data "${body}"
