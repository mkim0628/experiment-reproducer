#!/usr/bin/env bash
# RunPod GPU smoke test
# 목적: 실제로 GPU pod이 RUNNING 상태까지 도달하는지 확인. SSH 없이 REST API만 사용.
# 안전장치: trap EXIT 으로 어떤 경로로 빠져나가든 pod 을 무조건 DELETE 한다.
#
# 사용법:
#   workspace/validation/runpod_smoke_test.sh                     # 기본 GPU 후보 순회
#   workspace/validation/runpod_smoke_test.sh "NVIDIA RTX A4000"  # 특정 GPU 지정
#
# 환경변수:
#   RUNPOD_API_KEY     (필수)
#   POLL_TIMEOUT_SEC   기본 300 (RUNNING 까지 기다리는 최대 시간)
#   CLOUD_TYPE         기본 COMMUNITY (저렴). SECURE 가능.

set -euo pipefail

API="https://rest.runpod.io/v1"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
AUTH=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")

POLL_TIMEOUT_SEC="${POLL_TIMEOUT_SEC:-300}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"

# 인자로 GPU 타입을 받으면 그것만, 아니면 저렴한 순서대로 fallback
if [[ $# -ge 1 ]]; then
  GPU_CANDIDATES=("$1")
else
  GPU_CANDIDATES=(
    "NVIDIA RTX A4000"
    "NVIDIA GeForce RTX 3090"
    "NVIDIA RTX A5000"
    "NVIDIA GeForce RTX 4090"
  )
fi

POD_ID=""
START_EPOCH=$(date +%s)

cleanup() {
  local rc=$?
  if [[ -n "$POD_ID" ]]; then
    echo
    echo "[cleanup] DELETE pod $POD_ID ..."
    # best-effort; never fail the script on cleanup
    local code
    code=$(curl -sS -o /tmp/runpod_del.json -w "%{http_code}" -X DELETE "$API/pods/$POD_ID" "${AUTH[@]}" || echo "ERR")
    echo "[cleanup] DELETE response: HTTP $code"
    [[ -s /tmp/runpod_del.json ]] && cat /tmp/runpod_del.json && echo

    # verify gone
    local left
    left=$(curl -sS "$API/pods" "${AUTH[@]}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for p in d if p.get('id')=='$POD_ID'))" || echo "?")
    echo "[cleanup] pod still listed: $left (expect 0)"
  fi
  local end=$(date +%s)
  echo "[cleanup] total elapsed: $((end-START_EPOCH))s, exit=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

create_pod() {
  local gpu="$1"
  echo ">>> trying GPU: $gpu  (cloud=$CLOUD_TYPE)"
  local payload
  payload=$(python3 -c "
import json, sys
print(json.dumps({
  'name': 'smoke-test-' + str(__import__('time').time()).split('.')[0],
  'imageName': 'nvidia/cuda:12.2.0-base-ubuntu22.04',
  'cloudType': '$CLOUD_TYPE',
  'computeType': 'GPU',
  'gpuTypeIds': ['$gpu'],
  'gpuCount': 1,
  'containerDiskInGb': 10,
  'volumeInGb': 0,
  'minVCPUPerGPU': 2,
  'minRAMPerGPU': 8,
  'interruptible': False,
  'dockerStartCmd': ['sleep','120'],
  'gpuTypePriority': 'availability',
}))
")
  local resp code
  resp=$(curl -sS -w "\n%{http_code}" -X POST "$API/pods" "${AUTH[@]}" -d "$payload")
  code=$(echo "$resp" | tail -n1); body=$(echo "$resp" | sed '$d')
  echo "    POST /pods → HTTP $code"
  if [[ "$code" =~ ^2 ]]; then
    POD_ID=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
    if [[ -n "$POD_ID" ]]; then
      echo "    pod id: $POD_ID"
      return 0
    fi
  fi
  echo "    response body:"
  echo "$body" | sed 's/^/      /'
  return 1
}

poll_until_running() {
  local deadline=$(( $(date +%s) + POLL_TIMEOUT_SEC ))
  local last_status=""
  while (( $(date +%s) < deadline )); do
    local body
    body=$(curl -sS "$API/pods/$POD_ID" "${AUTH[@]}")
    local status started machine_id gpu_name location ports
    status=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('desiredStatus',''))")
    started=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('lastStartedAt') or '')")
    machine_id=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('machineId') or '')")
    gpu_name=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('machine') or {}; print(m.get('gpuDisplayName') or '')")
    location=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('machine') or {}; print(m.get('location') or '')")
    ports=$(echo "$body"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(bool(d.get('portMappings')))")

    if [[ "$status:$started:$machine_id" != "$last_status" ]]; then
      printf "    [%3ds] status=%s machine=%s gpu='%s' loc=%s started=%s\n" \
        "$(( $(date +%s) - START_EPOCH ))" "$status" "$machine_id" "$gpu_name" "$location" "$started"
      last_status="$status:$started:$machine_id"
    fi

    # RUNNING + 머신 할당 + 컨테이너 기동시간 표기 시점을 성공으로 간주
    if [[ "$status" == "RUNNING" && -n "$started" && -n "$machine_id" ]]; then
      echo
      echo "=== POD READY ==="
      echo "$body" | python3 -m json.tool
      return 0
    fi
    sleep 5
  done
  echo "!! timeout after ${POLL_TIMEOUT_SEC}s — pod did not reach RUNNING"
  curl -sS "$API/pods/$POD_ID" "${AUTH[@]}" | python3 -m json.tool || true
  return 1
}

# 메인 루프: 후보 GPU 순회
for gpu in "${GPU_CANDIDATES[@]}"; do
  if create_pod "$gpu"; then
    if poll_until_running; then
      echo
      echo "✅ smoke test PASS — GPU pod reached RUNNING"
      echo "   gpu=$gpu  pod=$POD_ID  elapsed=$(( $(date +%s) - START_EPOCH ))s"
      exit 0
    else
      echo "!! pod did not reach RUNNING for $gpu — cleaning up & trying next"
      # cleanup trap 가 현재 POD_ID 를 삭제. 다음 후보로 넘어가기 위해 비움.
      curl -sS -o /dev/null -X DELETE "$API/pods/$POD_ID" "${AUTH[@]}" || true
      POD_ID=""
    fi
  else
    echo "!! create failed for $gpu, trying next"
  fi
done

echo "❌ smoke test FAIL — no candidate GPU reached RUNNING"
exit 1
