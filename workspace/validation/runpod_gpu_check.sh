#!/usr/bin/env bash
set -euo pipefail

# RunPod GPU 환경 검증 스크립트
# 로컬 GPU 유무, RunPod REST API 인증, 네트워크 정책, 현재 자원 상태를 점검한다.
# 실제 GPU pod 생성/과금은 하지 않는다 (smoke test 옵션 별도).

API="https://rest.runpod.io/v1"
AUTH=(-H "Authorization: Bearer ${RUNPOD_API_KEY:-}")

pass() { printf "  [OK]   %s\n" "$1"; }
fail() { printf "  [FAIL] %s\n" "$1"; }
warn() { printf "  [WARN] %s\n" "$1"; }

echo "=== 1. 로컬 GPU/CUDA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  pass "nvidia-smi 존재"
  nvidia-smi -L | sed 's/^/         /'
else
  warn "nvidia-smi 없음 — 로컬 GPU 사용 불가 (예상된 상태: 원격 RunPod GPU만 사용)"
fi
python3 -c "import torch" 2>/dev/null && pass "torch import 가능" || warn "torch 미설치"

echo
echo "=== 2. 환경 변수 ==="
if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  pass "RUNPOD_API_KEY 설정됨 (len=${#RUNPOD_API_KEY})"
else
  fail "RUNPOD_API_KEY 미설정"; exit 1
fi

echo
echo "=== 3. 네트워크 정책 ==="
for host in rest.runpod.io api.runpod.io docs.runpod.io; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://$host" || true)
  if [[ "$code" == "403" ]]; then
    deny=$(curl -sI "https://$host" | grep -i "x-deny-reason" || true)
    if [[ -n "$deny" ]]; then warn "$host: 차단 ($deny)"; else pass "$host: $code"; fi
  else
    pass "$host: HTTP $code"
  fi
done

echo
echo "=== 4. RunPod REST API 인증 ==="
resp=$(curl -sS -w "\n%{http_code}" "$API/pods" "${AUTH[@]}")
code=$(echo "$resp" | tail -n1); body=$(echo "$resp" | sed '$d')
if [[ "$code" == "200" ]]; then
  pass "GET /pods → 200"
  pod_count=$(echo "$body" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
  echo "         현재 pod: $pod_count 개"
else
  fail "GET /pods → $code"; echo "$body"; exit 1
fi

echo
echo "=== 5. 자원 현황 ==="
for path in /pods /endpoints /networkvolumes /templates; do
  n=$(curl -sS "$API$path" "${AUTH[@]}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 'n/a')")
  echo "         $path: $n"
done

echo
echo "=== 6. 결론 ==="
echo "         - 로컬: GPU 없음 (RunPod 원격 실행 전제)"
echo "         - REST API: 접근 OK, 인증 OK"
echo "         - 실제 GPU 동작 확인은 'smoke test' (pod 생성 → nvidia-smi → 즉시 종료) 필요"
echo "           ※ 과금 발생 가능. 별도 승인 후 runpod_smoke_test.sh 실행."
