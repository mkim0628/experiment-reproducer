# Artificial Analysis Coding Index 벤치마크 분석

> **기준일:** 2026-05-19. AA Intelligence Index **v4.0** 기준. v5 이후 컴포넌트가 바뀌면 본 문서도 갱신 필요.

## 출처

- AA Coding Index 페이지: https://artificialanalysis.ai/models/capabilities/coding
- AA Intelligence Benchmarking Methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- AA Coding Agent Index Methodology: https://artificialanalysis.ai/methodology/coding-agents-benchmarking
- AA Intelligence Index 리더보드: https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index
- AA Terminal-Bench Hard 리더보드: https://artificialanalysis.ai/evaluations/terminalbench-hard
- AA SciCode 리더보드: https://artificialanalysis.ai/evaluations/scicode
- AA LiveCodeBench 리더보드: https://artificialanalysis.ai/evaluations/livecodebench
- AA AA-LCR 리더보드: https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning
- AA Coding Agent Index: https://artificialanalysis.ai/agents/coding-agents
- Terminal-Bench 논문: https://arxiv.org/abs/2601.11868
- SciCode 논문: https://arxiv.org/abs/2407.13168
- LiveCodeBench 논문: https://arxiv.org/abs/2403.07974
- VentureBeat — AA Intelligence Index 개편(2026-04): https://venturebeat.com/technology/artificial-analysis-overhauls-its-ai-intelligence-index-replacing-popular

---

## TL;DR

1. AA에는 이름이 비슷한 **두 개의 코딩 지수**가 있다. 혼동에 주의.
   - **Coding Index** (모델 비교용, `/models/capabilities/coding`)
   - **Coding Agent Index** (코딩 에이전트 비교용, `/agents/coding-agents`)
2. 현재(v4.0, 2026-04~) **Coding Index = `Terminal-Bench Hard` + `SciCode`** 두 벤치마크의 단순 평균.
   - v3까지는 LiveCodeBench + SciCode 였음. v4에서 LiveCodeBench가 빠지고 Terminal-Bench Hard가 추가됨.
3. **Coding Agent Index**는 별개로 `SWE-Bench-Pro-Hard-AA` + `Terminal-Bench v2` + `SWE-Atlas-QnA` 의 평균.
4. 두 지수 모두 채점은 **pass@1, 동일 task 3회 반복 평균**, 결합은 **단순 산술 평균** (가중치/정규화 없음).
5. `AA-LCR`은 이름이 비슷하지만 코딩이 아니라 **Long Context Reasoning** 벤치마크다. Coding Index와 무관.

---

## 1. 모델 Coding Index 구성

> AA 공식 설명: *"Coding Index represents the average of coding evaluations in the AA Intelligence Index, currently Terminal-Bench Hard and SciCode."*

| 구성 벤치마크 | 종류 | 문제 수 (AA 사용분) | 채점 단위 | 무엇을 보는가 |
|---|---|---|---|---|
| **Terminal-Bench Hard** | 에이전틱 / 터미널 | 44 task (Terminal-Bench 2.0의 hard subset; 원본 89 중) | task pass/fail (모든 테스트 통과해야 1) | 컨테이너 환경에서 자율적으로 셸을 다루는 능력 — 시스템 관리, 데이터 파이프라인, 다단계 SWE 작업 |
| **SciCode** | 코드 합성 / 과학 도메인 | 65 main / 288 subproblem (test split) | subproblem pass/fail | 16개 자연과학 분야(수학·물리·화학·생물·재료) 연구 코드 — 도메인 지식 + 수치해석 + Python 구현 |

### 1.1 Terminal-Bench Hard

- **출처:** Stanford × Laude Institute의 Terminal-Bench (arXiv 2601.11868). AA는 *hard* 서브셋만 사용.
- **태스크 4요소:**
  1. 도커 컨테이너 환경 (관련 패키지·파일 미리 셋업)
  2. 자연어 지시문
  3. 정답 검증용 테스트 스위트
  4. 수기 레퍼런스 솔루션
- **AA 평가 방식:**
  - 모델 간 공정성을 위해 **Terminus 2 에이전트 하네스**로 통일.
  - 각 task pass@1, **3회 반복 평균**.
  - 모든 테스트 통과 → 1, 아니면 0 (부분점수 없음).
- **측정 능력:** 자율적 멀티스텝 실행, 에러 회복, 파일시스템 다루기, 셸 명령 연쇄.
- **약점/주의:**
  - 컨테이너 환경에 묶여 OS 레벨 작업에 편향.
  - 부분점수 없음 → 분산이 크다(3-run 평균을 쓰는 이유).

### 1.2 SciCode

- **출처:** NeurIPS 2024 D&B, 과학자들이 직접 큐레이션 (arXiv 2407.13168).
- **문제 구조:** main 문제를 여러 subproblem으로 분해 (중위수 3, 최대 15). subproblem마다 docstring·입출력·필요 시 과학적 배경 명시.
- **AA 평가 방식:**
  - 프롬프트 형식: **Scientist Annotated Background** — 모델이 (a) 다음 단계 풀이에 필요한 과학 지식을 먼저 생성하고, (b) 이어 Python 코드를 작성.
  - pass@1, 3회 반복.
  - 보고 지표는 통상 **subproblem 정확도** (288개 기준). main 문제 정확도는 훨씬 낮음.
- **측정 능력:** 단순 알고리즘 퍼즐이 아닌 실제 연구 코드 — 시뮬레이션·수치해석·과학 계산.
- **약점/주의:**
  - 매우 어려움. 공개 시점 best (Claude 3.5 Sonnet)도 main 4.6%. 2026년 프론티어 모델도 ~60% 부근(예: Gemini 3.1 Pro Preview ~58.9%).
  - 도메인 지식 leakage 가능 — 사전학습된 표준 알고리즘일수록 유리.

### 1.3 점수 결합 방식

- 컴포넌트별 점수는 0~100 백분율 (pass@1, 3-run mean).
- **Coding Index = (Terminal-Bench Hard + SciCode) / 2** — 단순 평균, 가중치 없음.
- AA 전체 Intelligence Index에는 95% CI ±1% 미만으로 명시되나, Coding Index는 컴포넌트가 2개라 분산이 더 클 수 있음에 유의.

---

## 2. v3 → v4 변화 (2026-04 개편)

| 항목 | v3 (≤2026-03) | v4.0 (2026-04~) |
|---|---|---|
| Coding 컴포넌트 | LiveCodeBench + SciCode | **Terminal-Bench Hard + SciCode** |
| Coding이 Intelligence Index에서 차지하는 비중 | ~25% (4 카테고리 × 25%) | 4 카테고리(Agents / Coding / Sci Reasoning / General) 동등 가중 |
| 전체 Index 10개 평가 | (이전 7개 평가) | GDPval-AA, τ²-Bench Telecom, Terminal-Bench Hard, SciCode, AA-LCR, AA-Omniscience, IFBench, Humanity's Last Exam, GPQA Diamond, CritPt |

**의도된 변화:** "경쟁 프로그래밍(알고리즘 퀴즈)" 무게를 빼고 "실세계 에이전트 + 도메인 코딩"으로 무게중심 이동. LiveCodeBench는 standalone leaderboard로 유지되지만 Coding Index에는 더 이상 포함되지 않는다.

`AA-LCR`은 이름 때문에 코딩으로 오해되기 쉬우나 **Long Context Reasoning**(100k 토큰 다문서 추론) 벤치마크다. Coding과 무관.

---

## 3. 별개 지수: Coding Agent Index (혼동 주의)

`/agents/coding-agents`에 있는 것은 **모델이 아닌 코딩 에이전트(예: Claude Code, Cursor, Codex 등)** 비교용 별도 지수다.

| 컴포넌트 | 측정 | 비고 |
|---|---|---|
| SWE-Bench-Pro-Hard-AA | 1,865 task, 평균 4.1 파일·107 라인 변경. 멀티파일·멀티언어 실제 SWE | 가장 무거운 컴포넌트 |
| Terminal-Bench v2 | 89 → 84 task (환경 호환성으로 5개 제외) | Terminal-Bench Hard와 다른 풀-셋 사용 |
| SWE-Atlas-QnA | Scale의 SWE-Atlas — 레포 이해 / Q&A 루브릭 | 변경이 아닌 "이해" 능력 |

- 채점: 각 컴포넌트 pass@1 → 단순 평균.
- 같은 풀로 비용(USD/task), 토큰 사용량, 실행시간도 함께 보고 — 성능·효율을 같은 데이터로 비교한다는 점이 특징.

→ "Claude Code vs Cursor 같은 에이전트 줄세우기"를 원한다면 이 쪽 지수를 봐야 한다.

---

## 4. 메타 분석 — 이 Coding Index를 얼마나 믿어야 하나

**장점**
- 두 컴포넌트가 보완적: SciCode(코드 합성·도메인 지식) ↔ Terminal-Bench Hard(에이전트 행위·실행).
- 모두 자동 채점(테스트 통과 여부) → 인적 평가 편향 없음.
- 3-run 평균과 통일된 에이전트 하네스(Terminus 2) 사용 → 재현성은 비교적 양호.
- LiveCodeBench 제거로 알고리즘 퀴즈 오버피팅 신호가 줄어듬.

**한계**
- **표본이 두 벤치마크뿐.** 컴포넌트 다양성이 낮음 — 둘 중 하나에서 강한 모델이 전체를 좌우.
- **부분점수 없음(binary pass@1):** 거의 풀었지만 한 줄 틀린 케이스가 0점 처리 → 분산↑, 작은 모델일수록 노이즈↑.
- **언어 편향:** SciCode는 Python 단일, Terminal-Bench Hard도 사실상 Bash+Python. 멀티언어(Rust/Go/JS) 능력은 사실상 안 들어감.
- **에이전트 능력과 모델 능력의 경계 모호:** Terminal-Bench Hard 점수는 (모델 + harness) 조합 점수. 순수 모델 능력 측정과는 약간 상충.
- **"실제 SWE 작업" 측면 약함:** PR-스타일 리포지토리 단위 수정은 모델 Coding Index에는 없음 (Coding Agent Index의 SWE-Bench-Pro 영역).
- **단순 평균:** 두 벤치마크의 절대 난이도/포화 정도가 다른데 동일 가중치 → 결과적으로 어려운 쪽(SciCode)이 신호를 더 많이 준다.

**실용적 시사점**
- 한 모델을 코딩용으로 고를 때 Coding Index만 보지 말고 컴포넌트 점수를 따로 봐야 한다 (특히 SciCode와 Terminal-Bench Hard가 측정하는 능력이 서로 다름).
- 알고리즘 코딩 위주 사용 사례라면 LiveCodeBench standalone 리더보드를 별도로 봐야 함.
- 에이전트 / IDE 통합 사용 사례라면 Coding Agent Index 쪽을 봐야 함.

---

## 5. 요약 표

| 항목 | 값 |
|---|---|
| 지수 이름 | Artificial Analysis Coding Index |
| 버전 기준 | Intelligence Index v4.0 (2026-04~) |
| 컴포넌트 수 | 2 |
| 컴포넌트 | Terminal-Bench Hard (44 task), SciCode (65 main / 288 subproblem) |
| 채점 | pass@1, 3 repeats 평균 (binary scoring) |
| 결합 | 산술 평균, 가중치 없음 |
| 표시 단위 | 0~100% |
| 별개 지수 | Coding Agent Index — SWE-Bench-Pro-Hard-AA + Terminal-Bench v2 + SWE-Atlas-QnA |
| 빠진 것 | LiveCodeBench (v3에서 제외), SWE-Bench류, HumanEval 등 |
