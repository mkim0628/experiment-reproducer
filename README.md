# experiment-reproducer

논문 한 편을 가리키면 그 핵심 기술을 재현하는 코드까지 만들어내는 Claude Code 기반 멀티 에이전트 파이프라인.

## 파이프라인 개요

```
 paper ref
    │
    ▼
┌───────────────┐
│ paper-fetcher │  PDF + 부록 → 구조화 텍스트
└───────┬───────┘
        ▼
┌────────────────┐
│ paper-analyzer │  문제 / 기여 / 데이터셋 / 지표 / 베이스라인
└───────┬────────┘
        ▼
┌──────────────────┐     ┌──────────────────┐
│ method-extractor │     │ reference-hunter │
│ (spec + UNKNOWNs)│     │ (공식코드, 라이브러리, 데이터셋)
└───────┬──────────┘     └────────┬─────────┘
        │                         │
        ▼ (옵션)                  │
┌──────────────────────┐          │
│ ambiguity-resolver   │          │
│ UNKNOWN → 근거 있는  │          │
│ 값 또는 사용자 질문  │          │
└───────┬──────────────┘          │
        └───────────┬─────────────┘
                    ▼
            🚪 USER GATE 1 — 분석 요약 검토
                    ▼
        ┌──────────────────────┐
        │ implementation-planner│  모듈/인터페이스/테스트 계획
        └──────────┬───────────┘
                    ▼
            🚪 USER GATE 2 — 계획 승인 & open decisions
                    ▼
                ┌───────┐
                │ coder │   구현 + 단위 테스트
                └───┬───┘
                    ▼
        ┌────────────────────────┐
        │ experiment-validator   │ 논문 수치와 비교 + 불일치 진단
        └──────────┬─────────────┘
                    ▼
            🚪 USER GATE 3 — 재현 판정 검토
                    ▼
        ┌──────────────────────┐
        │ documentation-writer │  README + 재현 노트
        └──────────────────────┘
```

## 사용법

### 1) 파이프라인 실행 (Claude Code 안에서)

Claude Code 세션에서:

```
/reproduce-paper 2301.12345
```

또는 PDF URL / DOI / 로컬 PDF 경로를 인자로 넘겨도 됩니다.

오케스트레이터(`.claude/commands/reproduce-paper.md`)가 각 단계의 서브에이전트를 순서대로 호출하고, 3개의 사용자 게이트에서 멈춰 확인을 요청합니다.

### 2) 개별 에이전트 직접 호출

특정 단계만 다시 돌리고 싶을 때는 Claude Code Task 도구로 서브에이전트를 직접 호출하면 됩니다 (예: `paper-analyzer`만 재실행).

## 디렉토리 구조

```
.
├── .claude/
│   ├── agents/                       # 서브에이전트 정의 (각각 단일 책임)
│   │   ├── paper-fetcher.md
│   │   ├── paper-analyzer.md
│   │   ├── method-extractor.md
│   │   ├── reference-hunter.md
│   │   ├── implementation-planner.md
│   │   ├── coder.md
│   │   ├── experiment-validator.md
│   │   ├── ambiguity-resolver.md     # 선택
│   │   └── documentation-writer.md   # 선택
│   └── commands/
│       └── reproduce-paper.md        # 오케스트레이션 슬래시 커맨드
└── workspace/                        # 각 단계 산출물이 여기에 누적됨
    ├── paper/        # paper.pdf, paper.txt, structured.json
    ├── analysis/     # analysis.json
    ├── spec/         # method_spec.md, ambiguity_log.md
    ├── references/   # references.json
    ├── plan/         # plan.md
    ├── code/         # 실제 구현 + CODER_NOTES.md + README.md
    ├── validation/   # report.md
    └── docs/         # REPRODUCTION_NOTES.md
```

## 설계 원칙

1. **단일 책임** — 각 에이전트는 한 가지 일만 하고, 다음 에이전트가 읽을 수 있는 구조화된 산출물(JSON/Markdown)만 남깁니다.
2. **명시적 spec 단계** — 논문에서 곧장 코드로 가지 않습니다. `method-extractor`가 만든 spec이 코더의 진실 공급원이 됩니다.
3. **`[UNKNOWN]` 마커** — 논문에 없는 디테일은 환각하지 않고 명시적으로 표시합니다. `ambiguity-resolver`가 근거와 함께 채우거나 사용자에게 escalate.
4. **사용자 게이트 3곳** — 분석 후 / 계획 후 / 검증 후. 자동화는 효율을 위한 것이지 검토를 건너뛰기 위한 것이 아닙니다.
5. **재현은 baseline부터** — `experiment-validator`는 paper 수치와 우리 결과를 같은 평가 경로로 비교하고, 차이가 나면 표준 체크리스트로 진단합니다.

## 새 에이전트 추가하기

`.claude/agents/<name>.md` 한 파일을 추가하면 됩니다. 형식:

```markdown
---
name: <agent-name>
description: 이 에이전트를 언제 써야 하는지
tools: Read, Write, Bash, ...
---

# 시스템 프롬프트 본문
```

기존 에이전트들이 동일한 패턴을 따르니 그대로 참고하세요.
