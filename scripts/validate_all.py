#!/usr/bin/env python3
"""
Validate every artifact present under workspace/ against its JSON Schema.

Used by the orchestrator before each user gate so that downstream agents
never see a malformed artifact.

Usage:
    python scripts/validate_all.py

Exit codes:
    0  every artifact present is valid (missing artifacts are OK — pipeline
       may not have reached that step yet)
    1  one or more artifacts failed validation
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "workspace"
VALIDATE = REPO_ROOT / "scripts" / "validate.py"

# Same list as ARTIFACT_TO_SCHEMA in validate.py, but ordered to match the
# pipeline so the report reads top-to-bottom.
PIPELINE_ARTIFACTS: list[str] = [
    "paper/structured.json",
    "analysis/analysis.json",
    "references/references.json",
    "spec/method_spec.json",
    "spec/ambiguity_log.json",
    "plan/plan.json",
    "validation/report.json",
]


def main() -> None:
    failures: list[str] = []
    skipped: list[str] = []
    passed: list[str] = []

    for rel in PIPELINE_ARTIFACTS:
        path = WORKSPACE / rel
        if not path.exists():
            skipped.append(rel)
            continue
        result = subprocess.run(
            [sys.executable, str(VALIDATE), str(path)],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode == 0:
            passed.append(rel)
        else:
            failures.append(rel)

    sys.stdout.write("\n--- summary ---\n")
    sys.stdout.write(f"passed:  {len(passed)}\n")
    sys.stdout.write(f"failed:  {len(failures)}\n")
    sys.stdout.write(f"missing: {len(skipped)} (not produced yet — OK)\n")
    if failures:
        sys.stdout.write("failed artifacts:\n")
        for rel in failures:
            sys.stdout.write(f"  - workspace/{rel}\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
