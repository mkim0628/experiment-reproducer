#!/usr/bin/env python3
"""
Validate a single workspace artifact against its JSON Schema. Fail-fast.

Usage:
    python scripts/validate.py <artifact_path>

Exit codes:
    0  artifact is valid
    1  artifact is invalid (schema violation)
    2  artifact path is not recognized / no schema mapping
    3  IO or JSON parse error

The mapping from artifact path to schema is fixed (see ARTIFACT_TO_SCHEMA below)
so each agent always validates against the same contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError:
    sys.stderr.write(
        "ERROR: jsonschema is required. Install with: pip install -r requirements.txt\n"
    )
    sys.exit(3)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schemas"
WORKSPACE = REPO_ROOT / "workspace"

# Maps an artifact path (relative to workspace/) to its schema filename.
ARTIFACT_TO_SCHEMA: dict[str, str] = {
    "paper/structured.json": "structured.schema.json",
    "analysis/analysis.json": "analysis.schema.json",
    "references/references.json": "references.schema.json",
    "spec/method_spec.json": "method_spec.schema.json",
    "spec/ambiguity_log.json": "ambiguity_log.schema.json",
    "plan/plan.json": "plan.schema.json",
    "validation/report.json": "validation_report.schema.json",
}


def _resolve_relative(artifact_path: Path) -> str:
    """Return artifact path relative to workspace/ as a forward-slash string."""
    artifact_path = artifact_path.resolve()
    try:
        rel = artifact_path.relative_to(WORKSPACE.resolve())
    except ValueError:
        sys.stderr.write(
            f"ERROR: artifact must live under {WORKSPACE}, got {artifact_path}\n"
        )
        sys.exit(2)
    return rel.as_posix()


def _load_schema(schema_filename: str) -> dict:
    schema_path = SCHEMA_DIR / schema_filename
    if not schema_path.exists():
        sys.stderr.write(f"ERROR: schema file not found: {schema_path}\n")
        sys.exit(3)
    try:
        return json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"ERROR: schema {schema_path} is not valid JSON: {exc}\n")
        sys.exit(3)


def _load_instance(artifact_path: Path) -> object:
    if not artifact_path.exists():
        sys.stderr.write(f"ERROR: artifact does not exist: {artifact_path}\n")
        sys.exit(3)
    try:
        return json.loads(artifact_path.read_text())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"FAIL: {artifact_path} is not valid JSON: {exc}\n")
        sys.exit(1)


def validate(artifact_path: Path) -> None:
    rel = _resolve_relative(artifact_path)
    schema_name = ARTIFACT_TO_SCHEMA.get(rel)
    if schema_name is None:
        sys.stderr.write(
            f"ERROR: no schema mapping for workspace/{rel}.\n"
            f"Known artifacts:\n"
            + "\n".join(f"  - workspace/{p}" for p in sorted(ARTIFACT_TO_SCHEMA))
            + "\n"
        )
        sys.exit(2)

    schema = _load_schema(schema_name)
    instance = _load_instance(artifact_path)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if not errors:
        sys.stdout.write(f"OK   workspace/{rel}  (schema: {schema_name})\n")
        sys.exit(0)

    sys.stderr.write(
        f"FAIL workspace/{rel}  (schema: {schema_name}) — {len(errors)} error(s)\n"
    )
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        sys.stderr.write(f"  - at {path}: {err.message}\n")
    sys.exit(1)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.stderr.write("Usage: python scripts/validate.py <artifact_path>\n")
        sys.exit(2)
    validate(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
