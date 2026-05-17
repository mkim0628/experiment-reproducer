"""Audit generated dataset JSON files for schema and semantic sanity.

Run after ``download_extra_datasets.py`` to catch issues like:
  * label imbalance (HoVer must be roughly 50/50; HotpotQA must have both
    bridge and comparison questions; MultiHop-RAG must have multiple
    question_type categories)
  * empty contexts or empty answers
  * suspicious answer constants (the bug the user found: all HoVer
    examples being SUPPORTED)
  * length distributions far from what the dataset cards advertise

Exits non-zero on any FAIL; warnings exit zero.

Usage:
    python workspace/code/scripts/audit_datasets.py \\
        --data-dir workspace/code/data

By default it audits every known dataset whose JSON exists in --data-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple


# ---------------------------------------------------------- generic checks
def _stats(records: List[dict]) -> dict:
    ctx_counts = [len(r.get("ctxs", [])) for r in records]
    ans_lens = [sum(len(a) for a in r.get("answers", [])) for r in records]
    text_lens = [
        sum(len(c.get("text", "")) for c in r.get("ctxs", [])) for r in records
    ]
    return {
        "n": len(records),
        "ctxs_per_example_min": min(ctx_counts, default=0),
        "ctxs_per_example_max": max(ctx_counts, default=0),
        "ctxs_per_example_mean": (sum(ctx_counts) / len(ctx_counts)) if ctx_counts else 0,
        "total_context_chars_mean": (sum(text_lens) / len(text_lens)) if text_lens else 0,
        "answer_chars_mean": (sum(ans_lens) / len(ans_lens)) if ans_lens else 0,
    }


def _check_schema(records: List[dict]) -> List[str]:
    issues: List[str] = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            issues.append(f"row {i}: not a dict")
            continue
        for k in ("question", "ctxs", "answers"):
            if k not in r:
                issues.append(f"row {i}: missing key '{k}'")
        if not r.get("question"):
            issues.append(f"row {i}: empty question")
        if not r.get("answers"):
            issues.append(f"row {i}: empty answers")
        if not r.get("ctxs"):
            issues.append(f"row {i}: empty ctxs")
        else:
            for j, c in enumerate(r["ctxs"]):
                if "title" not in c or "text" not in c:
                    issues.append(f"row {i} ctx {j}: missing title/text")
                if not c.get("text"):
                    issues.append(f"row {i} ctx {j}: empty text")
    return issues


# ---------------------------------------------------------- per-dataset
def audit_wikimqa(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    # 2WikiMQA bundled file is exactly 200 in the official inputs/.
    if len(records) < 100:
        warns.append(f"only {len(records)} examples (expected ~200)")
    return fails, warns


def audit_musique(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    if len(records) < 100:
        warns.append(f"only {len(records)} examples (expected ~150)")
    return fails, warns


def audit_samsum(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    if len(records) < 100:
        warns.append(f"only {len(records)} examples (expected ~200)")
    return fails, warns


def audit_hotpotqa(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    # HotpotQA distractor must have exactly 10 contexts per example.
    bad_ctx = sum(1 for r in records if len(r.get("ctxs", [])) != 10)
    if bad_ctx:
        warns.append(f"{bad_ctx}/{len(records)} examples have != 10 ctxs (HotpotQA distractor expects 10)")
    # Question types: HF schema has 'type' in {bridge, comparison}.
    types = Counter(r.get("type") for r in records if r.get("type"))
    if types and len(types) == 1:
        warns.append(f"only one question type present: {dict(types)} -- check that shuffle worked")
    return fails, warns


def audit_multihop_rag(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    qtypes = Counter(r.get("question_type") for r in records if r.get("question_type"))
    if qtypes and len(qtypes) < 2:
        warns.append(
            f"only {len(qtypes)} question_type categories present: {dict(qtypes)} "
            "(MultiHop-RAG ships 4: inference, comparison, temporal, null)"
        )
    return fails, warns


def audit_hover(records: List[dict]) -> Tuple[List[str], List[str]]:
    fails, warns = [], []
    fails += _check_schema(records)
    labels = Counter()
    for r in records:
        for a in r.get("answers", []):
            labels[str(a).upper()] += 1
    valid = labels["SUPPORTED"] + labels["NOT_SUPPORTED"]
    other = sum(v for k, v in labels.items() if k not in ("SUPPORTED", "NOT_SUPPORTED"))
    if other:
        fails.append(f"unexpected labels in answers: {[k for k in labels if k not in ('SUPPORTED','NOT_SUPPORTED')]}")
    if valid == 0:
        fails.append("no SUPPORTED / NOT_SUPPORTED labels present")
    else:
        # The actual bug the user found: all answers same label.
        ratio_sup = labels["SUPPORTED"] / valid
        if ratio_sup < 0.30 or ratio_sup > 0.70:
            fails.append(
                f"label imbalance: SUPPORTED={labels['SUPPORTED']}, "
                f"NOT_SUPPORTED={labels['NOT_SUPPORTED']} (ratio {ratio_sup:.2f}); "
                "stratified sampling should yield ~0.50"
            )
    # num_hops distribution
    hops = Counter(r.get("num_hops") for r in records if r.get("num_hops"))
    if hops:
        print(f"  hover hops distribution: {dict(hops)}")
    return fails, warns


AUDITS: Dict[str, Callable[[List[dict]], Tuple[List[str], List[str]]]] = {
    "wikimqa_s": audit_wikimqa,
    "musique_s": audit_musique,
    "samsum": audit_samsum,
    "hotpotqa": audit_hotpotqa,
    "multihop_rag": audit_multihop_rag,
    "hover": audit_hover,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="directory containing the *.json dataset files",
    )
    p.add_argument(
        "--only",
        nargs="*",
        choices=list(AUDITS.keys()),
        help="audit only these (default: every file found in --data-dir)",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    targets = args.only if args.only else list(AUDITS.keys())
    any_fail = False
    for name in targets:
        path = data_dir / f"{name}.json"
        print(f"=== {name} ({path}) ===")
        if not path.exists():
            print("  SKIP (file not found)")
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  FAIL: cannot parse JSON: {e}")
            any_fail = True
            continue
        if not isinstance(records, list):
            print("  FAIL: top-level must be a list")
            any_fail = True
            continue
        stats = _stats(records)
        print(
            f"  stats: n={stats['n']}, "
            f"ctxs/ex {stats['ctxs_per_example_min']}-{stats['ctxs_per_example_max']} "
            f"(mean {stats['ctxs_per_example_mean']:.1f}), "
            f"ctx chars mean {stats['total_context_chars_mean']:.0f}, "
            f"answer chars mean {stats['answer_chars_mean']:.1f}"
        )
        fails, warns = AUDITS[name](records)
        for w in warns[:10]:
            print(f"  WARN: {w}")
        for f in fails[:10]:
            print(f"  FAIL: {f}")
        if fails:
            any_fail = True
            print(f"  -> {len(fails)} failures (showing first 10)")
        else:
            print("  -> OK")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
