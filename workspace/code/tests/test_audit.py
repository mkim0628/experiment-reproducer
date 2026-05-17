"""Tests for the dataset audit script.

We treat audit_datasets.py as a library and call its per-dataset auditors
directly with synthetic fixtures. This lets us regression-test the actual
bug the user found (all HoVer answers being SUPPORTED) without needing
the live download.
"""
from __future__ import annotations

import importlib.util
import pathlib

_AUDIT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "audit_datasets.py"
)
_spec = importlib.util.spec_from_file_location("audit_datasets", str(_AUDIT_PATH))
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)  # type: ignore[union-attr]


# --------------------------------------------------------- HoVer
def _hover_record(label: str, ctx_text: str = "context body") -> dict:
    return {
        "question": "Some claim.",
        "ctxs": [{"title": "T1", "text": ctx_text}],
        "answers": [label],
        "num_hops": 3,
    }


def test_audit_hover_detects_all_supported_bug() -> None:
    # 200 records all SUPPORTED -- the exact bug the user reported.
    records = [_hover_record("SUPPORTED") for _ in range(200)]
    fails, _ = audit.audit_hover(records)
    assert any("imbalance" in f or "no SUPPORTED" in f for f in fails), (
        f"audit should flag all-SUPPORTED as imbalance; got fails={fails}"
    )


def test_audit_hover_passes_balanced_input() -> None:
    records = (
        [_hover_record("SUPPORTED") for _ in range(100)]
        + [_hover_record("NOT_SUPPORTED") for _ in range(100)]
    )
    fails, _ = audit.audit_hover(records)
    assert fails == [], f"balanced 50/50 input should pass; got {fails}"


def test_audit_hover_flags_unknown_labels() -> None:
    records = [_hover_record("SUPPORTED") for _ in range(50)] + [_hover_record("MAYBE")]
    fails, _ = audit.audit_hover(records)
    assert any("unexpected labels" in f for f in fails)


def test_audit_hover_catches_empty_ctxs() -> None:
    bad = _hover_record("SUPPORTED")
    bad["ctxs"] = []
    records = [bad] + [_hover_record("NOT_SUPPORTED") for _ in range(100)]
    fails, _ = audit.audit_hover(records)
    assert any("empty ctxs" in f for f in fails)


# ---------------------------------------------------- HotpotQA / MultiHop-RAG
def test_audit_hotpotqa_warns_on_wrong_ctx_count() -> None:
    # Distractor split must have 10 contexts per question; flag if not.
    rec = {
        "question": "Q?",
        "ctxs": [{"title": f"T{i}", "text": "x"} for i in range(5)],  # only 5, not 10
        "answers": ["a"],
    }
    fails, warns = audit.audit_hotpotqa([rec])
    assert any("10 ctxs" in w for w in warns)


def test_audit_multihop_rag_warns_on_single_question_type() -> None:
    rec = {
        "question": "Q?",
        "ctxs": [{"title": "T", "text": "x"}],
        "answers": ["a"],
        "question_type": "inference_query",
    }
    fails, warns = audit.audit_multihop_rag([rec] * 50)
    assert any("question_type" in w for w in warns)


def test_audit_multinews_warns_on_single_article_examples() -> None:
    # MultiNews is multi-document by definition; flag single-article rows.
    rec_bad = {
        "question": "",
        "ctxs": [{"title": "Article 1", "text": "x" * 200}],
        "answers": ["A long enough summary " * 5],
    }
    fails, warns = audit.audit_multinews([rec_bad] * 10)
    assert any("< 2 source articles" in w for w in warns)


def test_audit_multinews_warns_on_trivial_summaries() -> None:
    rec_bad = {
        "question": "",
        "ctxs": [
            {"title": "Article 1", "text": "x" * 200},
            {"title": "Article 2", "text": "y" * 200},
        ],
        "answers": ["short"],
    }
    fails, warns = audit.audit_multinews([rec_bad] * 5)
    assert any("short summaries" in w for w in warns)


def test_audit_multinews_passes_on_realistic_input() -> None:
    rec = {
        "question": "",
        "ctxs": [
            {"title": "Article 1", "text": "first article body " * 50},
            {"title": "Article 2", "text": "second article body " * 50},
        ],
        "answers": ["A 2-3 sentence summary covering both articles." * 3],
    }
    fails, warns = audit.audit_multinews([rec] * 60)
    assert fails == [] and warns == []


def test_audit_passes_for_bundled_wikimqa_shape() -> None:
    # Mimic the real wikimqa_s.json shape: 10 contexts, short single answer.
    rec = {
        "question": "Where was the wife born?",
        "ctxs": [{"title": f"T{i}", "text": "passage text"} for i in range(10)],
        "answers": ["Ozalj"],
    }
    fails, warns = audit.audit_wikimqa([rec] * 200)
    assert fails == [] and warns == []
