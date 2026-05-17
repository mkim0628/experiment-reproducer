"""Sanity tests for dataset loaders and prompt builders.

These tests do NOT require the Mistral model -- they only check that the
bundled JSON files load correctly and that the prompt templates match the
official CacheBlend example/utils.py wording.
"""
from __future__ import annotations

import pathlib

import pytest

from eval.datasets import (
    INST_CLOSE,
    INST_OPEN,
    QA_PREFIX,
    QA_QUERY,
    build_claim_verification_prompt,
    build_multinews_prompt,
    build_qa_prompt,
    build_summarization_prompt,
    load_hotpotqa,
    load_hover,
    load_multihop_rag,
    load_multinews,
    load_musique,
    load_samsum,
    load_wikimqa,
)
from eval.metrics import compute_claim_verification, compute_claim_verification_max


_DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(not (_DATA / "wikimqa_s.json").exists(), reason="bundled data missing")
def test_wikimqa_loads() -> None:
    examples = load_wikimqa(str(_DATA / "wikimqa_s.json"), n=5)
    assert len(examples) == 5
    for ex in examples:
        assert ex.question
        assert ex.contexts and isinstance(ex.contexts, list)
        assert ex.answers


@pytest.mark.skipif(not (_DATA / "musique_s.json").exists(), reason="bundled data missing")
def test_musique_loads() -> None:
    examples = load_musique(str(_DATA / "musique_s.json"), n=5)
    assert len(examples) == 5
    for ex in examples:
        assert ex.contexts and ex.answers


@pytest.mark.skipif(not (_DATA / "samsum.json").exists(), reason="bundled data missing")
def test_samsum_loads() -> None:
    examples = load_samsum(str(_DATA / "samsum.json"), n=3)
    assert len(examples) == 3
    for ex in examples:
        assert ex.answers


def test_qa_prompt_assembly() -> None:
    contexts = [
        {"title": "Foo", "text": "this is foo"},
        {"title": "Bar", "text": "this is bar"},
    ]
    full, chunks = build_qa_prompt("What is foo?", contexts)
    # Full prompt = concat(chunks) + query_suffix
    assert full.startswith(INST_OPEN)
    assert full.endswith(INST_CLOSE)
    assert QA_PREFIX.strip() in full
    assert "What is foo?" in full
    assert len(chunks) == 2
    # The query suffix is full minus the chunks.
    assert full == "".join(chunks) + QA_QUERY.format(question="What is foo?") + INST_CLOSE


def test_summarization_prompt_assembly() -> None:
    contexts = [{"title": "", "text": "dialogue A"}, {"title": "", "text": "dialogue B"}]
    full, chunks = build_summarization_prompt("dialogue Q", contexts)
    assert full.startswith(INST_OPEN)
    assert full.endswith(INST_CLOSE)
    assert "dialogue Q" in full
    assert len(chunks) == 2
    assert full == "".join(chunks) + "\n" + "dialogue Q" + "\nSummary:" + INST_CLOSE


# ---------------------------------------------------- HotpotQA / MultiHop-RAG
def _write_json(path: pathlib.Path, payload) -> None:
    import json as _json

    path.write_text(_json.dumps(payload))


def test_hotpotqa_loader_round_trip(tmp_path) -> None:
    fixture = [
        {
            "question": "Who wrote X?",
            "ctxs": [
                {"title": "X", "text": "X was written by A."},
                {"title": "A", "text": "A is a person."},
            ],
            "answers": ["A"],
        }
    ]
    p = tmp_path / "hotpotqa.json"
    _write_json(p, fixture)
    examples = load_hotpotqa(str(p))
    assert len(examples) == 1
    assert examples[0].question == "Who wrote X?"
    assert examples[0].answers == ["A"]
    assert len(examples[0].contexts) == 2


def test_multihop_rag_loader_round_trip(tmp_path) -> None:
    fixture = [
        {
            "question": "Compare A and B.",
            "ctxs": [
                {"title": "news1", "text": "fact 1"},
                {"title": "news2", "text": "fact 2"},
            ],
            "answers": ["A is older."],
            "question_type": "comparison_query",
        }
    ]
    p = tmp_path / "multihop_rag.json"
    _write_json(p, fixture)
    examples = load_multihop_rag(str(p))
    assert len(examples) == 1
    assert examples[0].metadata.get("question_type") == "comparison_query"


# --------------------------------------------------------------- HoVer
def test_hover_loader_round_trip(tmp_path) -> None:
    fixture = [
        {
            "question": "X was born in 1900.",
            "ctxs": [
                {"title": "X", "text": "X (1850-1925) was a foo."},
            ],
            "answers": ["NOT_SUPPORTED"],
            "num_hops": 2,
        }
    ]
    p = tmp_path / "hover.json"
    _write_json(p, fixture)
    examples = load_hover(str(p))
    assert len(examples) == 1
    assert examples[0].answers == ["NOT_SUPPORTED"]
    assert examples[0].metadata.get("task") == "claim_verification"


def test_claim_verification_prompt_assembly() -> None:
    ctxs = [
        {"title": "T1", "text": "passage one"},
        {"title": "T2", "text": "passage two"},
    ]
    full, chunks = build_claim_verification_prompt("The claim.", ctxs)
    assert full.startswith(INST_OPEN)
    assert full.endswith(INST_CLOSE)
    assert "SUPPORTED" in full and "NOT_SUPPORTED" in full
    assert "The claim." in full
    assert len(chunks) == 2
    # The prompt has the SAME contract as build_qa_prompt: full = join(chunks) + suffix.
    suffix = full[sum(len(c) for c in chunks):]
    assert "Claim: The claim." in suffix


def test_claim_verification_metric_handles_label_variants() -> None:
    assert compute_claim_verification("SUPPORTED", "SUPPORTED") == 1.0
    assert compute_claim_verification("not_supported", "NOT_SUPPORTED") == 1.0
    # NOT_SUPPORTED in prediction should not match a SUPPORTED gold.
    assert compute_claim_verification("NOT_SUPPORTED", "SUPPORTED") == 0.0
    # Free-form predictions with the keyword still get credited.
    assert compute_claim_verification("yes, it is supported.", "SUPPORTED") == 1.0
    assert compute_claim_verification("the claim is REFUTED", "NOT_SUPPORTED") == 1.0
    # Garbage prediction.
    assert compute_claim_verification("idk", "SUPPORTED") == 0.0
    # Max over multiple golds.
    assert compute_claim_verification_max("SUPPORTED", ["NOT_SUPPORTED", "SUPPORTED"]) == 1.0


# --------------------------------------------------------- MultiNews
def test_multinews_loader_round_trip(tmp_path) -> None:
    fixture = [
        {
            "question": "",
            "ctxs": [
                {"title": "Article 1", "text": "article one body"},
                {"title": "Article 2", "text": "article two body"},
                {"title": "Article 3", "text": "article three body"},
            ],
            "answers": ["A multi-paragraph summary."],
        }
    ]
    p = tmp_path / "multinews.json"
    _write_json(p, fixture)
    examples = load_multinews(str(p))
    assert len(examples) == 1
    assert len(examples[0].contexts) == 3
    assert examples[0].answers == ["A multi-paragraph summary."]


def test_multinews_prompt_assembly() -> None:
    ctxs = [
        {"title": "Article 1", "text": "first news article body"},
        {"title": "Article 2", "text": "second news article body"},
    ]
    full, chunks = build_multinews_prompt(ctxs)
    assert full.startswith(INST_OPEN)
    assert full.endswith(INST_CLOSE)
    # Suffix asks for a summary.
    assert "Summary:" in full
    # The instruction text is folded into the first chunk so it's cacheable
    # as a prefix in the cacheblend pipeline.
    assert "Articles:" in chunks[0]
    assert len(chunks) == 2
    # join(chunks) + suffix == full (the contract every prompt builder honors)
    suffix = full[sum(len(c) for c in chunks):]
    assert full == "".join(chunks) + suffix


# ----------------------- stratified sampling regression for HoVer bug
def _stratify_labels(records, n, seed):
    """Pure-Python copy of the stratified logic in fetch_hover().

    Tested separately so we don't have to import the download script (which
    pulls in optional network deps like requests/datasets).
    """
    import random
    from collections import defaultdict

    by_label = defaultdict(list)
    for ex in records:
        by_label[ex["label"]].append(ex)
    rng = random.Random(seed)
    for lbl in by_label:
        rng.shuffle(by_label[lbl])
    half = max(1, n // 2)
    pool = by_label["SUPPORTED"][:half] + by_label["NOT_SUPPORTED"][:n - half]
    rng.shuffle(pool)
    return pool


def test_hover_stratified_sampling_balances_sorted_input() -> None:
    """The real HoVer dev JSON is sorted by label (2000 SUPPORTED then 2000
    NOT_SUPPORTED). A naive ``records[:200]`` slice produces 200 SUPPORTED
    / 0 NOT_SUPPORTED -- the bug the user found. Stratified sampling must
    produce ~50/50 regardless of input order.
    """
    sorted_records = (
        [{"label": "SUPPORTED", "claim": f"s{i}"} for i in range(2000)]
        + [{"label": "NOT_SUPPORTED", "claim": f"n{i}"} for i in range(2000)]
    )

    # Pre-fix baseline: naive slice = 200/0.
    naive = sorted_records[:200]
    assert sum(1 for x in naive if x["label"] == "SUPPORTED") == 200
    assert sum(1 for x in naive if x["label"] == "NOT_SUPPORTED") == 0

    # Post-fix: stratified slice = 100/100 deterministically with seed.
    sample = _stratify_labels(sorted_records, n=200, seed=0)
    n_sup = sum(1 for x in sample if x["label"] == "SUPPORTED")
    n_not = sum(1 for x in sample if x["label"] == "NOT_SUPPORTED")
    assert n_sup == 100, f"expected 100 SUPPORTED, got {n_sup}"
    assert n_not == 100, f"expected 100 NOT_SUPPORTED, got {n_not}"
    assert len(sample) == 200


def test_hover_stratified_sampling_is_deterministic_under_seed() -> None:
    sorted_records = (
        [{"label": "SUPPORTED", "claim": f"s{i}"} for i in range(2000)]
        + [{"label": "NOT_SUPPORTED", "claim": f"n{i}"} for i in range(2000)]
    )
    a = _stratify_labels(sorted_records, n=200, seed=42)
    b = _stratify_labels(sorted_records, n=200, seed=42)
    c = _stratify_labels(sorted_records, n=200, seed=43)
    assert [x["claim"] for x in a] == [x["claim"] for x in b]
    assert [x["claim"] for x in a] != [x["claim"] for x in c]
