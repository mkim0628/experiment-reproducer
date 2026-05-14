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
    build_qa_prompt,
    build_summarization_prompt,
    load_musique,
    load_samsum,
    load_wikimqa,
)


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
