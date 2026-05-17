"""Dataset loaders and prompt builders.

Templates are copied verbatim from the official CacheBlend example/utils.py
(see workspace/spec/ambiguity_log.json for the exact strings).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

# --- Prompt template constants from example/utils.py -------------------------
QA_PREFIX = (
    "Answer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n"
)
QA_QUERY = (
    "\n\nAnswer the question based on the given passages. "
    "Answer the question within 5 words. Do NOT repeat the question or "
    "output any other words. Question: {question}\nAnswer:"
)

# Mistral instruction wrapping (token ids 733, 16289, 28793 etc.).
INST_OPEN = " [INST]"
INST_CLOSE = " [/INST]"


# --------------------------------------------------------------------- types
@dataclass
class Example:
    question: str
    contexts: List[dict]   # list of {"title": str, "text": str}
    answers: List[str]     # acceptable gold answers
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ loaders
def _load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_answers(raw) -> List[str]:
    """Bundled JSON files use either ['a', 'b'] or [['a','b'], ['c']]."""
    flat: List[str] = []
    for x in raw:
        if isinstance(x, list):
            flat.extend(str(y) for y in x)
        else:
            flat.append(str(x))
    return flat


def load_wikimqa(path: str, n: Optional[int] = None) -> List[Example]:
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
            )
        )
    return out


def load_musique(path: str, n: Optional[int] = None) -> List[Example]:
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
            )
        )
    return out


def load_samsum(path: str, n: Optional[int] = None) -> List[Example]:
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
                metadata={"input": ex.get("input", "")},
            )
        )
    return out


def load_hotpotqa(path: str, n: Optional[int] = None) -> List[Example]:
    """HotpotQA (distractor split via LongBench / HF download).

    Same JSON schema as wikimqa_s.json: list of
    ``{"question", "ctxs": [{"title","text"}], "answers"}``.
    """
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
            )
        )
    return out


def load_multihop_rag(path: str, n: Optional[int] = None) -> List[Example]:
    """MultiHop-RAG (Tang & Yang, 2024).

    Same JSON schema as wikimqa_s.json. Each query needs 2-4 news articles
    to answer; the evidence_list from the original release is reshaped into
    one ``ctxs`` entry per source article.
    """
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
                metadata={"question_type": ex.get("question_type", "")},
            )
        )
    return out


def load_hover(path: str, n: Optional[int] = None) -> List[Example]:
    """HoVer (Jiang et al., EMNLP 2020) -- multi-hop claim verification.

    The question is the CLAIM, the gold answer is the verdict label string
    ("SUPPORTED" / "NOT_SUPPORTED"). Contexts are the wiki abstracts of
    each supporting article. Same JSON schema as wikimqa_s.json on disk.
    """
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex["question"],   # = claim
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),  # = [label_string]
                metadata={"num_hops": ex.get("num_hops"), "task": "claim_verification"},
            )
        )
    return out


def load_multinews(path: str, n: Optional[int] = None) -> List[Example]:
    """MultiNews (Fabbri et al., 2019) -- multi-document news summarization.

    Each example has 2+ news articles as ``ctxs`` and a single gold summary
    string in ``answers``. Same on-disk schema as ``samsum.json``. The
    paper uses 60 examples; metric is Rouge-L.
    """
    data = _load_json(path)
    out: List[Example] = []
    for ex in data[: n if n else len(data)]:
        out.append(
            Example(
                question=ex.get("question", ""),
                contexts=ex["ctxs"],
                answers=_flatten_answers(ex["answers"]),
                metadata={"input": ex.get("input", "")},
            )
        )
    return out


# ----------------------------------------------------------------- chunking
def chunk_text(text: str, tokenizer, chunk_size_tokens: int = 512) -> List[str]:
    """Naive token-budgeted chunking (no overlap)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    for start in range(0, len(ids), chunk_size_tokens):
        chunk_ids = ids[start : start + chunk_size_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


# -------------------------------------------------------------- prompt build
def build_qa_prompt(question: str, contexts: List[dict]) -> tuple[str, List[str]]:
    """Return ``(full_prompt, per_chunk_strings)``.

    The per-chunk strings are what should be precomputed as standalone KV
    blocks. ``full_prompt`` is the concatenation that ``full_recompute`` uses.
    """
    chunk_strs: List[str] = []
    # First chunk includes INST_OPEN + QA_PREFIX so the Mistral chat wrapping
    # is present. Subsequent chunks are bare context blocks.
    first = INST_OPEN + " " + QA_PREFIX
    for i, ctx in enumerate(contexts):
        body = f"{ctx.get('title','')}\n\n{ctx.get('text','')}\n\n"
        if i == 0:
            chunk_strs.append(first + body)
        else:
            chunk_strs.append(body)
    query_str = QA_QUERY.format(question=question) + INST_CLOSE
    full_prompt = "".join(chunk_strs) + query_str
    return full_prompt, chunk_strs


def build_claim_verification_prompt(
    claim: str, contexts: List[dict]
) -> tuple[str, List[str]]:
    """HoVer-style claim verification.

    Reads several wiki abstracts and decides if the claim is SUPPORTED or
    NOT_SUPPORTED. We constrain the output vocabulary in the suffix so F1 /
    string-match scoring is well-defined.
    """
    cv_prefix = (
        "Decide whether the claim is SUPPORTED or NOT_SUPPORTED by the "
        "given passages. Only output one of the two labels.\n\n"
        "The following are given passages.\n"
    )
    cv_query = (
        "\n\nClaim: {claim}\n"
        "Answer SUPPORTED or NOT_SUPPORTED only. Do NOT output any other "
        "words.\nAnswer:"
    )
    chunk_strs: List[str] = []
    first = INST_OPEN + " " + cv_prefix
    for i, ctx in enumerate(contexts):
        body = f"{ctx.get('title','')}\n\n{ctx.get('text','')}\n\n"
        chunk_strs.append((first if i == 0 else "") + body)
    query_str = cv_query.format(claim=claim) + INST_CLOSE
    full_prompt = "".join(chunk_strs) + query_str
    return full_prompt, chunk_strs


def build_summarization_prompt(
    dialogue: str, contexts: List[dict]
) -> tuple[str, List[str]]:
    """SAMSum-style few-shot summarization."""
    chunk_strs: List[str] = []
    first = INST_OPEN + " "
    for i, ctx in enumerate(contexts):
        body = ctx.get("text", "") + "\n\n"
        if i == 0:
            chunk_strs.append(first + body)
        else:
            chunk_strs.append(body)
    query_str = "\n" + dialogue.strip() + "\nSummary:" + INST_CLOSE
    full_prompt = "".join(chunk_strs) + query_str
    return full_prompt, chunk_strs


def build_multinews_prompt(contexts: List[dict]) -> tuple[str, List[str]]:
    """MultiNews: multi-document news summarization.

    Each context is one news article; the model is asked to write a
    single summary covering all of them. We do NOT take a separate query
    string -- the articles themselves are the input. The first chunk
    carries the Mistral [INST] wrapping and a brief instruction; the
    suffix asks for the summary.
    """
    mn_prefix = (
        "Write a concise summary of the following news articles. "
        "Cover the key facts in all articles in 2-4 sentences. "
        "Do NOT output anything other than the summary.\n\n"
        "Articles:\n"
    )
    mn_query = "\nSummary:"
    chunk_strs: List[str] = []
    first = INST_OPEN + " " + mn_prefix
    for i, ctx in enumerate(contexts):
        body = (ctx.get("text", "") or "").rstrip() + "\n\n"
        chunk_strs.append((first if i == 0 else "") + body)
    query_str = mn_query + INST_CLOSE
    full_prompt = "".join(chunk_strs) + query_str
    return full_prompt, chunk_strs
