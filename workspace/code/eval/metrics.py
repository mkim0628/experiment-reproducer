"""SQuAD-style token F1 and Rouge-L metrics.

Adapted from: https://github.com/YaoJiayi/CacheBlend/blob/main/example/utils.py
(compute_f1, normalize_answer, compute_rl).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Optional

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:  # pragma: no cover
    _HAS_ROUGE = False


_ROUGE_SCORER = None


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        if not _HAS_ROUGE:
            raise RuntimeError(
                "rouge_score is required for compute_rouge_l; install via "
                "`pip install rouge_score`."
            )
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


# --------------------------------------------------------------------- F1
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, strip punct, drop articles, fix whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _toks(text: str, tokenizer) -> List[int]:
    """Tokenize and drop the BOS token if the tokenizer adds one.

    Matches the official compute_f1 which calls ``tokenizer.encode(text)[1:]``.
    """
    if tokenizer is None:
        return normalize_answer(text).split()
    norm = normalize_answer(text)
    if not norm:
        return []
    ids = tokenizer.encode(norm)
    # Mirror example/utils.py: always drop the leading BOS (encode(..)[1:]).
    if len(ids) > 1:
        return ids[1:]
    return ids


def _f1_from_toks(pred_toks: List[int], gold_toks: List[int]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_f1(pred: str, gold: str, tokenizer=None) -> float:
    """Token-id F1 using the LLM tokenizer (BOS dropped) per SQuAD convention."""
    return _f1_from_toks(_toks(pred, tokenizer), _toks(gold, tokenizer))


def compute_f1_max(pred: str, golds: Iterable[str], tokenizer=None) -> float:
    """Max F1 over a list of acceptable gold answers."""
    return max((compute_f1(pred, g, tokenizer) for g in golds), default=0.0)


# ------------------------------------------------------------------ Rouge-L
def compute_rouge_l(pred: str, gold: str) -> float:
    """Rouge-L f-measure per google-research/rouge with use_stemmer=True."""
    scorer = _get_rouge_scorer()
    return scorer.score(gold, pred)["rougeL"].fmeasure
