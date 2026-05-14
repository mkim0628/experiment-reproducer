from eval.metrics import (
    compute_f1,
    compute_f1_max,
    compute_rouge_l,
    normalize_answer,
)


def test_normalize_answer_strips_articles_and_punct() -> None:
    assert normalize_answer("The QUICK, brown fox!") == "quick brown fox"
    assert normalize_answer("an apple a day") == "apple day"
    assert normalize_answer("") == ""


def test_f1_known_pairs_word_tokenizer() -> None:
    # Without a tokenizer we fall back to word tokenization (still
    # SQuAD-style normalization).
    assert compute_f1("barack obama", "Barack Obama") == 1.0
    assert compute_f1("paris", "london") == 0.0
    # Partial overlap.
    pred, gold = "the quick brown fox", "quick fox"
    f1 = compute_f1(pred, gold)
    assert 0.0 < f1 < 1.0


def test_f1_max_picks_best() -> None:
    pred = "Barack Obama"
    golds = ["Donald Trump", "barack obama", "joe biden"]
    assert compute_f1_max(pred, golds) == 1.0


def test_rouge_l_known_pairs() -> None:
    # Exact match -> 1.0
    assert abs(compute_rouge_l("hello world", "hello world") - 1.0) < 1e-6
    # No overlap -> 0.0
    assert compute_rouge_l("foo", "bar") == 0.0
    # Cross-check against rouge_score directly to pin the convention.
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pred = "the quick brown fox jumps"
    gold = "the quick fox jumps over"
    expected = scorer.score(gold, pred)["rougeL"].fmeasure
    assert abs(compute_rouge_l(pred, gold) - expected) < 1e-9
