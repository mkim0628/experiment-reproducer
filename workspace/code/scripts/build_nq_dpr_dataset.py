"""Fetch DPR Natural Questions retrievals and convert to the CacheBlend schema.

DPR provides 100 Wikipedia passages per NQ question with the gold passage
flagged via `has_answer`. Each passage is a 100-word Wikipedia chunk
(~130 tokens), much shorter than the CacheBlend paper's 512-token spec but
otherwise a perfect drop-in: the JSON schema is essentially the same as the
bundled `musique_s.json` / `wikimqa_s.json`, just with more ctxs.

Output:

* ``workspace/code/data/nq_dpr.json``   (default n=200, k=20)
* ``workspace/code/data/nq_dpr_report.json``

Run::

    python scripts/build_nq_dpr_dataset.py                 # 200 ex, 20 ctxs/q
    python scripts/build_nq_dpr_dataset.py --n 500 --k 50  # bigger
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
import urllib.request
from typing import Any, Dict, List

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DPR_URL = (
    "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz"
)


def _download(url: str, dst: pathlib.Path) -> None:
    if dst.exists():
        print(f"[download] cached -> {dst}")
        return
    print(f"[download] {url} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def _load_dpr(path: pathlib.Path) -> List[Dict[str, Any]]:
    print(f"[load] {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def convert(rows: List[Dict[str, Any]], n: int, k: int) -> List[Dict[str, Any]]:
    """Take first ``n`` queries, keep top ``k`` ctxs each, project to our schema."""
    out: List[Dict[str, Any]] = []
    for r in rows[:n]:
        ctxs = []
        for c in r["ctxs"][:k]:
            ctxs.append({"title": c.get("title", ""), "text": c["text"]})
        out.append(
            {
                "question": r["question"],
                "ctxs": ctxs,
                "answers": r["answers"],
            }
        )
    return out


def _flatten_answers(raw: Any) -> List[str]:
    out: List[str] = []
    for x in raw:
        if isinstance(x, list):
            out.extend(str(y) for y in x)
        else:
            out.append(str(x))
    return out


def validate(
    name: str,
    rows: List[Dict[str, Any]],
    source_rows: List[Dict[str, Any]],
    target_chunks: int,
) -> Dict[str, Any]:
    """CacheBlend suitability check for the converted DPR NQ split."""

    chunk_counts = [len(r["ctxs"]) for r in rows]
    chunk_char_lens = [len(c["text"]) for r in rows for c in r["ctxs"]]
    avg_char = sum(chunk_char_lens) / max(1, len(chunk_char_lens))
    avg_tokens = avg_char / 4.0
    min_tokens = min(chunk_char_lens) / 4.0

    # Gold preservation:
    #   (a) explicit:  count queries where any of the kept top-k ctxs had DPR's
    #       `has_answer == True` flag (most precise, no false negatives from
    #       paraphrased answers like in MuSiQue).
    #   (b) substring: any of the gold answer strings appears in any kept ctx.
    has_answer_kept = 0
    substring_kept = 0
    has_answer_in_source = 0
    for src, ext in zip(source_rows, rows):
        # (a) using DPR's flag, restricted to the top-k we kept
        src_flags = [c.get("has_answer", False) for c in src["ctxs"][: len(ext["ctxs"])]]
        if any(src_flags):
            has_answer_kept += 1
        if any(c.get("has_answer", False) for c in src["ctxs"]):
            has_answer_in_source += 1
        # (b) substring
        answers = _flatten_answers(ext["answers"])
        text = "\n".join(c["text"] for c in ext["ctxs"]).lower()
        if any(a.lower() in text for a in answers):
            substring_kept += 1

    n = len(rows)
    has_answer_ratio = has_answer_kept / max(1, n)
    substring_ratio = substring_kept / max(1, n)
    has_answer_vs_full = has_answer_kept / max(1, has_answer_in_source)

    checks = {
        "chunks_per_query_ge_10": min(chunk_counts) >= 10,
        "chunks_per_query_uniform_eq_target": (
            min(chunk_counts) == max(chunk_counts) == target_chunks
        ),
        # We're more permissive than the MuSiQue extension: DPR's top-k is a
        # noisy retrieval, so we set the floor at 0.50 on the explicit flag,
        # which is the standard NQ@k metric.
        "has_answer_in_top_k_ge_0.50": has_answer_ratio >= 0.50,
        "min_chunk_tokens_ge_50": min_tokens >= 50,
    }
    notes = [
        "single-hop open-domain QA (NQ-Open); harder for full_reuse than for cacheblend because the answer is often in 1 of 20+ chunks",
        "ctxs ordered by DPR retriever score (top-k retained)",
        "chunk size ~130 tokens (DPR 100-word convention); shorter than the paper's 512-token chunks, but the CacheBlend algorithm is chunk-size agnostic",
        f"top-k recall against full top-100 = {has_answer_vs_full:.2%} (top-{target_chunks} retains this fraction of queries that had a gold passage somewhere in the full 100-list)",
    ]
    return {
        "dataset": name,
        "n_examples": n,
        "chunks_per_query_min": min(chunk_counts),
        "chunks_per_query_max": max(chunk_counts),
        "chunks_per_query_target": target_chunks,
        "avg_chunk_chars": round(avg_char, 1),
        "avg_chunk_tokens_estimated": round(avg_tokens, 1),
        "min_chunk_tokens_estimated": round(min_tokens, 1),
        "gold_has_answer_in_top_k": has_answer_kept,
        "gold_has_answer_in_top_k_ratio": round(has_answer_ratio, 4),
        "gold_substring_in_top_k": substring_kept,
        "gold_substring_in_top_k_ratio": round(substring_ratio, 4),
        "cacheblend_suitability": {
            "checks": checks,
            "verdict_pass": all(checks.values()),
            "notes": notes,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="number of queries to keep")
    p.add_argument("--k", type=int, default=20, help="ctxs per query")
    p.add_argument("--cache-dir", default="/tmp", help="where to cache the .gz download")
    p.add_argument("--out", default=str(DATA_DIR / "nq_dpr.json"))
    p.add_argument("--report", default=str(DATA_DIR / "nq_dpr_report.json"))
    args = p.parse_args()

    if args.k < 10:
        sys.exit(f"--k must be >= 10 (CacheBlend regime), got {args.k}")

    cache_path = pathlib.Path(args.cache_dir) / "nq-test.json.gz"
    _download(DPR_URL, cache_path)
    rows = _load_dpr(cache_path)
    print(f"[load] {len(rows)} DPR NQ-test queries (100 ctxs each)")

    converted = convert(rows, args.n, args.k)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[write] {len(converted)} examples -> {out_path}")

    report = validate("nq_dpr", converted, rows[: args.n], args.k)
    report["source"] = DPR_URL
    report["k"] = args.k
    report["n"] = args.n

    pathlib.Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[write] report -> {args.report}")
    print()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["cacheblend_suitability"]["verdict_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
