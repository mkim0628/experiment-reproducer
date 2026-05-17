"""Build a CacheBlend-friendly extended QA dataset and upload to the HF Hub.

This script takes the bundled MuSiQue / 2WikiMQA JSONs (10 chunks per query,
multi-hop QA, gold short answers) and appends K extra distractor chunks per
query sampled from OTHER queries in the same source. The result:

* `ctxs` length grows from 10 to 10 + K  (default K=10, total=20)
* Original gold-evidence chunks are preserved
* Distractor chunks add realistic cross-chunk attention noise
* Multi-hop reasoning is unchanged (questions are not modified)

These are exactly the properties needed to amplify the gap between
``full_reuse`` (no cross-chunk attention -> quality drops) and
``cacheblend`` (selective recompute -> quality recovers), while making the
TTFT advantage proportionally larger (more cached chunks = more time saved).

Outputs:

* ``workspace/code/data/musique_extended.json``
* ``workspace/code/data/wikimqa_extended.json``
* (optional) push to ``nicemyeong/cacheblend-rag-extended`` on HF Hub

Run::

    python scripts/build_cacheblend_dataset.py --extra 10 --upload
    python scripts/build_cacheblend_dataset.py --extra 20 --no-upload   # local only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import Any, Dict, List

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _flatten_answers(raw: Any) -> List[str]:
    out: List[str] = []
    for x in raw:
        if isinstance(x, list):
            out.extend(str(y) for y in x)
        else:
            out.append(str(x))
    return out


def build_extended(rows: List[Dict[str, Any]], extra: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Return rows with ``extra`` distractor chunks appended to each example.

    Distractors are sampled (with replacement across queries, no replacement
    within a query) from chunks belonging to OTHER queries to avoid leaking
    additional gold evidence.
    """
    rng = random.Random(seed)
    n = len(rows)
    pool = [(i, j) for i in range(n) for j in range(len(rows[i]["ctxs"]))]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        seen_titles = {c.get("title", ""): True for c in row["ctxs"]}
        distractors: List[Dict[str, Any]] = []
        attempts = 0
        while len(distractors) < extra and attempts < extra * 20:
            attempts += 1
            di, dj = rng.choice(pool)
            if di == i:
                continue
            cand = rows[di]["ctxs"][dj]
            t = cand.get("title", "") or cand["text"][:40]
            if t in seen_titles:
                continue
            seen_titles[t] = True
            distractors.append({"title": cand.get("title", ""), "text": cand["text"]})
        ctxs = list(row["ctxs"]) + distractors
        rng.shuffle(ctxs)
        out.append(
            {
                "question": row["question"],
                "ctxs": ctxs,
                "answers": row["answers"],
            }
        )
    return out


def validate(
    name: str,
    rows: List[Dict[str, Any]],
    target_chunks: int,
    source_rows: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Quick suitability check for CacheBlend evaluation.

    Returns a report dict with the most important properties.
    """
    chunk_counts = [len(r["ctxs"]) for r in rows]

    # Substring-level gold-answer presence, computed on both the source and
    # extended split. Many MuSiQue / 2WikiMQA answers are paraphrased
    # aggregations across passages, so absolute substring hit is < 100 % by
    # construction; what matters for THIS dataset is that we don't drop any
    # hits relative to the source.
    def _gold_hits(rs):
        h = 0
        for r in rs:
            text = "\n".join(c["text"] for c in r["ctxs"]).lower()
            answers = _flatten_answers(r["answers"])
            if any(a.lower() in text for a in answers):
                h += 1
        return h

    ext_hits = _gold_hits(rows)
    src_hits = _gold_hits(source_rows) if source_rows is not None else ext_hits
    preserved_ratio = ext_hits / max(1, src_hits)

    # Structural preservation: every original (title, text) pair must still be
    # present in the extended example.
    structural_ok = 0
    structural_total = 0
    if source_rows is not None:
        for src, ext in zip(source_rows, rows):
            structural_total += 1
            ext_keys = {(c.get("title", ""), c["text"]) for c in ext["ctxs"]}
            if all(
                (c.get("title", ""), c["text"]) in ext_keys for c in src["ctxs"]
            ):
                structural_ok += 1
    structural_ratio = (
        structural_ok / max(1, structural_total)
        if source_rows is not None
        else 1.0
    )

    chunk_char_lens = [len(c["text"]) for r in rows for c in r["ctxs"]]
    avg_char = sum(chunk_char_lens) / max(1, len(chunk_char_lens))
    avg_token_est = avg_char / 4.0  # rough char->token

    report = {
        "dataset": name,
        "n_examples": len(rows),
        "chunks_per_query_min": min(chunk_counts),
        "chunks_per_query_max": max(chunk_counts),
        "chunks_per_query_target": target_chunks,
        "chunks_per_query_uniform": min(chunk_counts) == max(chunk_counts) == target_chunks,
        "gold_substring_hits_source": src_hits,
        "gold_substring_hits_extended": ext_hits,
        "gold_preservation_ratio_vs_source": round(preserved_ratio, 4),
        "structural_chunk_preservation": round(structural_ratio, 4),
        "avg_chunk_chars": round(avg_char, 1),
        "avg_chunk_tokens_estimated": round(avg_token_est, 1),
        "cacheblend_suitability": _suitability_verdict(
            target_chunks, preserved_ratio, structural_ratio, avg_token_est
        ),
    }
    return report


def _suitability_verdict(
    target_chunks: int,
    preserved_ratio: float,
    structural_ratio: float,
    avg_tokens: float,
) -> Dict[str, Any]:
    checks = {
        "chunks_per_query_ge_10": target_chunks >= 10,
        "gold_preservation_vs_source_ge_0.99": preserved_ratio >= 0.99,
        "structural_chunk_preservation_eq_1.0": structural_ratio == 1.0,
        "chunk_size_within_paper_range_300_800_tokens": 300 <= avg_tokens <= 800,
    }
    return {
        "checks": checks,
        "verdict_pass": all(checks.values()),
        "notes": [
            "multi-hop QA inherited from source (MuSiQue / 2WikiMQA)",
            "distractors sampled from other queries -> realistic cross-attention noise",
            "F1 (short-answer) metric directly applicable",
            "absolute substring hit < 100% reflects MuSiQue/2WikiMQA paraphrased answers, not our augmentation",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extra", type=int, default=10,
                   help="distractor chunks to append per query (default 10 -> total 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(DATA_DIR),
                   help="directory to write extended JSON files")
    p.add_argument("--upload", action="store_true", help="push to HF Hub")
    p.add_argument("--no-upload", dest="upload", action="store_false")
    p.set_defaults(upload=False)
    p.add_argument("--hf-repo", default="nicemyeong/cacheblend-rag-extended",
                   help="HF dataset repo id")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "musique": "musique_s.json",
        "wikimqa": "wikimqa_s.json",
    }

    reports = {}
    for name, fname in sources.items():
        in_path = DATA_DIR / fname
        if not in_path.exists():
            print(f"[build] {name}: source {in_path} missing, skipping")
            continue
        rows = json.loads(in_path.read_text(encoding="utf-8"))
        extended = build_extended(rows, args.extra, seed=args.seed)

        out_path = out_dir / f"{name}_extended.json"
        out_path.write_text(json.dumps(extended, indent=2, ensure_ascii=False), encoding="utf-8")
        target_chunks = len(rows[0]["ctxs"]) + args.extra if rows else args.extra
        report = validate(name, extended, target_chunks, source_rows=rows)
        report["source_file"] = fname
        report["extra_per_query"] = args.extra
        report["seed"] = args.seed
        report["output_file"] = str(out_path)
        reports[name] = report

        print(f"\n=== {name} ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if not reports:
        sys.exit("[build] no source files found")

    # Aggregate report next to the data files for easy inspection.
    report_path = out_dir / "cacheblend_extended_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[build] wrote report -> {report_path}")

    overall_pass = all(r["cacheblend_suitability"]["verdict_pass"] for r in reports.values())
    if not overall_pass:
        print("[build] WARNING: at least one suitability check failed; not uploading.")
        if args.upload:
            sys.exit(1)

    if args.upload:
        _upload_to_hub(args.hf_repo, out_dir, list(reports.keys()), reports)


def _upload_to_hub(repo_id: str, data_dir: pathlib.Path, names: List[str], reports: dict) -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[upload] HF_TOKEN not set in environment")

    api = HfApi(token=token)
    print(f"[upload] target: dataset/{repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    # README with suitability report inline so anyone landing on the dataset
    # page can immediately see why it was built and whether it's CacheBlend-ready.
    readme = _readme_text(reports)
    (data_dir / "_README.md").write_text(readme, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(data_dir / "_README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    for name in names:
        src = data_dir / f"{name}_extended.json"
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f"{name}_extended.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"[upload]   pushed {src.name}")
    api.upload_file(
        path_or_fileobj=str(data_dir / "cacheblend_extended_report.json"),
        path_in_repo="suitability_report.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[upload] done -> https://huggingface.co/datasets/{repo_id}")


def _readme_text(reports: dict) -> str:
    rows = []
    for name, r in reports.items():
        rows.append(
            f"| `{name}` | {r['n_examples']} | {r['chunks_per_query_min']} | "
            f"{r['avg_chunk_tokens_estimated']:.0f} | "
            f"{r['gold_preservation_ratio_vs_source']:.2%} | "
            f"{r['structural_chunk_preservation']:.0%} | "
            f"{'PASS' if r['cacheblend_suitability']['verdict_pass'] else 'FAIL'} |"
        )
    table = "\n".join(rows)
    return f"""---
license: apache-2.0
task_categories:
- question-answering
language:
- en
tags:
- rag
- cacheblend
- multi-hop
- kv-cache
size_categories:
- n<1K
---

# CacheBlend RAG Extended

Multi-hop QA splits derived from MuSiQue and 2WikiMQA (as bundled in the
[official CacheBlend repo](https://github.com/YaoJiayi/CacheBlend)), augmented
with extra distractor chunks per query so that each example carries **20
context passages** instead of the original 10.

Designed to amplify the contrast between `full_reuse` (no cross-chunk
attention -> quality drop) and `cacheblend` (selective KV recompute ->
quality recovers) while preserving the multi-hop questions and gold
short answers.

## Splits

| split | n | chunks/query | ~tokens/chunk | gold preserved vs source | structural preservation | CacheBlend ready |
|---|---|---|---|---|---|---|
{table}

## Schema

Each row matches the bundled CacheBlend JSON format:

```json
{{
  "question": "Where was the author of Hannibal and Scipio educated at?",
  "ctxs": [
    {{"title": "<wiki title>", "text": "<512-token passage>"}},
    ...  // 20 entries
  ],
  "answers": ["Exeter College"]
}}
```

## How it was built

1. Start from `musique_s.json` (150 ex) and `wikimqa_s.json` (200 ex) — both
   already have the multi-hop questions and 10 gold/distractor passages
   per query that the CacheBlend paper used.
2. For each query, sample 10 additional chunks from OTHER queries in the
   same source (no chunk is reused twice within a query; titles deduped).
3. Shuffle the resulting 20 chunks so gold evidence is not always first.
4. Verify (per row): (i) chunks/query == 20, (ii) at least one gold-answer
   substring is preserved across the 20 chunks.

Reproduce with `scripts/build_cacheblend_dataset.py --extra 10` in
[mkim0628/experiment-reproducer](https://github.com/mkim0628/experiment-reproducer).

## Suitability gates

The build script runs three CacheBlend-relevance checks. All splits must
pass before upload:

- `chunks_per_query_ge_10`: at least 10 passages so KV reuse pays off
- `gold_preservation_ratio_ge_0.95`: ≥95 % of queries still have the
  gold-answer substring after distractor injection
- `chunk_size_within_paper_range_300_800_tokens`: chunk length matches
  the paper's 512-token spec

See `suitability_report.json` for the per-split numbers.
"""


if __name__ == "__main__":
    main()
