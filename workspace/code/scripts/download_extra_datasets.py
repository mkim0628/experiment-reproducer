"""Fetch HotpotQA, MultiHop-RAG, HoVer and convert to the standard schema.

Output schema (matches workspace/code/data/wikimqa_s.json):

    [
      {
        "question": str,            # for HoVer this is the CLAIM
        "ctxs": [{"title": str, "text": str}, ...],
        "answers": [str, ...]       # for HoVer this is [LABEL]
      }, ...
    ]

Run from the repository root:

    python workspace/code/scripts/download_extra_datasets.py --which all

Requires internet. Caches Wikipedia abstracts locally to
``workspace/code/data/_wiki_cache.json`` to avoid re-fetching across runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WIKI_CACHE = DATA_DIR / "_wiki_cache.json"


# --------------------------------------------------------- HotpotQA
def fetch_hotpotqa(out_path: Path, n: int = 200, seed: int = 0) -> None:
    """HotpotQA distractor validation split via HuggingFace datasets.

    HF's hotpot_qa is typically already shuffled, but we apply a seeded
    shuffle anyway to be defensive against any upstream ordering bias
    (e.g. by question_type bridge vs. comparison).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("`pip install datasets` required for HotpotQA") from e

    print(f"[hotpotqa] loading hotpot_qa/distractor (validation) ...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    out: List[dict] = []
    for ex in ds.select(range(n)):
        ctx = ex["context"]
        ctxs = []
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            ctxs.append({"title": title, "text": "".join(sents)})
        out.append(
            {
                "question": ex["question"],
                "ctxs": ctxs,
                "answers": [ex["answer"]],
                "level": ex.get("level"),
                "type": ex.get("type"),
            }
        )
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[hotpotqa] wrote {len(out)} examples -> {out_path} (seed={seed})")


# --------------------------------------------------------- MultiHop-RAG
# Notes on sources:
# - The official `yixuantt/MultiHop-RAG` GitHub repo stores the dataset JSON
#   files via Git LFS. The raw.githubusercontent.com URL therefore returns
#   only a ~132-byte LFS pointer text, NOT the actual JSON. We must go
#   through the HuggingFace mirror `yixuantt/MultiHopRAG` (which serves
#   the real files), or pull from the LFS smudge endpoint via git-lfs.
HF_MULTIHOP_REPO = "yixuantt/MultiHopRAG"
HF_MULTIHOP_FILES = ("MultiHopRAG.json", "dataset/MultiHopRAG.json")


def _hf_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Wrap huggingface_hub.hf_hub_download with a clear error message."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "`pip install huggingface_hub` required for MultiHop-RAG / HoVer"
        ) from e
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def fetch_multihop_rag(out_path: Path, n: int = 200, seed: int = 0) -> None:
    """MultiHop-RAG: pull the canonical JSON from the HF mirror, then
    stratified-sample across ``question_type`` so each category
    (inference, comparison, temporal, null) is represented.

    The GitHub repo stores the file via Git LFS, so the raw URL is unusable
    (returns a 132-byte pointer). We try the HF mirror first; if that fails,
    fall back to ``datasets.load_dataset``.
    """
    import random
    from collections import OrderedDict, defaultdict

    data = None
    last_err = None
    for fname in HF_MULTIHOP_FILES:
        try:
            print(f"[multihop_rag] hf_hub_download {HF_MULTIHOP_REPO}:{fname}")
            local = _hf_download(HF_MULTIHOP_REPO, fname)
            data = json.loads(Path(local).read_text(encoding="utf-8"))
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            print(f"[multihop_rag]   failed: {e!r}")

    if data is None:
        try:
            print(f"[multihop_rag] load_dataset({HF_MULTIHOP_REPO}) ...")
            from datasets import load_dataset

            ds = load_dataset(HF_MULTIHOP_REPO, split="train")
            data = list(ds)
        except Exception as e:  # pragma: no cover
            last_err = e

    if data is None:
        raise SystemExit(
            "Could not fetch MultiHop-RAG. The GitHub raw URL serves only "
            "the LFS pointer; you need either huggingface_hub access to "
            f"'{HF_MULTIHOP_REPO}' or a manually-downloaded MultiHopRAG.json "
            f"placed at workspace/code/data/multihop_rag.json. Last error: {last_err!r}"
        )

    # Stratified sample by question_type so all categories are represented.
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for ex in data:
        by_type[ex.get("question_type", "unknown")].append(ex)
    rng = random.Random(seed)
    for t in by_type:
        rng.shuffle(by_type[t])
    types = sorted(by_type.keys())
    if not types:
        raise SystemExit("MultiHop-RAG JSON parsed but produced 0 records")
    per_type = max(1, n // len(types))
    pool: List[dict] = []
    for t in types:
        pool.extend(by_type[t][:per_type])
    rng.shuffle(pool)
    pool = pool[:n]
    cat_counts = {t: sum(1 for x in pool if x.get("question_type") == t) for t in types}
    print(f"[multihop_rag] stratified sample by question_type: {cat_counts} (seed={seed})")

    out: List[dict] = []
    for ex in pool:
        by_title: "OrderedDict[str, List[str]]" = OrderedDict()
        for ev in ex.get("evidence_list", []):
            title = ev.get("title") or ev.get("source") or ev.get("url") or ""
            fact = ev.get("fact", "")
            by_title.setdefault(title, []).append(fact)
        ctxs = [{"title": t or "Article", "text": "\n".join(fs)} for t, fs in by_title.items()]
        if not ctxs:
            continue
        out.append(
            {
                "question": ex.get("query") or ex.get("question") or "",
                "ctxs": ctxs,
                "answers": [str(ex.get("answer", ""))],
                "question_type": ex.get("question_type", ""),
            }
        )
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[multihop_rag] wrote {len(out)} examples -> {out_path}")


# --------------------------------------------------------- HoVer
def _load_wiki_cache() -> Dict[str, str]:
    if WIKI_CACHE.exists():
        try:
            return json.loads(WIKI_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_wiki_cache(cache: Dict[str, str]) -> None:
    WIKI_CACHE.write_text(json.dumps(cache, ensure_ascii=False))


def _fetch_wiki_abstract(title: str, cache: Dict[str, str], throttle: float) -> str:
    if title in cache:
        return cache[title]
    import urllib.parse
    import urllib.request

    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title.replace(" ", "_"), safe="")
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "cacheblend-repro/0.1 (research)"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            text = data.get("extract", "") or ""
    except Exception:
        text = ""
    cache[title] = text
    if throttle > 0:
        time.sleep(throttle)
    return text


HOVER_DEV_URL = (
    "https://raw.githubusercontent.com/hover-nlp/hover/main/"
    "data/hover/hover_dev_release_v1.1.json"
)


def fetch_hover(out_path: Path, n: int = 200, throttle: float = 0.05, seed: int = 0) -> None:
    """HoVer dev split (from the official GitHub repo, NOT Git LFS) +
    Wikipedia REST abstracts for the supporting articles.

    The HF dataset ``hover`` was deprecated when HF migrated top-level
    namespaces to org-scoped paths; the official JSON release on
    github.com/hover-nlp/hover is the canonical source and is served as a
    plain file (no LFS). We download it directly.

    IMPORTANT: the dev JSON is sorted by label -- the first 2000 records
    are all SUPPORTED, the last 2000 are all NOT_SUPPORTED. Taking the
    leading n rows produces a 100% SUPPORTED subset. We do a STRATIFIED
    sample (n/2 from each label, then deterministic shuffle) so the
    resulting JSON has a balanced label distribution regardless of n.

    Schema of each record (per github.com/hover-nlp/hover):
        uid, claim, supporting_facts (list of [title, sent_id]),
        label ("SUPPORTED" | "NOT_SUPPORTED"), num_hops, hpqa_id
    """
    import random
    import urllib.request

    print(f"[hover] GET {HOVER_DEV_URL}")
    try:
        req = urllib.request.Request(
            HOVER_DEV_URL, headers={"User-Agent": "cacheblend-repro/0.1 (research)"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            ds = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise SystemExit(
            f"could not download HoVer dev JSON from {HOVER_DEV_URL}: {e!r}"
        )

    # Stratified sample: n/2 SUPPORTED + n/2 NOT_SUPPORTED, then shuffle.
    by_label: Dict[str, List[dict]] = {"SUPPORTED": [], "NOT_SUPPORTED": []}
    for ex in ds:
        lbl = ex.get("label", "")
        if lbl in by_label:
            by_label[lbl].append(ex)
    rng = random.Random(seed)
    for lbl in by_label:
        rng.shuffle(by_label[lbl])
    half = max(1, n // 2)
    pool = (by_label["SUPPORTED"][:half] + by_label["NOT_SUPPORTED"][n - half:][:n - half]) \
        if n - half > 0 else by_label["SUPPORTED"][:half]
    # Cleaner expression: take half of each, then interleave-shuffle.
    pool = by_label["SUPPORTED"][:half] + by_label["NOT_SUPPORTED"][:n - half]
    rng.shuffle(pool)
    print(
        f"[hover] stratified sample: SUPPORTED={sum(1 for x in pool if x['label']=='SUPPORTED')}, "
        f"NOT_SUPPORTED={sum(1 for x in pool if x['label']=='NOT_SUPPORTED')} "
        f"(seed={seed})"
    )

    cache = _load_wiki_cache()
    out: List[dict] = []
    for i, ex in enumerate(pool):
        sf = ex.get("supporting_facts", [])
        titles: List[str] = sorted(
            {(x["key"] if isinstance(x, dict) else x[0]) for x in sf}
        )
        ctxs = []
        for t in titles:
            text = _fetch_wiki_abstract(t, cache, throttle)
            if text:
                ctxs.append({"title": t, "text": text})
        if not ctxs:
            continue
        label = ex.get("label", "")
        out.append(
            {
                "question": ex["claim"],
                "ctxs": ctxs,
                "answers": [str(label).upper()],
                "num_hops": ex.get("num_hops"),
                "uid": ex.get("uid"),
            }
        )
        if (i + 1) % 25 == 0:
            _save_wiki_cache(cache)
            print(f"[hover]   {i+1}/{len(pool)} (kept {len(out)}) ...")
    _save_wiki_cache(cache)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(
        f"[hover] wrote {len(out)} examples -> {out_path} "
        f"(wiki cache size {len(cache)})"
    )


# --------------------------------------------------------- entry point
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--which",
        choices=("hotpotqa", "multihop_rag", "hover", "all"),
        default="all",
    )
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="seed for stratified sampling")
    p.add_argument("--throttle", type=float, default=0.05, help="HoVer wiki API throttle (seconds)")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.which in ("hotpotqa", "all"):
        fetch_hotpotqa(DATA_DIR / "hotpotqa.json", n=args.n, seed=args.seed)
    if args.which in ("multihop_rag", "all"):
        fetch_multihop_rag(DATA_DIR / "multihop_rag.json", n=args.n, seed=args.seed)
    if args.which in ("hover", "all"):
        fetch_hover(DATA_DIR / "hover.json", n=args.n, throttle=args.throttle, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
