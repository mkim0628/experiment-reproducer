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
def fetch_hotpotqa(out_path: Path, n: int = 200) -> None:
    """HotpotQA distractor validation split via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("`pip install datasets` required for HotpotQA") from e

    print(f"[hotpotqa] loading hotpot_qa/distractor (validation) ...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    n = min(n, len(ds))
    out: List[dict] = []
    for ex in ds.select(range(n)):
        ctx = ex["context"]
        # context = {"title": [str], "sentences": [[str]]}.
        ctxs = []
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            ctxs.append({"title": title, "text": "".join(sents)})
        out.append(
            {
                "question": ex["question"],
                "ctxs": ctxs,
                "answers": [ex["answer"]],
            }
        )
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[hotpotqa] wrote {len(out)} examples -> {out_path}")


# --------------------------------------------------------- MultiHop-RAG
MULTIHOP_URLS = [
    "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json",
    # Fallback name some forks use:
    "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/multihop_rag.json",
]


def fetch_multihop_rag(out_path: Path, n: int = 200) -> None:
    """MultiHop-RAG: download the canonical JSON and reshape to our schema."""
    import urllib.request
    from collections import OrderedDict

    data = None
    for url in MULTIHOP_URLS:
        try:
            print(f"[multihop_rag] GET {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:  # pragma: no cover
            print(f"[multihop_rag]   failed: {e!r}")
    if data is None:
        raise SystemExit("could not download MultiHop-RAG JSON from any URL")

    n = min(n, len(data))
    out: List[dict] = []
    for ex in data[:n]:
        # group evidence_list facts by source article.
        by_title: "OrderedDict[str, List[str]]" = OrderedDict()
        for ev in ex.get("evidence_list", []):
            title = ev.get("title") or ev.get("source") or ev.get("url") or ""
            fact = ev.get("fact", "")
            by_title.setdefault(title, []).append(fact)
        ctxs = [{"title": t or "Article", "text": "\n".join(fs)} for t, fs in by_title.items()]
        # Skip queries with no usable evidence.
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


def fetch_hover(out_path: Path, n: int = 200, throttle: float = 0.05) -> None:
    """HoVer dev split + Wikipedia REST abstracts for supporting articles."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("`pip install datasets` required for HoVer") from e

    print(f"[hover] loading hover/validation ...")
    try:
        ds = load_dataset("hover", split="validation")
    except Exception as e:
        # Several HF mirrors exist with slightly different names.
        for cand in ("pminervini/hover", "hover-team/hover"):
            try:
                print(f"[hover]   retry: {cand}")
                ds = load_dataset(cand, split="validation")
                break
            except Exception:
                ds = None
        if ds is None:
            raise SystemExit(f"could not load HoVer dataset: {e}")

    n = min(n, len(ds))
    cache = _load_wiki_cache()
    out: List[dict] = []
    label_map_int = {0: "SUPPORTED", 1: "NOT_SUPPORTED"}
    for i, ex in enumerate(ds.select(range(n))):
        # supporting_facts can be either [{key,value}, ...] or [[title,sent_id],...]
        titles: List[str] = []
        sf = ex.get("supporting_facts", [])
        if sf and isinstance(sf[0], dict):
            titles = sorted({x["key"] for x in sf})
        elif sf and isinstance(sf[0], (list, tuple)):
            titles = sorted({x[0] for x in sf})
        # Fetch abstracts.
        ctxs = []
        for t in titles:
            text = _fetch_wiki_abstract(t, cache, throttle)
            if text:
                ctxs.append({"title": t, "text": text})
        if not ctxs:
            continue
        label = ex.get("label")
        label_str = (
            label_map_int.get(label, str(label)) if isinstance(label, int) else str(label).upper()
        )
        out.append(
            {
                "question": ex["claim"],
                "ctxs": ctxs,
                "answers": [label_str],
                "num_hops": ex.get("num_hops"),
            }
        )
        if (i + 1) % 25 == 0:
            _save_wiki_cache(cache)
            print(f"[hover]   {i+1}/{n} ...")
    _save_wiki_cache(cache)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[hover] wrote {len(out)} examples -> {out_path} (cache size {len(cache)})")


# --------------------------------------------------------- entry point
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--which",
        choices=("hotpotqa", "multihop_rag", "hover", "all"),
        default="all",
    )
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--throttle", type=float, default=0.05, help="HoVer wiki API throttle (seconds)")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.which in ("hotpotqa", "all"):
        fetch_hotpotqa(DATA_DIR / "hotpotqa.json", n=args.n)
    if args.which in ("multihop_rag", "all"):
        fetch_multihop_rag(DATA_DIR / "multihop_rag.json", n=args.n)
    if args.which in ("hover", "all"):
        fetch_hover(DATA_DIR / "hover.json", n=args.n, throttle=args.throttle)


if __name__ == "__main__":
    sys.exit(main())
