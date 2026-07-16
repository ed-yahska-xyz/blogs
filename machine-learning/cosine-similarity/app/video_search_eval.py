#!/usr/bin/env python3
"""Build a video-search index and evaluate it against labeled sample queries.

Ingests ``data/sample-videos.jsonl`` (one ``{id, title, description}`` per line)
into a fresh SQLite index using the generic ``TextIndex`` from ``text_search.py``,
then runs the labeled queries in ``data/sample-video-queries.jsonl`` and reports
recall@k. Each query lists the video ids a good system should surface, and
several share NO words with their target (e.g. "leaky faucet" -> "dripping tap")
to test semantic matching over keyword matching.

Usage
-----
    # Build data/sample-videos.db and evaluate (cosine -> cross-encoder rerank)
    python video_search_eval.py

    # Compare against pure cosine nearest-neighbor (no reranker)
    python video_search_eval.py --no-rerank

    # Custom paths / display depth
    python video_search_eval.py --videos data/sample-videos.jsonl \\
        --queries data/sample-video-queries.jsonl --db data/sample-videos.db --top-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from text_search import TextIndex


def load_videos(path: str) -> List[Dict[str, str]]:
    """Load videos and render each into a {id, text} document (title + desc)."""
    docs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            title = v.get("title", "").strip()
            desc = v.get("description", "").strip()
            text = f"{title}. {desc}".strip(". ").strip()
            docs.append({"id": v["id"], "text": text})
    return docs


def load_queries(path: str) -> List[dict]:
    """Load the labeled queries (a JSON object with a ``queries`` array)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["queries"] if isinstance(data, dict) else data


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of a query's relevant ids found within the top-k results."""
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hit = sum(1 for r in relevant_ids if r in top)
    return hit / len(relevant_ids)


def first_relevant_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> int:
    """1-based rank of the first relevant id, or 0 if none retrieved."""
    rel = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in rel:
            return rank
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a video index and evaluate labeled queries (recall@k)."
    )
    parser.add_argument("--videos", default="data/sample-videos.jsonl")
    parser.add_argument("--queries", default="data/sample-video-queries.jsonl")
    # NOTE: its own db, NOT data/sample-videos.db — build() wipes and rebuilds
    # the table, so the eval must not clobber the large free-form search corpus.
    parser.add_argument("--db", default="data/sample-videos-eval.db")
    parser.add_argument("--top-n", type=int, default=5, help="Results shown/scored.")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Pure cosine nearest-neighbor; skip the cross-encoder.",
    )
    args = parser.parse_args(argv)

    for path in (args.videos, args.queries):
        if not os.path.exists(path):
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    rerank = not args.no_rerank
    ks = [k for k in (1, 3, args.top_n) if k <= args.top_n]
    ks = sorted(set(ks))

    # 1) Build a fresh index from the video corpus.
    videos = load_videos(args.videos)
    print(f"Indexing {len(videos)} videos -> {args.db} ...")
    index = TextIndex(args.db)
    index.build(videos)
    index.load()
    text_by_id = {d["id"]: d["text"] for d in videos}

    # 2) Run each labeled query and score it.
    queries = load_queries(args.queries)
    mode = "cosine + rerank" if rerank else "cosine only (--no-rerank)"
    print(f"\nEvaluating {len(queries)} queries  [{mode}]\n" + "=" * 64)

    recall_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    for q in queries:
        hits = index.retrieve(
            q["text"], top_n=args.top_n, candidate_k=len(videos), rerank=rerank
        )
        retrieved = [h.id for h in hits]
        relevant = q.get("relevant_ids", [])
        rel_set = set(relevant)

        for k in ks:
            recall_sums[k] += recall_at_k(retrieved, relevant, k)
        rank = first_relevant_rank(retrieved, relevant)
        rr_sum += (1.0 / rank) if rank else 0.0

        print(f"\n{q['id']}: {q['text']!r}")
        print(f"   relevant: {relevant}")
        for i, h in enumerate(hits, 1):
            mark = "OK " if h.id in rel_set else "   "
            title = text_by_id.get(h.id, "").split(". ")[0]
            print(f"   {mark}{i}. {h.score:+.3f}  {h.id}  {title[:54]}")
        missed = [r for r in relevant if r not in set(retrieved[: args.top_n])]
        if missed:
            print(f"   MISSED in top-{args.top_n}: {missed}")

    # 3) Aggregate.
    n = len(queries)
    print("\n" + "=" * 64)
    print(f"Mean over {n} queries  [{mode}]:")
    for k in ks:
        print(f"   recall@{k}: {recall_sums[k] / n:.3f}")
    print(f"   MRR:       {rr_sum / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
