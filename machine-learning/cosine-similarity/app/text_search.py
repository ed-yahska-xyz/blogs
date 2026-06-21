#!/usr/bin/env python3
"""Simple semantic search over arbitrary text documents.

A lightweight, use-case-agnostic frontend to the same BGE-M3 embedding + cosine
+ cross-encoder rerank machinery that powers the commit pipeline. Where
``commit_index.py`` is bound to the commit JSON shape, this stores plain text:
each document is just an ``id`` and its ``text``.

It reuses the shared, format-agnostic cores from ``commit_index`` — ``embed_texts``
(area-batched BGE-M3 encoder) and ``rerank_scores`` (area-batched cross-encoder)
— so there is no duplicated embedding logic and no separate model download.

Storage
-------
One SQLite file, one row per document: ``id`` (content hash or caller-supplied),
the raw ``text``, and the raw float32 embedding as a ``vec`` BLOB. On load the
BLOBs are stacked into one ``(N, 1024)`` matrix and unit-normalized once.

Input
-----
* A plain text file: each non-empty line is one document (ids are content
  hashes, so re-indexing the same lines is idempotent).
* A ``.jsonl`` file: one JSON object per line with a ``"text"`` field and an
  optional ``"id"`` (and any other keys, which are ignored).

CLI
---
    # Build an index from a text file (one document per line) -> notes.db
    python text_search.py index notes.txt

    # Add more documents later (incremental; skips duplicates)
    python text_search.py add more.txt --db notes.db

    # Search (dense cosine -> rerank). Use --no-rerank for pure nearest-neighbor.
    python text_search.py search "how do refunds work" --db notes.db --top-n 5

Library
-------
    from text_search import TextIndex
    idx = TextIndex("notes.db")
    idx.build(["first note", "second note"])
    idx.load()
    for hit in idx.retrieve("note about X", top_n=3):
        print(hit.score, hit.id, hit.text[:80])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# Reuse the shared, format-agnostic cores — no duplicated embedding/rerank code.
from commit_index import (
    EMBED_DIM,
    db_path_for_jsonl,
    embed_texts,
    rerank_scores,
    _unit_normalize,
)

# A document is either a bare string or a {"id"?: str, "text": str} mapping.
Doc = Union[str, Dict[str, str]]


# --------------------------------------------------------------------------- #
# Document normalization
# --------------------------------------------------------------------------- #
def _doc_id(text: str) -> str:
    """Stable content-addressed id so re-indexing the same text is idempotent."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_doc(doc: Doc) -> Dict[str, str]:
    """Coerce a string or mapping into a ``{"id", "text"}`` record."""
    if isinstance(doc, str):
        text = doc
        doc_id = _doc_id(text)
    else:
        text = doc["text"]
        doc_id = doc.get("id") or _doc_id(text)
    return {"id": doc_id, "text": text}


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class TextHit:
    """A retrieved document with its score (cosine, or rerank logit)."""

    id: str
    text: str
    score: float


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
class TextIndex:
    """SQLite store of text documents + raw embeddings, with cosine + rerank."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS docs (
            rowid INTEGER PRIMARY KEY,
            id    TEXT UNIQUE,
            text  TEXT,
            vec   BLOB           -- raw float32 embedding, EMBED_DIM elements
        );
    """

    _INSERT_SQL = "INSERT OR IGNORE INTO docs (id, text, vec) VALUES (?, ?, ?)"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._docs: List[Dict[str, str]] = []
        self._matrix: Optional[np.ndarray] = None  # (N, 1024) unit-normalized

    @staticmethod
    def _rows(docs: Sequence[Dict[str, str]], vecs: np.ndarray) -> List[tuple]:
        return [
            (d["id"], d["text"], vec.astype(np.float32).tobytes())
            for d, vec in zip(docs, vecs)
        ]

    # ----- build / add ----------------------------------------------------- #
    def build(self, docs: Sequence[Doc], batch_size: int = 32) -> int:
        """Embed every document and (re)write the table from scratch."""
        records = [normalize_doc(d) for d in docs]
        vecs = embed_texts([r["text"] for r in records], batch_size=batch_size)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.execute("DELETE FROM docs;")
            conn.executemany(self._INSERT_SQL, self._rows(records, vecs))
            conn.commit()
        return len(records)

    def existing_ids(self) -> set:
        if not os.path.exists(self.db_path):
            return set()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            return {row[0] for row in conn.execute("SELECT id FROM docs")}

    def add(self, docs: Sequence[Doc], batch_size: int = 32) -> int:
        """Embed and INSERT only documents whose id is not already stored."""
        records = [normalize_doc(d) for d in docs]
        existing = self.existing_ids()
        # Dedup within the batch too, so repeated lines don't collide on INSERT.
        seen = set(existing)
        fresh: List[Dict[str, str]] = []
        for r in records:
            if r["id"] not in seen:
                seen.add(r["id"])
                fresh.append(r)
        if not fresh:
            return 0
        vecs = embed_texts([r["text"] for r in fresh], batch_size=batch_size)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.executemany(self._INSERT_SQL, self._rows(fresh, vecs))
            conn.commit()
        return len(fresh)

    # ----- load ------------------------------------------------------------ #
    def load(self) -> "TextIndex":
        """Read every row, reconstruct the (N, 1024) matrix, normalize once."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            rows = conn.execute(
                "SELECT id, text, vec FROM docs ORDER BY rowid"
            ).fetchall()
        self._docs = [{"id": i, "text": t} for i, t, _ in rows]
        vectors = [np.frombuffer(v, dtype=np.float32) for _, _, v in rows]
        if vectors:
            self._matrix = _unit_normalize(np.vstack(vectors).astype(np.float32))
        else:
            self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
        return self

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def is_empty(self) -> bool:
        return self._matrix is None or self._matrix.shape[0] == 0

    # ----- retrieval ------------------------------------------------------- #
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed the query the SAME way as documents (no BGE-M3 prefix)."""
        return embed_texts([query])[0]

    def search_naive(self, query: str, k: int = 50) -> List[TextHit]:
        """Cosine similarity written out as a loop — the literal definition."""
        if self.is_empty:
            return []
        q = self._embed_query(query)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        scored: List[TextHit] = []
        for i, doc in enumerate(self._docs):
            d = self._matrix[i]  # already unit-normalized at load time
            cosine = float(np.dot(q, d) / q_norm)
            scored.append(TextHit(doc["id"], doc["text"], cosine))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(k, 0)]

    def search(self, query: str, k: int = 50) -> List[TextHit]:
        """Same cosine, vectorized: one matmul against the normalized matrix."""
        if self.is_empty:
            return []
        q = self._embed_query(query)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        sims = self._matrix @ (q / q_norm)  # rows are unit vectors -> cosine
        k = min(max(k, 0), sims.shape[0])
        if k == 0:
            return []
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            TextHit(self._docs[i]["id"], self._docs[i]["text"], float(sims[i]))
            for i in top_idx
        ]

    def rerank(
        self, query: str, candidates: Sequence[TextHit], top_n: int
    ) -> List[TextHit]:
        """Cross-encoder rerank the candidates; return the top_n by score."""
        if not candidates:
            return []
        scores = rerank_scores(query, [c.text for c in candidates])
        reranked = [TextHit(c.id, c.text, s) for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[: max(top_n, 0)]

    def retrieve(
        self,
        query: str,
        top_n: int = 5,
        candidate_k: int = 50,
        rerank: bool = True,
    ) -> List[TextHit]:
        """Full flow: dense cosine -> top candidate_k -> (optional) rerank -> top_n."""
        candidates = self.search(query, k=candidate_k if rerank else top_n)
        if not candidates:
            return []
        if not rerank:
            return candidates[:top_n]
        return self.rerank(query, candidates, top_n=top_n)


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_documents(path: str) -> List[Doc]:
    """Load documents from a ``.jsonl`` file or a one-document-per-line text file."""
    docs: List[Doc] = []
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        else:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(line)
    return docs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class _HelpfulParser(argparse.ArgumentParser):
    """ArgumentParser that prints FULL help (not a one-line error) on misuse."""

    def error(self, message: str) -> "NoReturn":  # type: ignore[name-defined]
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help(sys.stderr)
        raise SystemExit(2)


def _print_hits(query: str, hits: List[TextHit], reranked: bool) -> None:
    label = "rerank" if reranked else "cosine"
    print(f"Query: {query!r}\n")
    if not hits:
        print("(no results)")
        return
    for rank, hit in enumerate(hits, 1):
        snippet = " ".join(hit.text.split())
        if len(snippet) > 100:
            snippet = snippet[:100] + " ..."
        print(f"{rank:2d}. {label}={hit.score:+.4f}  [{hit.id}]")
        print(f"    {snippet}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _HelpfulParser(
        prog="text_search.py",
        description="Semantic search over arbitrary text documents (BGE-M3).",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Build an index from a file.")
    p_index.add_argument("file", help="Text file (one doc/line) or .jsonl.")
    p_index.add_argument(
        "--db", default=None, help="SQLite path (default: <file>.db)."
    )

    p_add = sub.add_parser("add", help="Add documents to an existing index.")
    p_add.add_argument("file", help="Text file (one doc/line) or .jsonl.")
    p_add.add_argument(
        "--db", default=None, help="SQLite path (default: <file>.db)."
    )

    p_search = sub.add_parser("search", help="Search an index.")
    p_search.add_argument("query", help="Search query text.")
    p_search.add_argument("--db", required=True, help="SQLite index path.")
    p_search.add_argument("--top-n", type=int, default=5)
    p_search.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Dense candidates fed to the reranker (default 50).",
    )
    p_search.add_argument(
        "--no-rerank",
        action="store_true",
        help="Pure cosine nearest-neighbor; skip the cross-encoder.",
    )

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help(sys.stderr)
        return 2

    if args.cmd in ("index", "add"):
        if not os.path.exists(args.file):
            print(f"error: {args.file} not found", file=sys.stderr)
            return 1
        db_path = args.db or db_path_for_jsonl(args.file)
        docs = load_documents(args.file)
        index = TextIndex(db_path)
        if args.cmd == "index":
            print(f"Indexing {len(docs)} document(s) -> {db_path} ...")
            index.build(docs)
            added = len(docs)
        else:
            print(f"Adding from {args.file} -> {db_path} ...")
            added = index.add(docs)
        index.load()
        print(f"{args.cmd}: {added} new; index now holds {len(index)} document(s).")
        return 0

    # search
    index = TextIndex(args.db).load()
    if index.is_empty:
        print("Index is empty — nothing to search.")
        return 0
    hits = index.retrieve(
        args.query,
        top_n=args.top_n,
        candidate_k=args.candidate_k,
        rerank=not args.no_rerank,
    )
    _print_hits(args.query, hits, reranked=not args.no_rerank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
