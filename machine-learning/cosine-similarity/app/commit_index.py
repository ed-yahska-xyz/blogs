#!/usr/bin/env python3
"""Embedding + reranking stage of the commit-search pipeline.

Consumes commit entries shaped exactly like ``fetch_commits.py``'s JSONL output
(dicts with keys ``hash``, ``author``, ``date``, ``subject``, ``body``,
``files``), embeds each commit with **BGE-M3**, stores the raw vectors plus
metadata in a single SQLite file, and serves retrieval as:

    dense cosine search (top ~50 candidates)  ->  cross-encoder rerank  ->  top_n

Models
------
* Embeddings : ``BAAI/bge-m3``            (dense, 1024-dim, 8192-token context)
* Reranker   : ``BAAI/bge-reranker-v2-m3`` (cross-encoder)

BGE-M3 formatting
-----------------
BGE-M3 does **not** use an instruction / query prefix. Queries and documents are
encoded the *same* way — no ``"Represent this sentence..."`` or ``"query:"``.
(This differs from the bge-en-v1.5 family; getting it wrong hurts retrieval.)

Storage
-------
Raw (un-normalized) float32 vectors are stored as a ``vec`` BLOB, one row per
commit. At load time every BLOB is stacked into one ``(N, 1024)`` matrix and
unit-normalized exactly once, so the cosine formula stays a single matmul and
the stored vectors remain honest.

Usage example
-------------
    from commit_index import CommitIndex, load_commits_jsonl

    commits = load_commits_jsonl("commits.jsonl")
    index = CommitIndex("commits.db")
    index.build(commits)              # embed + persist (idempotent rebuild)
    index.load()                      # stack BLOBs, normalize once

    for hit in index.retrieve("fix flaky websocket reconnect", top_n=5):
        print(f"{hit.score:+.3f}  {hit.commit['hash'][:10]}  {hit.commit['subject']}")

Or run this file directly:
    python commit_index.py commits.jsonl --db commits.db \\
        --query "fix flaky websocket reconnect" --top-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 8192  # BGE-M3's full context — avoid truncating long bodies.

Commit = dict  # {hash, author, date, subject, body, files}


# --------------------------------------------------------------------------- #
# Model loading (cached so a process loads each model at most once)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_embedder() -> "SentenceTransformer":  # noqa: F821 (lazy import below)
    """Return the cached BGE-M3 SentenceTransformer with full 8192 context."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":  # noqa: F821 (lazy import below)
    """Return the cached BGE-reranker-v2-m3 cross-encoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=MAX_SEQ_LENGTH)


# --------------------------------------------------------------------------- #
# Document construction
# --------------------------------------------------------------------------- #
def commit_to_document(commit: Commit, summary: Optional[str] = None) -> str:
    """Render a commit to the single text blob that gets embedded / reranked.

    Order: optional ``summary`` (LLM diff-summary enrichment hook), then the
    subject, the body (if non-empty), and a ``Changed files:`` line listing the
    changed paths. The same text is used for embedding and for reranking so the
    two stages judge identical inputs.
    """
    parts: List[str] = []
    if summary:
        parts.append(summary.strip())
    parts.append(commit["subject"].strip())

    body = (commit.get("body") or "").strip()
    if body:
        parts.append(body)

    files = commit.get("files") or []
    if files:
        parts.append("Changed files: " + ", ".join(files))

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def embed_commit(commit: Commit, summary: Optional[str] = None) -> np.ndarray:
    """Embed one commit; return a RAW (un-normalized) float32 vector (1024,)."""
    doc = commit_to_document(commit, summary=summary)
    vec = get_embedder().encode(
        doc,
        normalize_embeddings=False,  # store raw; normalize once at load time
        convert_to_numpy=True,
    )
    return np.asarray(vec, dtype=np.float32)


# Transformer self-attention materializes a (batch x heads x seq x seq) score
# tensor, so memory scales with batch_size * seq_len**2 — NOT linearly in
# batch_size. A flat batch_size of 32 over BGE-M3's 8192-token context would
# pad a batch of long commits to 8192 and try to allocate ~128 GiB. To keep the
# full 8192 capacity (no truncation) while staying within device memory, we cap
# the per-batch "attention area" = (#docs in batch) * (padded length)**2.
# 7e7 tokens^2 ~= a single 8192-token doc alone (8192**2 = 6.7e7), or hundreds
# of short docs together — the win, since most commit docs are tiny.
_ATTENTION_AREA_BUDGET = 7_000_000


def embed_commits(
    commits: Sequence[Commit],
    batch_size: int = 32,
    summaries: Optional[Sequence[Optional[str]]] = None,
) -> np.ndarray:
    """Batch-embed a whole list of commits -> RAW float32 matrix (N, 1024).

    Batching (not a per-commit loop) is the throughput win. Batches are sized by
    a token-area budget (see ``_ATTENTION_AREA_BUDGET``) so short commits pack
    into large batches while the rare long commit gets a small batch — the full
    8192-token context is preserved without blowing up attention memory.
    ``batch_size`` is the upper bound on docs per batch.
    """
    if not commits:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    if summaries is None:
        docs = [commit_to_document(c) for c in commits]
    else:
        docs = [commit_to_document(c, s) for c, s in zip(commits, summaries)]

    model = get_embedder()
    lengths = _token_lengths(model, docs)

    # Process longest-first so the heaviest (smallest) batches run while memory
    # is freshest; results are scattered back to original order at the end.
    order = sorted(range(len(docs)), key=lambda i: lengths[i], reverse=True)
    out = np.empty((len(docs), EMBED_DIM), dtype=np.float32)

    for batch_idx in _area_batches(
        [lengths[i] for i in order], batch_size, _ATTENTION_AREA_BUDGET
    ):
        original = [order[j] for j in batch_idx]
        vecs = model.encode(
            [docs[i] for i in original],
            batch_size=len(original),
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out[original] = np.asarray(vecs, dtype=np.float32)

    return out


def _token_lengths(model: object, docs: Sequence[str]) -> List[int]:
    """Tokenized length of each doc, capped at the model's max sequence length.

    Works for both SentenceTransformer (``max_seq_length``) and CrossEncoder
    (``max_length``).
    """
    cap = getattr(model, "max_seq_length", None) or getattr(model, "max_length", MAX_SEQ_LENGTH)
    encoded = model.tokenizer(
        list(docs), truncation=True, max_length=cap, return_length=False
    )["input_ids"]
    return [len(ids) for ids in encoded]


def _area_batches(
    lengths_desc: Sequence[int], max_count: int, area_budget: int
) -> List[List[int]]:
    """Group indices (lengths sorted descending) into attention-area batches.

    Returns lists of indices into ``lengths_desc``. A batch always holds at
    least one doc, even if that single doc exceeds the budget.
    """
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i, length in enumerate(lengths_desc):
        tentative_max = max(cur_max, length)
        fits_area = (len(cur) + 1) * tentative_max * tentative_max <= area_budget
        if cur and (len(cur) >= max_count or not fits_area):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, length)
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------- #
# Retrieval results
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    """A retrieved commit with its score (cosine, or rerank logit)."""

    commit: Commit
    score: float


# --------------------------------------------------------------------------- #
# The index: SQLite storage + cosine retrieval + reranking
# --------------------------------------------------------------------------- #
class CommitIndex:
    """SQLite-backed store of commit metadata + raw embedding BLOBs.

    ``build`` embeds and persists; ``load`` stacks the BLOBs into one matrix and
    unit-normalizes it once. ``search`` / ``search_naive`` do dense cosine,
    ``rerank`` applies the cross-encoder, and ``retrieve`` runs the full flow.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS commits (
            rowid    INTEGER PRIMARY KEY,
            hash     TEXT UNIQUE,
            author   TEXT,
            date     TEXT,
            subject  TEXT,
            body     TEXT,
            files    TEXT,        -- JSON-encoded list[str]
            vec      BLOB         -- raw float32 embedding, EMBED_DIM elements
        );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # Loaded state (populated by load()):
        self._commits: List[Commit] = []
        self._matrix: Optional[np.ndarray] = None  # (N, 1024) unit-normalized

    _INSERT_SQL = """
        INSERT OR IGNORE INTO commits
            (hash, author, date, subject, body, files, vec)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    @staticmethod
    def _rows(commits: Sequence[Commit], vecs: np.ndarray) -> List[tuple]:
        """Build the INSERT row tuples for commits + their embedding vectors."""
        return [
            (
                c["hash"],
                c.get("author", ""),
                c.get("date", ""),
                c.get("subject", ""),
                c.get("body", ""),
                json.dumps(c.get("files", [])),
                vec.astype(np.float32).tobytes(),
            )
            for c, vec in zip(commits, vecs)
        ]

    # ----- build ----------------------------------------------------------- #
    def build(self, commits: Sequence[Commit], batch_size: int = 32) -> None:
        """Embed every commit and (re)write the SQLite table from scratch."""
        vecs = embed_commits(commits, batch_size=batch_size)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.execute("DELETE FROM commits;")  # idempotent full rebuild
            conn.executemany(self._INSERT_SQL, self._rows(commits, vecs))
            conn.commit()

    # ----- incremental add ------------------------------------------------- #
    def existing_hashes(self) -> set:
        """Return the set of commit hashes already stored (empty if no db)."""
        if not os.path.exists(self.db_path):
            return set()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            return {row[0] for row in conn.execute("SELECT hash FROM commits")}

    def add(self, commits: Sequence[Commit], batch_size: int = 32) -> int:
        """Embed and INSERT only commits whose hash is not already stored.

        Returns the number of newly inserted commits. Embeds nothing (and never
        loads the model) when there is no new work to do.
        """
        if not commits:
            return 0
        existing = self.existing_hashes()
        fresh = [c for c in commits if c.get("hash") not in existing]
        if not fresh:
            return 0
        vecs = embed_commits(fresh, batch_size=batch_size)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.executemany(self._INSERT_SQL, self._rows(fresh, vecs))
            conn.commit()
        return len(fresh)

    def latest_commit_date(self) -> Optional[str]:
        """ISO date of the chronologically newest stored commit, or None.

        Parses the stored ISO-8601 author dates so timezone offsets are compared
        as true instants (a lexical ``MAX(date)`` could misorder across zones).
        """
        if not os.path.exists(self.db_path):
            return None
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            dates = [row[0] for row in conn.execute("SELECT date FROM commits") if row[0]]
        if not dates:
            return None
        return max(dates, key=_parse_iso)

    # ----- load ------------------------------------------------------------ #
    def load(self) -> "CommitIndex":
        """Read every row, reconstruct the (N, 1024) matrix, normalize once."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT hash, author, date, subject, body, files, vec "
                "FROM commits ORDER BY rowid"
            ).fetchall()

        self._commits = []
        vectors: List[np.ndarray] = []
        for hash_, author, date, subject, body, files, vec in rows:
            self._commits.append(
                {
                    "hash": hash_,
                    "author": author,
                    "date": date,
                    "subject": subject,
                    "body": body,
                    "files": json.loads(files) if files else [],
                }
            )
            vectors.append(np.frombuffer(vec, dtype=np.float32))

        if vectors:
            raw = np.vstack(vectors).astype(np.float32)
            self._matrix = _unit_normalize(raw)
        else:
            self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
        return self

    def __len__(self) -> int:
        return len(self._commits)

    @property
    def is_empty(self) -> bool:
        return self._matrix is None or self._matrix.shape[0] == 0

    # ----- query embedding ------------------------------------------------- #
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed the query the SAME way as documents (no BGE-M3 prefix)."""
        vec = get_embedder().encode(
            query, normalize_embeddings=False, convert_to_numpy=True
        )
        return np.asarray(vec, dtype=np.float32)

    # ----- cosine search (the definition, spelled out) --------------------- #
    def search_naive(self, query: str, k: int = 50) -> List[Hit]:
        """Cosine similarity written out as a loop — the literal definition."""
        if self.is_empty:
            return []
        q = self._embed_query(query)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        scored: List[Hit] = []
        for i, commit in enumerate(self._commits):
            d = self._matrix[i]  # already unit-normalized at load time
            # d is unit-length, so dot(q, d) / (|q| * 1) is the cosine.
            cosine = float(np.dot(q, d) / q_norm)
            scored.append(Hit(commit, cosine))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(k, 0)]

    # ----- cosine search (vectorized to one matmul) ------------------------ #
    def search(self, query: str, k: int = 50) -> List[Hit]:
        """Same cosine, vectorized: one matmul against the normalized matrix."""
        if self.is_empty:
            return []
        q = self._embed_query(query)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q_unit = q / q_norm

        # matrix rows are unit vectors -> matrix @ q_unit == per-row cosine.
        sims = self._matrix @ q_unit  # (N,)
        k = min(max(k, 0), sims.shape[0])
        if k == 0:
            return []
        # argpartition for the top-k, then sort just those k descending.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [Hit(self._commits[i], float(sims[i])) for i in top_idx]

    # ----- reranking ------------------------------------------------------- #
    def rerank(self, query: str, candidates: Sequence[Hit], top_n: int) -> List[Hit]:
        """Cross-encoder rerank: score (query, document) pairs, top_n by score.

        Uses the same ``commit_to_document`` text the embeddings were built from
        so both stages judge identical inputs. Higher score = more relevant.
        """
        if not candidates:
            return []
        docs = [commit_to_document(h.commit) for h in candidates]
        reranker = get_reranker()

        # Same attention-area cap as embedding: the cross-encoder pads each
        # batch to its longest (query, doc) pair, so a single batch of long
        # candidates over the 8192 context would blow up memory. Length is
        # dominated by the document, so we batch on the doc's token length.
        lengths = _token_lengths(reranker, docs)
        order = sorted(range(len(docs)), key=lambda i: lengths[i], reverse=True)

        scores = [0.0] * len(docs)
        for batch_idx in _area_batches(
            [lengths[i] for i in order], 32, _ATTENTION_AREA_BUDGET
        ):
            original = [order[j] for j in batch_idx]
            batch_scores = reranker.predict(
                [[query, docs[i]] for i in original],
                batch_size=len(original),
                show_progress_bar=False,
            )
            for i, s in zip(original, batch_scores):
                scores[i] = float(s)

        reranked = [Hit(h.commit, s) for h, s in zip(candidates, scores)]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[: max(top_n, 0)]

    # ----- full flow ------------------------------------------------------- #
    def retrieve(
        self, query: str, top_n: int = 5, candidate_k: int = 50
    ) -> List[Hit]:
        """Full retrieval: dense cosine -> top candidate_k -> rerank -> top_n."""
        candidates = self.search(query, k=candidate_k)
        if not candidates:
            return []
        return self.rerank(query, candidates, top_n=top_n)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp to a tz-aware datetime for ordering.

    Naive timestamps are assumed UTC; unparseable values sort as the minimum so
    they never win a ``max()``. Always tz-aware so values are mutually comparable.
    """
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _unit_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zero rows are left as zeros (avoid div-by-zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def db_path_for_jsonl(jsonl_path: str) -> str:
    """Convention: the SQLite db mirrors the JSONL's path with a .db extension.

    ``ebay-evo-web.jsonl`` -> ``ebay-evo-web.db`` (same directory and stem), so
    the JSONL filename is the single hint that determines the db filename.
    """
    root, _ext = os.path.splitext(jsonl_path)
    return root + ".db"


def load_commits_jsonl(path: str) -> List[Commit]:
    """Load commit dicts from a JSONL file (one JSON object per line)."""
    commits: List[Commit] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                commits.append(json.loads(line))
    return commits


# --------------------------------------------------------------------------- #
# Runnable example
# --------------------------------------------------------------------------- #
class _HelpfulParser(argparse.ArgumentParser):
    """ArgumentParser that prints FULL help (not a one-line error) on misuse."""

    def error(self, message: str) -> "NoReturn":  # type: ignore[name-defined]
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help(sys.stderr)
        raise SystemExit(2)


_USAGE_EXAMPLES = """\
examples:
  # Build index from JSONL and search (writes ./commits.db)
  python commit_index.py commits.jsonl --query "fix flaky websocket reconnect"

  # Quick demo on first 200 commits, custom db + result counts
  python commit_index.py commits.jsonl --db /tmp/demo.db \\
      --query "icon rendering bug" --limit 200 --candidate-k 30 --top-n 5

  # Reuse an already-built index (no re-embedding)
  python commit_index.py commits.jsonl --db /tmp/demo.db \\
      --query "accessibility aria labels" --reuse-db
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = _HelpfulParser(
        prog="commit_index.py",
        description="Build a BGE-M3 commit index and run dense + rerank search.",
        epilog=_USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("jsonl", help="Commits JSONL (fetch_commits.py output).")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite index path (default: the JSONL path with a .db extension).",
    )
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-n", type=int, default=5, help="Final results.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Dense candidates fed to the reranker (default 50).",
    )
    parser.add_argument(
        "--reuse-db",
        action="store_true",
        help="Skip embedding; load an already-built --db.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only index the first N commits (handy for a quick demo).",
    )
    args = parser.parse_args(argv)

    db_path = args.db or db_path_for_jsonl(args.jsonl)
    index = CommitIndex(db_path)
    if not args.reuse_db:
        commits = load_commits_jsonl(args.jsonl)
        if args.limit:
            commits = commits[: args.limit]
        print(f"Embedding {len(commits)} commits with {EMBED_MODEL} -> {db_path} ...")
        index.build(commits)
    index.load()

    print(f"Index holds {len(index)} commits.\n")
    if index.is_empty:
        print("Index is empty — nothing to search.")
        return 0

    print(f"Query: {args.query!r}")
    print(f"Dense retrieval (top {args.candidate_k}) -> rerank (top {args.top_n}):\n")
    hits = index.retrieve(args.query, top_n=args.top_n, candidate_k=args.candidate_k)
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:2d}. rerank={hit.score:+.4f}  {hit.commit['hash'][:10]}")
        print(f"    {hit.commit['subject']}")
        files = hit.commit.get("files", [])
        if files:
            preview = ", ".join(files[:4]) + (" ..." if len(files) > 4 else "")
            print(f"    files: {preview}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
