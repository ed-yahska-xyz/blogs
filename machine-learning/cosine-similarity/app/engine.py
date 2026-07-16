#!/usr/bin/env python3
"""Shared embedding + reranking engine (use-case agnostic).

The pieces both pipelines build on:

* Model loading (cached): BGE-M3 embedder and BGE-reranker-v2-m3 cross-encoder.
* ``embed_texts``   — area-batched BGE-M3 encoder returning raw float32 vectors.
* ``rerank_scores`` — area-batched cross-encoder relevance scoring.
* small helpers: ``unit_normalize``, ``db_path_for_jsonl``.

Both ``commit_index.py`` (commit-shaped records) and ``text_search.py`` (plain
text) import from here, so there is one copy of the embedding/rerank logic and a
single model download per process.

Models
------
* Embeddings : ``BAAI/bge-m3``            (dense, 1024-dim, 8192-token context)
* Reranker   : ``BAAI/bge-reranker-v2-m3`` (cross-encoder)

BGE-M3 formatting note
----------------------
BGE-M3 does **not** use an instruction / query prefix. Queries and documents are
encoded the *same* way — no ``"Represent this sentence..."`` or ``"query:"``.
(This differs from the bge-en-v1.5 family; getting it wrong hurts retrieval.)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

import numpy as np

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 8192  # BGE-M3's full context — avoid truncating long inputs.


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
# Embedding
# --------------------------------------------------------------------------- #
# Transformer self-attention materializes a (batch x heads x seq x seq) score
# tensor, so memory scales with batch_size * seq_len**2 — NOT linearly in
# batch_size. A flat batch_size of 32 over BGE-M3's 8192-token context would pad
# a batch of long inputs to 8192 and try to allocate ~128 GiB. To keep the full
# 8192 capacity (no truncation) while staying within device memory, we cap the
# per-batch "attention area" = (#docs in batch) * (padded length)**2. 7e7
# tokens^2 ~= a single 8192-token doc alone (8192**2 = 6.7e7), or hundreds of
# short docs together — the win, since most docs are tiny.
_ATTENTION_AREA_BUDGET = 7_000_000


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Batch-embed raw texts with BGE-M3 -> RAW float32 matrix (N, 1024).

    Format-agnostic core shared by every pipeline. Batches are sized by a
    token-area budget (see ``_ATTENTION_AREA_BUDGET``) so short texts pack into
    large batches while the rare long text gets a small batch — the full
    8192-token context is preserved without blowing up attention memory.
    ``batch_size`` upper-bounds docs/batch. Vectors are NOT normalized (callers
    normalize once at load time so the cosine formula stays honest).
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model = get_embedder()
    lengths = _token_lengths(model, texts)

    # Process longest-first so the heaviest (smallest) batches run while memory
    # is freshest; results are scattered back to original order at the end.
    order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
    out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)

    for batch_idx in _area_batches(
        [lengths[i] for i in order], batch_size, _ATTENTION_AREA_BUDGET
    ):
        original = [order[j] for j in batch_idx]
        vecs = model.encode(
            [texts[i] for i in original],
            batch_size=len(original),
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out[original] = np.asarray(vecs, dtype=np.float32)

    return out


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
def rerank_scores(query: str, docs: Sequence[str]) -> List[float]:
    """Cross-encoder relevance score for each (query, doc) pair (higher = better).

    Format-agnostic core shared by every pipeline. Returns scores aligned with
    ``docs`` (NOT sorted). Uses the same attention-area batching as embedding:
    the cross-encoder pads each batch to its longest pair, so a single batch of
    long docs over the 8192 context would blow up memory. Length is dominated by
    the doc, so we batch on doc length.
    """
    if not docs:
        return []
    reranker = get_reranker()
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
    return scores


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
# Helpers
# --------------------------------------------------------------------------- #
def unit_normalize(matrix: np.ndarray) -> np.ndarray:
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
