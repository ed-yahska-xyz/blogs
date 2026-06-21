# Semantic search app

A small, self-contained semantic-search app built on **BGE-M3** embeddings with
a **BGE-reranker-v2-m3** cross-encoder. It has two pipelines on top of one shared
engine:

- **Commit search** — clone a repo, extract its commits, embed them, and search
  with natural language ("which commit made the card image optional?").
- **Generic text search** — index any plain-text documents (e.g. a corpus of
  YouTube titles + descriptions) and search them.

Retrieval is always two stages: **dense cosine** retrieves a generous candidate
set, then the **cross-encoder reranks** for the final order.

---

## Layout

```
app/
├── engine.py            # shared: model loading, embed_texts, rerank_scores, helpers
├── fetch_commits.py     # commit pipeline: clone + git log -> JSONL (stdlib only)
├── commit_index.py      # commit pipeline: embed/store/search commits  (uses engine)
├── sync_commits.py      # commit pipeline: incremental fetch + index   (uses the above)
├── text_search.py       # generic text: index/add/search plain documents (uses engine)
├── video_search_eval.py # demo: build a video index + measure recall@k
├── requirements.txt
└── data/                # datasets + SQLite indexes (the .db files are gitignored)
    ├── ebay-evo-web.jsonl        # extracted commits
    ├── ebay-evo-web.db           # commit index (regenerable)
    ├── sample-videos.jsonl       # 23 labeled demo videos
    ├── sample-video-queries.jsonl
    └── sample-videos.db          # video search corpus (regenerable)
```

**Naming convention:** a JSONL file names its index — `foo.jsonl` → `foo.db`
(same directory and stem). A repo maps to a slug: `eBay/evo-web` →
`ebay-evo-web.jsonl` / `ebay-evo-web.db`.

The `.db` files are **gitignored** (large and regenerable); the `.jsonl` source
data is kept.

---

## Setup

Uses the project venv (Python 3.14 with `sentence-transformers`, `torch`,
`numpy`). From this directory, prefix commands with `../venv/bin/python`:

```bash
# (if recreating the env)
../venv/bin/pip install -r requirements.txt
```

`git` must be on PATH (used by the commit pipeline). On Apple Silicon the models
run on the MPS GPU. **The first search of a session downloads + loads the models
(~2 GB), so it has a one-time startup delay.**

---

## Commit search: fetch → index → search

### 1. Pull commits and build the index

The simplest path is `sync_commits.py`, which is incremental and self-bootstrapping.
On the **first** run (empty index) it pulls the repo's full history into
`data/<slug>.jsonl` and embeds it into `data/<slug>.db`:

```bash
../venv/bin/python sync_commits.py eBay/evo-web
```

For a **private** repo, export a token first (passed as an HTTP header, never put
in the clone URL):

```bash
export GITHUB_TOKEN=ghp_...
../venv/bin/python sync_commits.py eBay/evo-web
```

> Prefer a specific date window instead of full history? Use `fetch_commits.py`
> to dump a window, then `--migrate` to embed it:
> ```bash
> ../venv/bin/python fetch_commits.py eBay/evo-web \
>     --since 2025-01-01 --until 2026-06-20 --out data/ebay-evo-web.jsonl
> ../venv/bin/python sync_commits.py eBay/evo-web --migrate
> ```

### 2. Bring in new commits later

Re-running `sync` resumes from the newest commit already in the index, fetches
everything up to today, appends to the JSONL, and embeds only the new commits:

```bash
../venv/bin/python sync_commits.py eBay/evo-web
```

Useful flags: `--branch <name>`, `--no-index` (update the JSONL only, skip the
model), `--db-dir <dir>` (default `data/`).

### 3. Search the commits

Use `--reuse-db` so it queries the existing index instead of re-embedding:

```bash
../venv/bin/python commit_index.py data/ebay-evo-web.jsonl --reuse-db \
    --query "make ebay card image optional" --top-n 5
```

This runs dense cosine → top-50 candidates → cross-encoder rerank → top 5. Tune
with `--candidate-k` (candidates fed to the reranker) and `--top-n`.

---

## Video / generic text search

`text_search.py` indexes any plain text. Input is either a **one-document-per-line**
text file or a **`.jsonl`** of `{"text": ..., "id"?: ...}`. Ids default to a
content hash, so re-indexing the same text is idempotent.

The provided `data/sample-videos.db` is a corpus of ~11k YouTube videos
(`title. description` per video). Search it:

```bash
../venv/bin/python text_search.py search "rare street food in a remote village" \
    --db data/sample-videos.db --top-n 5
```

Flags:
- `--top-n N` — number of results (default 5)
- `--candidate-k K` — dense candidates fed to the reranker (default 50; raise it
  on a large corpus, e.g. `--candidate-k 100`)
- `--no-rerank` — pure cosine nearest-neighbor (faster, skips the cross-encoder)

Build your own index from a file:

```bash
# one document per line  ->  notes.db
../venv/bin/python text_search.py index notes.txt --db data/notes.db

# add more documents later (incremental, dedup by content hash)
../venv/bin/python text_search.py add more.txt --db data/notes.db
```

### Measuring quality (recall@k)

`video_search_eval.py` builds a small, controlled index from the 23 labeled demo
videos and runs the labeled queries in `data/sample-video-queries.jsonl`,
reporting recall@k and MRR. Several queries deliberately share **no words** with
their target (e.g. "leaky faucet" → "dripping tap") to test semantic matching
over keyword matching.

```bash
# cosine -> rerank
../venv/bin/python video_search_eval.py

# pure cosine nearest-neighbor, for comparison
../venv/bin/python video_search_eval.py --no-rerank
```

It builds its own `data/sample-videos-eval.db` (a clean 23-video set) and does
**not** touch the large `data/sample-videos.db` search corpus.

---

## How it fits together

```
fetch_commits.py ─► data/*.jsonl ─► commit_index.py (CommitIndex) ┐
                                                                   ├─► engine.py
text files / jsonl ───────────────► text_search.py (TextIndex) ───┘   (BGE-M3 +
                                                                       reranker,
sync_commits.py orchestrates fetch + incremental index for commits.   area-batched)
```

`engine.py` owns the only copy of the embedding (`embed_texts`) and reranking
(`rerank_scores`) logic, including the token-area batching that keeps BGE-M3's
full 8192-token context without exhausting GPU memory. Both `CommitIndex` and
`TextIndex` store **raw** float32 vectors and unit-normalize once at load, so the
cosine search stays a single matmul.
