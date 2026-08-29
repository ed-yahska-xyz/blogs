# Plan: commit-history RAG for task planning

> Status: **planning only** (written 2026-06-07). Nothing built yet.
> A direct, applied sequel to the cosine-similarity work in
> `../cosine-similarity/` — the retrieval step here *is* the cosine
> nearest-neighbour search from that blog, applied to commit vectors.

## 1. The goal (restated)

Build a system that:

1. Extracts **all commits** of a git repo (message + body + diff).
2. **Summarises** each commit's intent.
3. Stores a **vector representation** of each summary.
4. Accepts a **new task** described in natural language.
5. Finds the **closest past commit(s)** by cosine similarity.
6. Pulls those commits' **details + diffs** and drafts an **implementation plan**
   for the new task, grounded in how similar changes were done before.
7. **Evaluates** that plan.

Plus the meta-question: **does something already do this?**

---

## 2. Does this already exist? (research findings)

**The retrieval half is well-solved.** Several tools already embed commit
messages/diffs and rank them by cosine similarity to a natural-language query:

- **[Spelungit](https://github.com/haacked/spelungit)** — an **MCP server** that
  semantic-searches git history (embeds *both* messages and code changes) and
  plugs straight into **Claude Code**. Closest existing thing to steps 1–5, and
  it already speaks our harness's protocol.
- **[git-semantic-similarity (PyPI)](https://pypi.org/project/git-semantic-similarity/)**
  — sentence-transformers over commit messages, embeddings cached on disk.
- **[NaLCoS](https://nalcos.thepushkarp.com/)** — natural-language commit search;
  encodes query + commits and ranks by **cosine similarity** (multi-qa-MiniLM).
- **[dokosa](https://github.com/sile/dokosa)** — CLI semantic search over local
  repos with vector embeddings.
- **[Continue.dev + LanceDB](https://blog.continue.dev/building-a-semantic-code-history-search-with-lancedb/)**
  — semantic code-history search recipe.

**Commit/diff representation is a studied problem.**

- **[CC2Vec](https://arxiv.org/abs/2003.05620)** — distributed representations of
  code changes, supervised by log messages (attention over added/removed hunks).
- **[Commit2Vec](https://ar5iv.labs.arxiv.org/html/1911.07605)** — AST-path-based
  code-change embeddings.

**The "retrieve similar diffs → generate" half exists in research, not as a
product:**

- **[CoRaCMG](https://arxiv.org/abs/2509.18337)** — retrieval-augmented commit
  *message* generation: fetch similar diff–message pairs as few-shot context
  (big BLEU/CIDEr gains). REACT does the same idea.
- **[LAURA](https://arxiv.org/pdf/2512.01356)** — retrieves similar diffs + their
  review comments to augment code-*review* generation.

### Verdict

| Sub-problem | State of the art | Build or reuse? |
|---|---|---|
| Extract + embed commits | Solved (Spelungit, git-semantic-similarity, dokosa) | **Reuse** |
| Cosine nearest-neighbour retrieval | Solved (every tool above) | **Reuse** |
| Diff-aware embeddings | Researched (CC2Vec) | Reuse simple encoder first |
| Task → **plan** from similar diffs | Research only (CoRaCMG/REACT/LAURA are msg/review, not *plans*) | **Build (the novel glue)** |
| Evaluating the generated plan | Open | **Build** |

**No off-the-shelf tool does the full step-6 ("describe a task → get an
implementation plan synthesised from similar past diffs").** That synthesis +
evaluation layer is where the new work is. Recommendation: **don't reinvent
retrieval** — stand up Spelungit (or a 50-line sentence-transformers indexer)
and spend effort on the plan-generation + evaluation layer.

---

## 3. Recommended architecture

Two-reuse-one-build pipeline. Offline indexing, online query.

```
INDEX (offline, once + incremental)
  git log/show ──▶ per-commit {sha, msg, body, files, diff, stats}
       │
       ├─▶ summarise intent (LLM, cached)  ──▶ summary text
       │
       └─▶ embed(summary [+ message])  ──▶ vector  ──▶ vector store + metadata

QUERY (online)
  new-task text ──▶ embed ──▶ cosine top-k ──▶ [optional cross-encoder rerank]
       │
       ▼
  gather top-k commits' {summary, files, diff}
       │
       ▼
  LLM plan: "implement <task>, grounded in these k similar past changes"
       │
       ▼
  evaluate plan (LLM-judge + groundedness checks + optional leave-one-out)
```

### Component choices (MVP → scale)

| Component | MVP (reuse our env) | Scale-up |
|---|---|---|
| Commit extraction | `git log --no-merges` + `git show <sha>` via subprocess | GitPython; incremental by last-indexed sha |
| Summarisation | Claude (or skip: use subject+body) | batched, cached by sha; map-reduce for huge diffs |
| Embedding model | `sentence-transformers` (already in `venv313`); `all-MiniLM-L6-v2` or `multi-qa-MiniLM-L6-cos-v1` | code-aware model (e.g. `jinaai/jina-embeddings-v2-base-code`); or CC2Vec-style |
| Vector store | numpy `.npz` + cosine (same as cosine-similarity blog!) | Chroma / LanceDB / FAISS (HNSW) |
| Retrieval | normalised dot product, top-k | ANN index + metadata filters (path, author, date) |
| Rerank | none | cross-encoder over (task, commit summary) |
| Plan generation | Claude Code itself, or Claude API | structured output; tool-use to read current files |
| Evaluation | LLM-judge + file-existence checks | leave-one-commit-out harness (below) |

### What to embed (key decision)

Commit **messages are short and noisy**; **diffs are large and code-y**. Best
signal comes from embedding an **LLM summary of (message + diff)** that captures
*intent + approach* in clean prose — then the encoder (trained on prose) works
in its comfort zone. Cheaper fallback: embed `subject + body`; add diff later.
Consider storing **two vectors** per commit (message-vector, diff-summary-vector)
and fusing scores.

---

## 4. Data model

```
Commit {
  sha, author, date, subject, body,
  files: [path, +adds, -dels], diff (or per-file diffs),
  summary,                    # LLM-generated intent
  embedding: float[d],        # of summary (+ optional message embedding)
  parent_sha, is_merge
}
```

Persist: `commits.parquet`/`.npz` for vectors + a JSONL/SQLite sidecar for
metadata. Index incrementally: store the last-indexed sha, only process new ones.

---

## 5. Evaluation methodology (the "evaluate this plan" step)

Two things to evaluate: **retrieval quality** and **plan quality**. Crucially,
git history gives us **free ground truth**.

**A. Retrieval eval** — leave-one-commit-out:
- Take a real commit `C`. Use its message as the "new task". Hide `C`.
- Does the system retrieve commits that touch the **same files/component** as `C`?
- Metrics: precision@k, MRR, recall of `C`'s files among retrieved commits' files.

**B. End-to-end plan eval** — the strong test (CoRaCMG/REACT-style):
- Task = a held-out commit's message; **gold = that commit's actual diff**.
- Run the pipeline (which must *not* see `C`'s diff) to produce a plan.
- Score plan vs. gold diff:
  - **file-set overlap** (did it predict the right files to change?) — precision/recall,
  - **embedding similarity** of plan text vs. actual diff summary,
  - optional BLEU/CIDEr against the real change description.

**C. Plan-quality LLM-judge** — rubric: grounded in retrieved diffs (no
invented files), references real repo paths, ordered/actionable, matches repo
conventions. Score 1–5 per criterion; report mean + disagreement.

**D. Groundedness guardrail** — every file/path the plan names must exist in the
repo (or be a plausible new file). Flag hallucinated references automatically.

This doubles as the answer to "evaluate this plan": run B + C and report the
file-overlap number, the judge scores, and any groundedness flags.

---

## 6. Risks & limitations

- **Codebase drift** — a similar *old* commit may use patterns since refactored;
  the plan must be checked against the *current* tree, not just history.
- **Diff size vs. token limits** — summarise/chunk diffs; don't embed raw 5k-line diffs.
- **Commit granularity** — squashed vs. atomic commits vary wildly across repos;
  retrieval quality depends on history hygiene.
- **Message quality** — "fix", "wip" commits carry no signal; the LLM-summary
  step (from the diff) mitigates this.
- **Summarisation cost** — one LLM call per commit; cache by sha, do once,
  incremental thereafter.
- **"Similar" ≠ "correct"** — retrieval finds *topically* similar work, not a
  guaranteed-correct recipe. Keep a human in the loop; treat the plan as a draft.
- **Single-vector ceiling** — averaging/one-vector-per-commit loses structure
  (same caveat as `../cosine-similarity/rag-intuition.md`).

---

## 7. Milestones

- **M0 — spike (½ day):** point **Spelungit** at this repo; eyeball whether
  natural-language retrieval over our commits is good enough. Decides reuse vs.
  build for retrieval.
- **M1 — indexer (1 day):** subprocess `git` extractor → sentence-transformers
  embed of `subject+body` → `.npz` + metadata. Reuse `venv313`. CLI: `index`.
- **M2 — retrieval (½ day):** `query "<task>"` → cosine top-k with metadata.
  Validate with the leave-one-out **retrieval** eval (§5A).
- **M3 — summaries (1 day):** add cached LLM intent-summaries; re-embed; compare
  retrieval quality vs. M1 (does summary-embedding beat message-embedding?).
- **M4 — plan generation (1 day):** feed top-k diffs + task to an LLM → structured
  plan. Could be a **Claude Code skill / MCP** so it runs in-harness.
- **M5 — evaluation harness (1–2 days):** leave-one-commit-out end-to-end (§5B)
  + LLM-judge (§5C) + groundedness guard (§5D). Report the numbers.
- **M6 — polish:** ANN index (LanceDB/FAISS), reranker, incremental indexing.

MVP = **M1+M2+M4** (skip summaries and formal eval) to feel the loop quickly;
add M3/M5 once it's worth measuring.

---

## 8. Open decisions (for when you're back)

1. **Reuse Spelungit** for retrieval, or build the thin indexer? (Spike M0 first.)
2. **Embed messages, diff-summaries, or both?** (M3 answers it empirically.)
3. **Where does plan-gen run** — standalone Python + Claude API, or as a **Claude
   Code skill/MCP** so it can also read the current tree while planning?
4. **Scope:** single-repo tool, or generic across repos?
5. **Embedding model:** general (MiniLM/BGE) vs. code-aware (jina-code)? Bench on
   the M2 retrieval eval.

---

## 9. References

- Spelungit (MCP git semantic search): https://github.com/haacked/spelungit
- git-semantic-similarity: https://pypi.org/project/git-semantic-similarity/
- NaLCoS: https://nalcos.thepushkarp.com/
- dokosa: https://github.com/sile/dokosa
- Continue.dev + LanceDB: https://blog.continue.dev/building-a-semantic-code-history-search-with-lancedb/
- CC2Vec: https://arxiv.org/abs/2003.05620
- Commit2Vec: https://ar5iv.labs.arxiv.org/html/1911.07605
- CoRaCMG (RAG for commit msgs): https://arxiv.org/abs/2509.18337
- LAURA (RAG for code review): https://arxiv.org/pdf/2512.01356
```
