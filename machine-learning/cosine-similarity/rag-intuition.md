# From word vectors to RAG: it's the same cosine nearest-neighbour

You've seen `king − man + woman ≈ queen` work by finding the **cosine nearest
neighbour** of a point in a 300-d space. Retrieval-augmented generation (RAG)
is that *exact* operation — just over sentence/chunk vectors instead of word
vectors. This note builds the intuition and answers the natural question:
**is a sentence embedding just the sum of its word vectors?**

Short answer: that's the *toy* version, and it's a surprisingly strong
baseline — but real systems replace the sum with a trained encoder.

---

## The through-line

To compare two **strings** you do the same two things you did for words:

1. turn each string into a **single fixed-length vector**, then
2. take the **cosine similarity** (= dot product on normalised vectors).

The naive way to build that sentence vector really *is* addition — the classic
"bag of embeddings" baseline:

```
E("the cat sat") = mean( E(the), E(cat), E(sat) )      # average the word vectors
```

So your intuition is half right: collapse to one vector, compare by cosine. The
half that's wrong is *how* you collapse.

## Why "just add the word vectors" isn't enough

Averaging throws away three things, and you can measure each one:

![word vectors to sentence vectors to retrieval](sub-scripts/images/rag_sentence_similarity.png)

The right-hand panel pits word2vec averaging against a trained encoder
(`all-MiniLM-L6-v2`) on three probes:

| Probe | Averaging | Encoder | What averaging gets wrong |
|---|---|---|---|
| **word order** — `dog bit man` / `man bit dog` | **1.00** | 0.98 | Identical bag of words → cosine is **1.00 by construction**. Averaging is *mathematically blind* to order. |
| **polysemy** — loan `bank` / river `bank` | 0.41 | **0.21** | The shared word `bank` inflates the score; the encoder separates the two senses. |
| **paraphrase** — children/gift / kids/present | 0.66 | **0.80** | Few shared words → averaging scores a true paraphrase too low; the encoder sees the meaning. |

The honest takeaway: **word2vec averaging is a strong baseline** — often in the
right ballpark, and sometimes *higher* than the encoder (shared and common words
inflate it). The encoder doesn't win by making every number bigger; it wins by
being **discriminative in the right places**: order, word sense, and paraphrase.
That's exactly what retrieval needs.

The left-hand heatmap is the encoder doing this across several sentences — note
`Python is a programming language` vs `the python slithered…` sits at **0.30**:
same word, different meaning, correctly pulled apart. Plain word2vec, which gives
`python` one static vector, cannot do that.

## What a real encoder does instead of `+`

| Step | word2vec averaging | Trained encoder |
|---|---|---|
| token vectors | static, context-free | **context-dependent** (`bank` differs per sentence) |
| combine | plain mean | learned pooling (mean of the final layer, or a `[CLS]` token) |
| trained for? | nothing — reused as-is | **contrastive objective**: similar texts pulled together, dissimilar pushed apart |

It's still "many token vectors → one vector," but the combiner is a **nonlinear,
trained function**, not `+`. That training is what makes cosine in the output
space track *meaning* rather than word overlap.

## How that becomes RAG

```
Offline:   docs → split into chunks → encoder → one vector per chunk → vector DB
Query:     question → encoder → query vector
Retrieve:  cosine / dot-product nearest neighbours of the query vector   ← the king/queen step
           → top-k chunks → paste into the LLM prompt → generate answer
```

The retrieve step is literally the operation from `word2vec_analogy.py`: take a
query vector, find the highest-cosine neighbours among stored vectors. The only
new engineering:

- the vectors come from a **trained text encoder** instead of `+`, and
- at millions of chunks you use an **approximate** index (FAISS, HNSW) instead of
  scanning everything — but the metric underneath is still cosine, usually on
  normalised vectors (so it's a plain dot product — the equivalence from
  [cosine-vs-euclidean](cosine-vs-euclidean.md)).

## Bottom line

| Question | Answer |
|---|---|
| Is sentence similarity just vector addition? | **In spirit yes** (collapse to one vector, compare by cosine); **in practice no** — averaging is the toy version, real RAG uses a trained encoder. |
| What finds the nearest chunk? | **Cosine similarity** (= normalised dot product), same as the word analogy. |
| Why not just average word vectors? | Blind to order, fooled by shared words (polysemy), and misses paraphrases — see the three probes above. |

> Word2vec taught us the metric (cosine) and the operation (nearest neighbour).
> RAG keeps both and upgrades the *encoder* that turns text into the vector.

---

## Reproducing the figure

```bash
# needs the py3.13 sidecar venv (gensim word2vec + sentence-transformers)
venv313/bin/python sub-scripts/rag_intuition.py
```
