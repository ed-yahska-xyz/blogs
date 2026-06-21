"""From word vectors to sentence vectors: the retrieval step behind RAG.

Two ideas, two panels:

  LEFT  — Cosine-similarity heatmap of several sentences embedded with a real
          trained encoder (sentence-transformers). This is the operation a
          vector DB runs at query time: highest cosine = nearest neighbour =
          retrieved chunk.

  RIGHT — Why a trained encoder beats "just add the word vectors". Two probes:
            (A) word ORDER:   "dog bit man"  vs  "man bit dog"
                same words → averaging gives cosine 1.00 (can't tell them apart);
                the encoder separates them.
            (B) PARAPHRASE:   "dog bit man"  vs  "a canine attacked the person"
                few shared words → averaging scores them low;
                the encoder sees the meaning and scores them high.

Needs the py3.13 sidecar venv (gensim + sentence-transformers):
    venv313/bin/python sub-scripts/rag_intuition.py

First run downloads a small sentence encoder (~90 MB) and reuses the cached
word2vec-google-news-300 vectors.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
ENCODER_NAME = "all-MiniLM-L6-v2"
W2V_NAME = "word2vec-google-news-300"

# Sentences for the heatmap: three meaning clusters + a polysemy pair.
SENTENCES = [
    "The cat sat on the mat.",            # 0  pets
    "A kitten napped on the rug.",        # 1  pets  (≈ 0)
    "The stock market crashed today.",    # 2  finance
    "Shares plunged on the exchange.",    # 3  finance (≈ 2)
    "Python is a great programming language.",   # 4  code-python
    "The python slithered through the grass.",   # 5  snake-python (same word, ≠ meaning)
]

# Probe pairs for the averaging-vs-encoder contrast. Each exposes a different
# failure mode of plain word-vector averaging:
#   word order — identical bag of words → averaging is pinned at cos 1.00.
#   polysemy   — shared word "bank", opposite meaning → averaging fooled high.
#   paraphrase — different words, same meaning → averaging too low.
PAIRS = [
    ("word order", "The dog bit the man", "The man bit the dog"),
    ("polysemy", "The bank approved my mortgage loan",
                 "We picnicked on the grassy river bank"),
    ("paraphrase", "The children were delighted by the gift",
                   "The kids felt joy receiving the present"),
]

_TOKEN = re.compile(r"[a-z]+")


def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def w2v_average(wv, sentence):
    """Bag-of-embeddings baseline: mean of the in-vocab word2vec vectors."""
    vecs = [wv[t] for t in _TOKEN.findall(sentence.lower()) if t in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(wv.vector_size)


def heatmap(ax, enc):
    emb = enc.encode(SENTENCES, normalize_embeddings=True)
    sim = emb @ emb.T                      # cosine, vectors already unit-length
    im = ax.imshow(sim, cmap="RdYlGn", vmin=0, vmax=1)
    labels = [s[:28] for s in SENTENCES]
    ax.set_xticks(range(len(SENTENCES)))
    ax.set_yticks(range(len(SENTENCES)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(SENTENCES)):
        for j in range(len(SENTENCES)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#222")
    ax.set_title(f"Sentence cosine similarity  ({ENCODER_NAME})\n"
                 "this is the RAG retrieval step", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")


def contrast(ax, enc, wv):
    avg_scores, enc_scores, labels = [], [], []
    print("\nAveraging vs. encoder:")
    for name, s1, s2 in PAIRS:
        a = cos(w2v_average(wv, s1), w2v_average(wv, s2))
        e1, e2 = enc.encode([s1, s2], normalize_embeddings=True)
        e = float(e1 @ e2)
        avg_scores.append(a)
        enc_scores.append(e)
        labels.append(f"{name}\n“{s1[:26]}…”\n“{s2[:26]}…”")
        print(f"  {name:<11} avg={a:.2f}  encoder={e:.2f}   ({s1!r} / {s2!r})")

    x = np.arange(len(PAIRS))
    w = 0.36
    ax.bar(x - w / 2, avg_scores, w, label="word2vec averaging", color="#9467bd")
    ax.bar(x + w / 2, enc_scores, w, label="trained encoder", color="#2ca02c")
    for xi, (a, e) in enumerate(zip(avg_scores, enc_scores)):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)
    # One-line verdict per pair: is averaging too high, blind, or too low?
    verdicts = ["blind to order\n(1.00 by construction)",
                "fooled high\nby shared word", "too low\n(misses meaning)"]
    for xi, vtxt in enumerate(verdicts):
        ax.text(xi, 1.07, vtxt, ha="center", va="center", fontsize=7.5,
                color="#9467bd", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("cosine similarity")
    ax.set_title("Why RAG isn't 'just add the word vectors'",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.axhline(1.0, color="#aaa", lw=0.8, ls="--")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading encoder '{ENCODER_NAME}'...")
    enc = SentenceTransformer(ENCODER_NAME)
    print(f"loading '{W2V_NAME}' (cached)...")
    wv = api.load(W2V_NAME)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    heatmap(axes[0], enc)
    contrast(axes[1], enc, wv)
    fig.suptitle("Word vectors → sentence vectors → cosine retrieval (RAG)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "rag_sentence_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
