"""Word2Vec analogy demo + parallelogram plot.

Demonstrates the famous vector arithmetic on real Word2Vec embeddings:

    E(king) - E(man) + E(woman)  ≈  E(queen)

Prints the nearest neighbours for several analogies, then plots the
king / man / woman / queen parallelogram in 2D (PCA), showing both the
*predicted* point (king - man + woman) and the *actual* queen vector.

Uses gensim's pretrained vectors. Default model is the classic
word2vec-google-news-300 (~1.6 GB on first download, cached afterwards).
Pass a smaller model name as argv[1] to iterate faster, e.g.

    python word2vec_analogy.py glove-wiki-gigaword-100

Run (needs the py3.13 sidecar venv where gensim is installed):
    venv313/bin/python sub-scripts/word2vec_analogy.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "word2vec-google-news-300"

# Analogies to print: result ≈ positive[0] - negative[0] + positive[1]
ANALOGIES = [
    (["king", "woman"], ["man"], "king - man + woman"),
    (["paris", "italy"], ["france"], "paris - france + italy"),
    (["walking", "swim"], ["walk"], "walking - walk + swim"),
    (["bigger", "small"], ["big"], "bigger - big + small"),
]

C_WORD = "#1f77b4"
C_PRED = "#2ca02c"
C_OFFSET = "#d62728"
C_NB = "#ff7f0e"     # candidate neighbours of the predicted vector
TOP_N = 6            # how many nearest neighbours to show


def report_analogies(wv):
    print(f"\nModel: {MODEL_NAME}   (vocab={len(wv):,}, dim={wv.vector_size})\n")
    for positive, negative, label in ANALOGIES:
        if not all(w in wv for w in positive + negative):
            missing = [w for w in positive + negative if w not in wv]
            print(f"{label:<26} -> skipped (missing {missing})")
            continue
        top = wv.most_similar(positive=positive, negative=negative, topn=3)
        pretty = ", ".join(f"{w} ({s:.2f})" for w, s in top)
        print(f"{label:<26} -> {pretty}")
    print()


def plot_parallelogram(wv, words, offset_labels, title_expr, out_name,
                       metric="cosine"):
    """Plot the analogy parallelogram for a:b :: c:d, where the arithmetic is
    predicted = E(b) - E(a) + E(c)  and  d  is the expected answer.

    offset_labels = (along a->b / c->pred,  along a->c / b->pred).

    metric controls the number shown next to each candidate:
      "cosine"    — cosine similarity in the original 300-d space (higher =
                    closer; this is what actually picks the answer).
      "euclidean" — straight-line distance in the *PCA plane* (lower = closer).
                    This is the only metric the 2-D picture faithfully shows,
                    since PCA does not preserve cosine.
    """
    a, b, c, d = words
    for w in words:
        if w not in wv:
            print(f"cannot plot {out_name}: '{w}' not in vocab")
            return

    # Predicted vector via the analogy, in the original space.
    predicted = wv[b] - wv[a] + wv[c]
    off1, off2 = offset_labels

    # The analogy is really a ranked nearest-neighbour search around `predicted`.
    # gensim excludes the three input words automatically.
    neighbors = wv.most_similar(positive=[b, c], negative=[a], topn=TOP_N)
    neighbor_words = [w for w, _ in neighbors]

    # Words to project: inputs, then any neighbours / expected answer not already in.
    inputs = [a, b, c]
    extras = []
    for w in neighbor_words + [d]:
        if w not in inputs and w not in extras:
            extras.append(w)
    all_words = inputs + extras

    # Project everything with one PCA so distances are comparable.
    mat = np.vstack([wv[w] for w in all_words] + [predicted])
    pca = PCA(n_components=2, random_state=0)
    pts = pca.fit_transform(mat)
    P = {w: pts[i] for i, w in enumerate(all_words)}
    p_pred = pts[-1]
    p_a, p_b, p_c = P[a], P[b], P[c]

    # Per-candidate score under the chosen metric, plus the ranked display order.
    cos_of = dict(neighbors)
    if metric == "euclidean":
        score = {w: float(np.linalg.norm(P[w] - p_pred)) for w in neighbor_words}
        ranked = sorted(neighbor_words, key=lambda w: score[w])   # nearer first
        d_score = float(np.linalg.norm(P[d] - p_pred))
        metric_line = f"PCA-plane dist(predicted, {d}) = {d_score:.2f}"
        list_header = "nearest to predicted  (PCA-plane distance):"
        fmt = "{:.2f}"
    else:  # cosine
        score = cos_of
        ranked = sorted(neighbor_words, key=lambda w: -score[w])  # higher first
        d_score = float(wv.cosine_similarities(predicted, wv[d][None, :])[0])
        metric_line = f"cosine(predicted, {d}) = {d_score:.3f}"
        list_header = "nearest to predicted  (cosine):"
        fmt = "{:.2f}"
    rank_of = {w: i + 1 for i, w in enumerate(ranked)}
    nearest_word = ranked[0]
    hit = "✓ top match" if nearest_word == d else f"✗ top match is '{nearest_word}'"

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Parallelogram a -> b -> predicted -> c -> a.
    quad = np.vstack([p_a, p_b, p_pred, p_c, p_a])
    ax.plot(quad[:, 0], quad[:, 1], color="#bbbbbb", lw=1.5, ls="--", zorder=1)

    # The two shared offsets, drawn as arrows. Label each only once (on the
    # input-side arrow) to keep the crowded prediction corner readable.
    def arrow(p0, p1, color, label=""):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4))
        if label:
            mid = (p0 + p1) / 2
            ax.text(mid[0], mid[1], label, color=color, fontsize=10,
                    fontweight="bold", ha="center", va="bottom")

    arrow(p_a, p_b, C_OFFSET, off1)       # a -> b   (labelled)
    arrow(p_c, p_pred, C_OFFSET)          # c -> predicted
    arrow(p_a, p_c, "#9467bd", off2)      # a -> c   (labelled)
    arrow(p_b, p_pred, "#9467bd")         # b -> predicted

    # Push a label radially away from the prediction so coincident points
    # (king ≈ queen ≈ predicted here) fan out instead of overprinting.
    def away(point, dist=16):
        v = np.asarray(point) - p_pred
        n = np.linalg.norm(v)
        return (8.0, 8.0) if n < 1e-9 else (v[0] / n * dist, v[1] / n * dist)

    # Input word points.
    for w in inputs:
        ax.scatter(*P[w], s=130, color=C_WORD, zorder=3, edgecolor="white")
        ax.annotate(w, P[w], textcoords="offset points", xytext=away(P[w], 18),
                    fontsize=13, fontweight="bold", color=C_WORD)

    # Candidate neighbours of `predicted`, ranked under the chosen metric.
    # Faint spokes + rank badges show they cluster around the prediction and
    # we simply take the closest.
    for w in neighbor_words:
        ax.plot([p_pred[0], P[w][0]], [p_pred[1], P[w][1]],
                color=C_NB, lw=0.8, ls=":", alpha=0.6, zorder=2)
        ax.scatter(*P[w], s=90, marker="D", color=C_NB, zorder=3,
                   edgecolor="white")
        ax.annotate(str(rank_of[w]), P[w], textcoords="offset points",
                    xytext=away(P[w], 13), fontsize=10, fontweight="bold",
                    color=C_NB)

    # Expected answer always gets a cyan dashed spoke, a ring, and a legend
    # entry, so its gap from the prediction is visible in every plot. When it
    # also made the neighbour list it already has an orange diamond underneath;
    # when it didn't, add a solid X and a label so it isn't an empty ring.
    ax.plot([p_pred[0], P[d][0]], [p_pred[1], P[d][1]],
            color="#17becf", lw=1.6, ls="--", zorder=2)
    ax.scatter(*P[d], s=260, marker="o", facecolor="none", edgecolor="#17becf",
               linewidths=2.0, zorder=4,
               label=f"expected: {d} ({fmt.format(d_score)})")
    if d not in neighbor_words:
        ax.scatter(*P[d], s=150, marker="X", color="#17becf", zorder=4,
                   edgecolor="white")
        ax.annotate(f"expected: {d}", P[d], textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold",
                    color="#0e8a99")

    # Predicted point.
    ax.scatter(*p_pred, s=200, marker="*", color=C_PRED, zorder=5,
               edgecolor="white", label=f"{b} − {a} + {c} (predicted)")
    ax.annotate("predicted", p_pred, textcoords="offset points",
                xytext=(0, -22), ha="center", va="top",
                fontsize=11, color=C_PRED, fontweight="bold")

    # Ranked neighbour list, to read the scores off the picture. The expected
    # answer is tagged; if it missed the top-N it's appended below a divider so
    # its score is still visible next to the winners.
    def line(w):
        tag = "  <- expected" if w == d else ""
        return f"{rank_of[w]}. {w:<11} " + fmt.format(score[w]) + tag
    lines = [line(w) for w in ranked]
    if d not in neighbor_words:
        lines.append("-" * 24)
        lines.append(f"   {d:<11} " + fmt.format(d_score) + "  <- expected")
    ax.text(0.02, 0.98, list_header + "\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=10, family="monospace",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff5e6", ec=C_NB))

    ax.set_title(
        f"{title_expr}\n"
        f"{metric_line}   ·   {hit}   ·   {MODEL_NAME}",
        fontsize=13, fontweight="bold",
    )
    if metric == "euclidean":
        ax.text(
            0.5, -0.13,
            "PCA-plane distances are relative to this figure's projection "
            "(fit on these words only) — not absolute. Cosine is projection-"
            "independent.",
            transform=ax.transAxes, fontsize=8.5, style="italic",
            color="#666666", ha="center", va="top", wrap=True,
        )
    ax.scatter([], [], s=130, color=C_WORD, label="input words")
    ax.scatter([], [], s=90, marker="D", color=C_NB, label="ranked neighbours")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# (words, offset_labels, title_expr, file stem)
PLOTS = [
    # Works cleanly: king - man + woman ≈ queen
    (("man", "king", "woman", "queen"), ("+royalty", "gender"),
     "king − man + woman ≈ queen", "word2vec_analogy"),
    # Fails instructively: paris - france + italy should be rome (it isn't)
    (("france", "paris", "italy", "rome"), ("+capital", "country"),
     "paris − france + italy ≈ rome ?", "word2vec_analogy_paris"),
    # Works cleanly: walking - walk + swim ≈ swimming (verb -ing inflection)
    (("walk", "walking", "swim", "swimming"), ("+ing", "verb"),
     "walking − walk + swim ≈ swimming", "word2vec_analogy_swim"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading '{MODEL_NAME}' (first run downloads + caches it)...")
    wv = api.load(MODEL_NAME)
    report_analogies(wv)
    for words, offsets, title, stem in PLOTS:
        # cosine version keeps the original filename; euclidean adds a suffix.
        plot_parallelogram(wv, words, offsets, title,
                           f"{stem}.png", metric="cosine")
        plot_parallelogram(wv, words, offsets, title,
                           f"{stem}_euclid.png", metric="euclidean")


if __name__ == "__main__":
    main()
