"""Why, after L2-normalisation, cosine similarity and Euclidean distance rank
neighbours identically — and why you must normalise first.

Three panels:
  1. Raw vectors      — same direction, different length: cosine = 1 but the
                        Euclidean gap is large. Magnitude fools Euclidean.
  2. Unit circle      — once both are unit length the straight-line (chord)
                        distance is a pure function of the angle:
                        ‖â − b̂‖ = √(2 − 2·cos θ) = 2·sin(θ/2).
  3. Monotonic link   — that function is strictly decreasing in cos θ, so
                        "highest cosine" and "smallest distance" are the same
                        ranking.

Writes images/cosine_euclidean_equiv.png.  Run:  python cosine_euclidean_equiv.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
C_A, C_B, C_DIST = "#1f77b4", "#d62728", "#2ca02c"


def style(ax, lim):
    ax.set_xlim(-0.4, lim)
    ax.set_ylim(-0.4, lim)
    ax.set_aspect("equal")
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, ls=":", alpha=0.3)
    ax.set_axisbelow(True)


def vec(ax, v, color, label, loff=(0.12, 0.12)):
    ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6))
    ax.text(v[0] + loff[0], v[1] + loff[1], label, color=color,
            fontsize=13, fontweight="bold", ha="center", va="center")


def panel_raw(ax):
    u = np.array([2.4, 1.0])
    v = 0.45 * u                      # same direction, shorter
    style(ax, 3.0)
    vec(ax, u, C_A, "a")
    vec(ax, v, C_B, "b = 0.45·a", loff=(0.0, -0.28))
    # Euclidean gap between the tips.
    ax.plot([u[0], v[0]], [u[1], v[1]], color=C_DIST, lw=2, ls="--")
    dist = np.linalg.norm(u - v)
    # After L2-normalising, both collapse onto the SAME unit-circle point.
    n = u / np.linalg.norm(u)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#bbb",
                            lw=1.0, ls=":"))
    ax.scatter(*n, s=90, color=C_DIST, zorder=4, edgecolor="white")
    ax.annotate("after normalise:\nâ = b̂  →  dist 0", n,
                textcoords="offset points", xytext=(16, -4),
                fontsize=9.5, color=C_DIST, fontweight="bold", va="center")
    ax.text(0.5, -0.20,
            f"same direction → cos θ = 1.00\n‖a − b‖ = {dist:.2f}  (looks far!)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#ccc"))
    ax.set_title("1. Raw vectors: magnitude fools Euclidean",
                 fontsize=12, fontweight="bold")


def panel_circle(ax):
    theta = np.radians(50)
    a = np.array([1.0, 0.0])
    b = np.array([np.cos(theta), np.sin(theta)])
    style(ax, 1.5)
    circ = plt.Circle((0, 0), 1.0, fill=False, color="#999", lw=1.2, ls="-")
    ax.add_patch(circ)
    vec(ax, a, C_A, "â", loff=(0.13, -0.12))
    vec(ax, b, C_B, "b̂", loff=(0.14, 0.10))
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_DIST, lw=2.4, ls="--")
    ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=50,
                     color="#555", lw=1.6))
    ax.text(0.42, 0.16, "θ", fontsize=12, color="#555")
    mid = (a + b) / 2
    ax.text(mid[0] + 0.05, mid[1] + 0.05, "‖â − b̂‖", color=C_DIST,
            fontsize=11, fontweight="bold")
    ax.text(0.5, -0.18,
            "both unit length →\n‖â − b̂‖ = √(2 − 2·cos θ) = 2·sin(θ/2)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef7ee", ec=C_DIST))
    ax.set_title("2. Normalised: distance is a function of the angle",
                 fontsize=12, fontweight="bold")


def panel_curve(ax):
    c = np.linspace(-1, 1, 400)
    d = np.sqrt(2 - 2 * c)
    ax.plot(c, d, color=C_DIST, lw=2.5)
    # Mark the θ = 50° point from panel 2.
    c0 = np.cos(np.radians(50))
    ax.scatter([c0], [np.sqrt(2 - 2 * c0)], s=80, color=C_B, zorder=3,
               edgecolor="white")
    ax.annotate("θ = 50°", (c0, np.sqrt(2 - 2 * c0)),
                textcoords="offset points", xytext=(10, 8), fontsize=9)
    for cc, lab in [(1, "identical"), (0, "orthogonal"), (-1, "opposite")]:
        ax.annotate(lab, (cc, np.sqrt(2 - 2 * cc)),
                    textcoords="offset points",
                    xytext=(0, 10 if cc != 1 else -16),
                    ha="center", fontsize=8, color="#666")
    ax.set_xlabel("cosine similarity  cos θ", fontsize=11)
    ax.set_ylabel("Euclidean distance  ‖â − b̂‖", fontsize=11)
    ax.set_title("3. Strictly decreasing → same ranking",
                 fontsize=12, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    ax.invert_xaxis()   # so "more similar" (cos→1) is on the right-to-left sweep
    ax.text(0.5, 0.92,
            "higher cosine  ⇔  smaller distance\nso the nearest neighbour is the same",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color="#444")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    panel_raw(axes[0])
    panel_circle(axes[1])
    panel_curve(axes[2])
    fig.suptitle(
        "Normalise → cosine and Euclidean give the same neighbours",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cosine_euclidean_equiv.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
