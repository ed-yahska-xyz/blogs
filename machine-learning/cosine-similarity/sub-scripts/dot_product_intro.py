"""Illustrate the dot product / cosine similarity geometrically.

Generates four PNGs into ./images (relative to this script):

  1. dot_product_intro.png   one vector on the x-axis, one in the x-y plane,
                             with the dot product and cosine similarity shown.
  2. orthogonal.png          two perpendicular vectors  (cos = 0).
  3. overlapping.png         two co-linear vectors, one longer than the other
                             (cos = 1).
  4. cosine_panels.png       all three cases side by side.

Run:  python dot_product_intro.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")

# A small, color-blind-friendly palette.
C_A = "#1f77b4"  # vector a
C_B = "#d62728"  # vector b
C_ARC = "#555555"


def draw_vector(ax, vec, color, label, label_offset=(0.15, 0.15)):
    """Draw `vec` as an arrow from the origin and label its tip."""
    ax.annotate(
        "",
        xy=(vec[0], vec[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, shrinkA=0, shrinkB=0),
    )
    ax.text(
        vec[0] + label_offset[0],
        vec[1] + label_offset[1],
        label,
        color=color,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )


def angle_between(a, b):
    """Angle (degrees) measured from the positive x-axis."""
    return np.degrees(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def style_axes(ax, lim):
    """Center the spines on the origin so it reads as a coordinate plane."""
    ax.set_xlim(-0.6, lim)
    ax.set_ylim(-0.6, lim)
    ax.set_aspect("equal")
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(colors="#999999", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def annotate_angle(ax, a, b, radius=0.9):
    """Draw an arc between vectors a and b and return the angle in degrees."""
    start = np.degrees(np.arctan2(a[1], a[0]))
    end = np.degrees(np.arctan2(b[1], b[0]))
    arc = Arc(
        (0, 0),
        2 * radius,
        2 * radius,
        angle=0,
        theta1=min(start, end),
        theta2=max(start, end),
        color=C_ARC,
        lw=1.8,
    )
    ax.add_patch(arc)
    mid = np.radians((start + end) / 2)
    theta = abs(end - start)
    ax.text(
        (radius + 0.25) * np.cos(mid),
        (radius + 0.25) * np.sin(mid),
        f"θ = {theta:.0f}°",
        color=C_ARC,
        fontsize=11,
        ha="center",
        va="center",
    )


def info_box(ax, a, b):
    """Print dot product, magnitudes and cosine similarity in the corner."""
    dot = float(np.dot(a, b))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = cosine_similarity(a, b)
    text = (
        f"a · b = {dot:.2f}\n"
        f"‖a‖ = {na:.2f}   ‖b‖ = {nb:.2f}\n"
        f"cos θ = a·b / (‖a‖‖b‖) = {cos:.2f}"
    )
    ax.text(
        0.5,
        -0.16,
        text,
        transform=ax.transAxes,
        fontsize=10.5,
        family="monospace",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#cccccc"),
    )


def render_case(ax, a, b, title, lim, label_a="a", label_b="b"):
    style_axes(ax, lim)
    annotate_angle(ax, a, b)
    draw_vector(ax, a, C_A, label_a)
    draw_vector(ax, b, C_B, label_b)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    info_box(ax, a, b)


# The three illustrative cases. `a` always lies on the x-axis.
CASES = {
    "dot_product_intro": dict(
        a=np.array([3.0, 0.0]),
        b=np.array([2.6, 2.0]),
        title="A vector on the x-axis and one in the x-y plane",
        lim=3.6,
    ),
    "orthogonal": dict(
        a=np.array([3.0, 0.0]),
        b=np.array([0.0, 3.0]),
        title="Orthogonal vectors  (cos θ = 0)",
        lim=3.6,
    ),
    "overlapping": dict(
        a=np.array([3.0, 0.0]),
        b=np.array([1.7, 0.0]),
        title="Overlapping vectors, one longer  (cos θ = 1)",
        lim=3.6,
    ),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1-3: one figure per case.
    for name, cfg in CASES.items():
        fig, ax = plt.subplots(figsize=(5.5, 6))
        render_case(ax, cfg["a"], cfg["b"], cfg["title"], cfg["lim"])
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # 4: all three side by side.
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, cfg) in zip(axes, CASES.items()):
        render_case(ax, cfg["a"], cfg["b"], cfg["title"], cfg["lim"])
    fig.suptitle(
        "Dot product & cosine similarity", fontsize=15, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cosine_panels.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
