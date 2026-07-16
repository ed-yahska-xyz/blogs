"""A spectrum of vector pairs at gradually increasing angles.

`a` stays fixed on the x-axis; `b` sweeps from 0° up to 180°. Each panel shows
the angle and its cosine similarity, so you can watch cos θ slide from 1 (same
direction) through 0 (orthogonal) to -1 (opposite).

Generates into ./images (relative to this script):

  cosine_spectrum.png   a grid of vector pairs, one per angle.
  cosine_curve.png      the cos θ curve with each sampled angle marked.

Run:  python cosine_spectrum.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")

C_A = "#1f77b4"   # vector a (fixed on x-axis)
C_B = "#d62728"   # vector b (sweeping)
C_ARC = "#555555"

# Angles to sample, in degrees.
ANGLES = list(range(0, 181, 15))   # 0, 15, 30, ... 180
RADIUS = 1.0                        # unit vectors: cos θ = a·b directly


def cos_color(cos):
    """Green when aligned, grey near orthogonal, red when opposed."""
    # map cos in [-1, 1] -> colormap
    return plt.cm.RdYlGn((cos + 1) / 2)


def style_axes(ax, lim=1.3):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.5, lim)   # b stays in the upper half for θ in [0, 180]
    ax.set_aspect("equal")
    for side in ("left", "bottom"):
        ax.spines[side].set_position("zero")
        ax.spines[side].set_color("#bbbbbb")
    for side in ("right", "top"):
        ax.spines[side].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)


def draw_arrow(ax, vec, color):
    ax.annotate(
        "",
        xy=(vec[0], vec[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, shrinkA=0, shrinkB=0),
    )


def annotate_angle(ax, deg, radius=0.35):
    if deg <= 0.5:
        return
    arc = Arc((0, 0), 2 * radius, 2 * radius, angle=0,
              theta1=0, theta2=deg, color=C_ARC, lw=1.4)
    ax.add_patch(arc)


def render_panel(ax, deg):
    a = np.array([RADIUS, 0.0])
    rad = np.radians(deg)
    b = RADIUS * np.array([np.cos(rad), np.sin(rad)])
    cos = float(np.cos(rad))

    style_axes(ax)
    annotate_angle(ax, deg)
    draw_arrow(ax, a, C_A)
    draw_arrow(ax, b, C_B)

    ax.set_title(f"θ = {deg}°", fontsize=11, fontweight="bold", pad=6)
    # cos θ badge in the (empty) lower half of the panel, tinted by the value.
    ax.text(
        0.5, 0.08,
        f"cos θ = {cos:+.2f}",
        transform=ax.transAxes,
        fontsize=10.5, family="monospace", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc=cos_color(cos), ec="#999999"),
    )


def make_spectrum():
    n = len(ANGLES)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, deg in zip(axes, ANGLES):
        render_panel(ax, deg)
    for ax in axes[n:]:           # hide any unused cells
        ax.axis("off")

    fig.suptitle(
        "Cosine similarity as the angle increases  (‖a‖ = ‖b‖ fixed)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(OUT_DIR, "cosine_spectrum.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_curve():
    fig, ax = plt.subplots(figsize=(9, 5))
    thetas = np.linspace(0, 180, 361)
    ax.plot(thetas, np.cos(np.radians(thetas)), color="#333333", lw=2, zorder=1)

    for deg in ANGLES:
        cos = np.cos(np.radians(deg))
        ax.scatter([deg], [cos], s=90, color=cos_color(cos),
                   edgecolor="#444444", zorder=3)
        ax.annotate(f"{cos:+.2f}", (deg, cos),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha="center", color="#444444")

    for y in (1, 0, -1):
        ax.axhline(y, color="#dddddd", lw=1, zorder=0)
    ax.axhline(0, color="#999999", lw=1, zorder=0)

    ax.set_xticks(ANGLES)
    ax.set_xlabel("angle θ between the vectors  (degrees)", fontsize=11)
    ax.set_ylabel("cos θ  (cosine similarity)", fontsize=11)
    ax.set_title("cos θ from 1 (aligned) → 0 (orthogonal) → −1 (opposite)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-5, 185)
    ax.set_ylim(-1.25, 1.25)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cosine_curve.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_spectrum()
    make_curve()


if __name__ == "__main__":
    main()
