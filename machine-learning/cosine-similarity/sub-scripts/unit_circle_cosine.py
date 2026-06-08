"""The starting point: cosine is the x-coordinate of a unit vector.

Draws the unit circle with one angle highlighted as a right triangle
(cos θ = adjacent/hypotenuse = x/1 = the x-coordinate), plus dots at the
standard angles labelled with their cosine values and tinted green→red as
cos θ runs from +1 to −1.

Writes images/unit_circle_cosine.png.  Run:  python unit_circle_cosine.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
C_VEC, C_COS, C_DROP = "#1f77b4", "#2ca02c", "#999999"
HIGHLIGHT = 35                       # degrees, the worked example
MARKS = list(range(0, 360, 45))      # 0,45,...,315


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Unit circle + axes through the origin.
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#888", lw=1.6))
    ax.axhline(0, color="#bbb", lw=1)
    ax.axvline(0, color="#bbb", lw=1)

    # Dots at the standard angles, tinted by cos θ, labelled with the value.
    for deg in MARKS:
        r = np.radians(deg)
        x, y = np.cos(r), np.sin(r)
        ax.scatter(x, y, s=80, color=plt.cm.RdYlGn((x + 1) / 2),
                   edgecolor="#444", zorder=3)
        lx, ly = 1.16 * x, 1.16 * y
        ax.text(lx, ly, f"{deg}°\ncos={x:+.2f}", ha="center", va="center",
                fontsize=8.5, color="#333")

    # Worked example: cos θ as the x-projection (a right triangle).
    r = np.radians(HIGHLIGHT)
    x, y = np.cos(r), np.sin(r)
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_VEC, lw=2.6))
    ax.plot([x, x], [y, 0], color=C_DROP, lw=1.6, ls="--")          # drop to axis
    ax.plot([0, x], [0, 0], color=C_COS, lw=4, solid_capstyle="butt")  # cos θ
    ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=0, theta2=HIGHLIGHT,
                     color="#555", lw=1.6))
    ax.text(0.32, 0.12, "θ", fontsize=13, color="#555")
    ax.text(x / 2, -0.09, "cos θ", color=C_COS, fontsize=12,
            fontweight="bold", ha="center", va="top")
    # "length 1" written along the hypotenuse (the unit vector).
    nx, ny = -np.sin(r), np.cos(r)            # unit normal to the vector
    ax.text(x / 2 + 0.07 * nx, y / 2 + 0.07 * ny, "length 1", color=C_VEC,
            fontsize=9.5, fontweight="bold", rotation=HIGHLIGHT,
            rotation_mode="anchor", ha="center", va="center")

    ax.text(0, -1.46,
            "cos θ  =  adjacent / hypotenuse  =  x / 1  =  the x-coordinate",
            ha="center", va="top", fontsize=12, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef7ee", ec=C_COS))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.65, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cosine is the x-coordinate on the unit circle",
                 fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "unit_circle_cosine.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
