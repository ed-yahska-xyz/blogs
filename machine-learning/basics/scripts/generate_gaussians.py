"""
Generate Gaussian splat illustrations for the ML basics blog post.
Dark background, glowing purple/blue Gaussians with contour rings.

Produces:
  1. gaussian_isotropic.png  - Circular (equal variance) Gaussian
  2. gaussian_anisotropic.png - Stretched and rotated Gaussian with axes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap: black -> deep purple -> light purple/blue -> white core
splat_cmap = LinearSegmentedColormap.from_list("splat", [
    (0.0, "#0d0d14"),
    (0.15, "#14122a"),
    (0.35, "#2a1f5e"),
    (0.55, "#4a3a9e"),
    (0.75, "#7b6cc8"),
    (0.90, "#a99de0"),
    (1.0, "#c8bff0"),
])

BG_COLOR = "#1a1a24"
CONTOUR_COLOR = "#8878c0"
AXIS_COLOR = "#d4a843"


def make_gaussian_2d(x, y, mu_x, mu_y, cov):
    """Evaluate 2D Gaussian at grid points given mean and covariance."""
    cov_inv = np.linalg.inv(cov)
    dx = x - mu_x
    dy = y - mu_y
    exponent = -(cov_inv[0, 0] * dx**2 +
                 (cov_inv[0, 1] + cov_inv[1, 0]) * dx * dy +
                 cov_inv[1, 1] * dy**2) / 2.0
    return np.exp(exponent)


def draw_splat(ax, Z, X, Y, mu, cov, contour_levels, show_axes=False, rotation_deg=0):
    """Draw the glowing gaussian with contour ellipses."""
    ax.set_facecolor(BG_COLOR)
    ax.imshow(
        Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower", cmap=splat_cmap, aspect="equal",
        interpolation="gaussian",
    )

    # Contour ellipses
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for level in contour_levels:
        scale = np.sqrt(-2 * np.log(level))  # radius at this density
        w = 2 * scale * np.sqrt(eigenvalues[0])
        h = 2 * scale * np.sqrt(eigenvalues[1])
        ellipse = Ellipse(
            xy=mu, width=w, height=h, angle=angle,
            fill=False, edgecolor=CONTOUR_COLOR, linewidth=1.2, alpha=0.6,
        )
        ax.add_patch(ellipse)

    if show_axes:
        # Draw dashed principal axes through the center
        length_major = 2.2 * np.sqrt(eigenvalues[1])
        length_minor = 1.4 * np.sqrt(eigenvalues[0])
        # Major axis
        dx_maj = length_major * eigenvectors[0, 1]
        dy_maj = length_major * eigenvectors[1, 1]
        ax.plot(
            [mu[0] - dx_maj, mu[0] + dx_maj],
            [mu[1] - dy_maj, mu[1] + dy_maj],
            color=AXIS_COLOR, linewidth=2, linestyle="--", alpha=0.85,
        )
        # Minor axis
        dx_min = length_minor * eigenvectors[0, 0]
        dy_min = length_minor * eigenvectors[1, 0]
        ax.plot(
            [mu[0] - dx_min, mu[0] + dx_min],
            [mu[1] - dy_min, mu[1] + dy_min],
            color=AXIS_COLOR, linewidth=2, linestyle="--", alpha=0.85,
        )

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


res = 600
grid = np.linspace(-4, 4, res)
X, Y = np.meshgrid(grid, grid)
contour_levels = [0.8, 0.5, 0.2, 0.05]

# --- 1. Isotropic Gaussian ---
mu_iso = (0.0, 0.0)
cov_iso = np.array([[1.0, 0.0],
                     [0.0, 1.0]])
Z_iso = make_gaussian_2d(X, Y, *mu_iso, cov_iso)

fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG_COLOR)
draw_splat(ax, Z_iso, X, Y, mu_iso, cov_iso, contour_levels)
plt.savefig("../assets/gaussian_isotropic.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR, pad_inches=0.3)
plt.close()

# --- 2. Anisotropic (stretched + rotated) Gaussian ---
mu_aniso = (0.0, 0.0)
theta = np.radians(55)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.diag([0.4, 2.5])  # stretch: thin along one axis, long along other
cov_aniso = R @ S @ R.T

Z_aniso = make_gaussian_2d(X, Y, *mu_aniso, cov_aniso)

fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG_COLOR)
draw_splat(ax, Z_aniso, X, Y, mu_aniso, cov_aniso, contour_levels, show_axes=True)
plt.savefig("../assets/gaussian_anisotropic.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR, pad_inches=0.3)
plt.close()

print("Saved: gaussian_isotropic.png, gaussian_anisotropic.png")

# --- 3. Projection diagram: 3D Gaussian -> camera -> 2D splat ---
from matplotlib.patches import FancyArrowPatch, Ellipse as EllipsePatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-1, 11)
ax.set_ylim(-1.5, 4.5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

TEXT_COLOR = "#c8bff0"
PURPLE_GLOW = "#7b6cc8"
BLUE_LINE = "#5a8fd4"
GREEN = "#5ec46b"
ORANGE = "#d4a843"

# -- Camera (left side) --
cam_x, cam_y = 0.5, 1.5
# Camera body (trapezoid-ish)
cam_body = plt.Polygon(
    [[cam_x, cam_y - 0.4], [cam_x, cam_y + 0.4],
     [cam_x + 0.6, cam_y + 0.3], [cam_x + 0.6, cam_y - 0.3]],
    facecolor="#3a3555", edgecolor=PURPLE_GLOW, linewidth=1.5,
)
ax.add_patch(cam_body)
# Lens
cam_lens = plt.Circle((cam_x + 0.7, cam_y), 0.15,
                       facecolor="#4a4570", edgecolor=PURPLE_GLOW, linewidth=1.2)
ax.add_patch(cam_lens)
ax.text(cam_x + 0.3, cam_y - 0.75, "Camera", fontsize=10,
        color=TEXT_COLOR, ha="center", fontweight="bold")

# -- Image plane (vertical rectangle, middle-left) --
plane_x = 3.0
plane_y_bot, plane_y_top = 0.0, 3.0
ax.plot([plane_x, plane_x], [plane_y_bot, plane_y_top],
        color=BLUE_LINE, linewidth=2.5, alpha=0.8)
ax.text(plane_x, plane_y_bot - 0.4, "Image Plane", fontsize=9,
        color=BLUE_LINE, ha="center")

# -- 3D Gaussians (right side) --
# Three Gaussians at different "depths"
gaussians = [
    {"cx": 7.5, "cy": 2.5, "w": 1.2, "h": 0.6, "angle": 30, "color": "#6b5bbd", "alpha": 0.7},
    {"cx": 8.5, "cy": 1.0, "w": 0.8, "h": 1.4, "angle": -20, "color": "#8b6cc8", "alpha": 0.6},
    {"cx": 6.0, "cy": 1.8, "w": 0.9, "h": 0.5, "angle": 50, "color": "#5a4a9e", "alpha": 0.8},
]

for g in gaussians:
    # Outer glow
    for s, a in [(1.8, 0.08), (1.4, 0.15), (1.0, 0.3)]:
        ell = EllipsePatch(
            xy=(g["cx"], g["cy"]), width=g["w"] * s, height=g["h"] * s,
            angle=g["angle"], facecolor=g["color"], edgecolor="none", alpha=a,
        )
        ax.add_patch(ell)
    # Core
    ell = EllipsePatch(
        xy=(g["cx"], g["cy"]), width=g["w"] * 0.6, height=g["h"] * 0.6,
        angle=g["angle"], facecolor="#c8bff0", edgecolor="none", alpha=0.4,
    )
    ax.add_patch(ell)

ax.text(7.5, -0.4, "3D Gaussians", fontsize=10,
        color=TEXT_COLOR, ha="center", fontweight="bold")

# -- Projection rays (from gaussians to image plane, converging toward camera) --
ray_alpha = 0.3
for g in gaussians:
    # Ray from gaussian center to a point on the image plane
    # Project: interpolate y position onto the image plane
    t = (plane_x - cam_x - 0.7) / (g["cx"] - cam_x - 0.7)
    proj_y = cam_y + t * (g["cy"] - cam_y)
    # Ray from camera lens to gaussian (through image plane)
    ax.plot([cam_x + 0.7, g["cx"]], [cam_y, g["cy"]],
            color=ORANGE, linewidth=1, alpha=ray_alpha, linestyle="--")
    # Mark where it hits the image plane
    ax.plot(plane_x, proj_y, "o", color=PURPLE_GLOW, markersize=5, alpha=0.8)

# -- 2D splats on the image plane (small ellipses) --
for g in gaussians:
    t = (plane_x - cam_x - 0.7) / (g["cx"] - cam_x - 0.7)
    proj_y = cam_y + t * (g["cy"] - cam_y)
    # Smaller projected splat
    scale = 0.3
    ell_2d = EllipsePatch(
        xy=(plane_x, proj_y), width=g["h"] * scale, height=g["w"] * scale * 0.5,
        angle=g["angle"] * 0.7,
        facecolor=g["color"], edgecolor=CONTOUR_COLOR, linewidth=0.8, alpha=0.6,
    )
    ax.add_patch(ell_2d)

# -- Arrow: "project" label --
ax.annotate(
    "project &\nalpha-blend",
    xy=(plane_x + 0.1, 1.5), xytext=(4.8, 3.8),
    fontsize=9, color=GREEN, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
)

# -- Arrow: differentiable backprop --
ax.annotate(
    "",
    xy=(8.5, -0.8), xytext=(plane_x, -0.8),
    arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=1.8),
)
ax.text(5.7, -1.15, "gradients flow end-to-end", fontsize=9,
        color="#e74c3c", ha="center", style="italic")

plt.savefig("../assets/gaussian_projection.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR, pad_inches=0.4)
plt.close()

print("Saved: gaussian_projection.png")
