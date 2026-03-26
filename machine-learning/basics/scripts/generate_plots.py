"""
Generate illustrations for ML basics blog post.
Uses PyTorch for gradient descent and matplotlib for plotting.

Produces:
  1. fitting_progress.png - Shows the line fitting to data over training steps
  2. loss_curve.png       - Shows the loss decreasing over training steps
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})

# --- Data: simple linear relationship with small values ---
# True relationship: y = 2x + 1
torch.manual_seed(42)
x_data = torch.linspace(0, 5, 15).unsqueeze(1)
noise = torch.randn_like(x_data) * 0.4
y_data = 2.0 * x_data + 1.0 + noise

# --- Single neuron (nn.Linear = weight * input + bias) ---
model = nn.Linear(1, 1)
# Start with a clearly wrong initial guess for visual effect
with torch.no_grad():
    model.weight.fill_(-1.0)
    model.bias.fill_(8.0)

optimizer = optim.SGD(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

epochs = 500
losses = []
snapshot_epochs = [0, 5, 20, 80, 499]
snapshots = {}

x_line = torch.linspace(-0.5, 5.5, 100).unsqueeze(1)

for epoch in range(epochs):
    predictions = model(x_data)
    loss = loss_fn(predictions, y_data)
    losses.append(loss.item())

    if epoch in snapshot_epochs:
        with torch.no_grad():
            y_line = model(x_line).squeeze().numpy().copy()
        snapshots[epoch] = {
            "y": y_line,
            "w": model.weight.item(),
            "b": model.bias.item(),
            "loss": loss.item(),
        }

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_np = x_line.squeeze().numpy()
x_data_np = x_data.squeeze().numpy()
y_data_np = y_data.squeeze().numpy()

# --- Plot 1: Fitting progress ---
fig, axes = plt.subplots(1, len(snapshot_epochs), figsize=(20, 4), sharey=True)
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#2980b9"]

for ax, (epoch, snap), color in zip(axes, snapshots.items(), colors):
    ax.scatter(x_data_np, y_data_np, color="#34495e", s=60, zorder=5, label="data")
    ax.plot(x_np, snap["y"], color=color, linewidth=2.5, label="neuron output")
    label = "initial guess" if epoch == 0 else f"epoch {epoch}"
    if epoch == snapshot_epochs[-1]:
        label = f"epoch {epoch} (done)"
    ax.set_title(f"{label}\nw={snap['w']:.2f}  b={snap['b']:.2f}", fontsize=11)
    ax.set_xlabel("x")
    if ax == axes[0]:
        ax.set_ylabel("y")
    ax.set_ylim(y_data_np.min() - 2, y_data_np.max() + 2)

fig.suptitle(
    "A Single Neuron Learning to Fit a Line  (y = weight · x + bias)",
    fontsize=15, fontweight="bold", y=1.04,
)
plt.tight_layout()
plt.savefig("../assets/fitting_progress.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: Loss curve ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(epochs), losses, color="#2980b9", linewidth=2.5)
ax.set_xlabel("Epoch (training step)")
ax.set_ylabel("Mean Squared Error (loss)")
ax.set_title("Loss Decreasing Over Training", fontsize=14, fontweight="bold")

ax.annotate(
    f"start: {losses[0]:.1f}",
    xy=(0, losses[0]),
    xytext=(20, losses[0] * 0.85),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#e74c3c"),
    color="#e74c3c",
)
ax.annotate(
    f"end: {losses[-1]:.2f}",
    xy=(epochs - 1, losses[-1]),
    xytext=(epochs - 60, losses[0] * 0.25),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#2ecc71"),
    color="#2ecc71",
)

plt.tight_layout()
plt.savefig("../assets/loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"True relationship: y = 2.0x + 1.0")
print(f"Learned:           y = {model.weight.item():.2f}x + {model.bias.item():.2f}")
print(f"Final loss:        {losses[-1]:.4f}")
print("Saved: fitting_progress.png, loss_curve.png")

# --- Plot 3: 3D loss surface ---
import numpy as np

# Compute MSE loss for a grid of (weight, bias) values using the same data
w_range = np.linspace(-2, 5, 200)
b_range = np.linspace(-4, 8, 200)
W, B = np.meshgrid(w_range, b_range)

x_np_data = x_data.squeeze().numpy()
y_np_data = y_data.squeeze().numpy()

# MSE = mean((w*x + b - y)^2) for each (w, b) pair
Loss_surface = np.zeros_like(W)
for i in range(len(x_np_data)):
    Loss_surface += (W * x_np_data[i] + B - y_np_data[i]) ** 2
Loss_surface /= len(x_np_data)

# Cap the loss for better visualization (clip extreme peaks)
Loss_surface = np.clip(Loss_surface, 0, 60)

fig = plt.figure(figsize=(12, 7))
ax3d = fig.add_subplot(111, projection="3d")

surf = ax3d.plot_surface(
    W, B, Loss_surface,
    cmap="viridis",
    alpha=0.85,
    edgecolor="none",
)

# Plot the gradient descent path on the surface
path_w = [snapshots[e]["w"] for e in snapshot_epochs]
path_b = [snapshots[e]["b"] for e in snapshot_epochs]
path_loss = [snapshots[e]["loss"] for e in snapshot_epochs]
# Clip path losses to match surface clipping
path_loss_clipped = [min(l, 60) for l in path_loss]

ax3d.plot(
    path_w, path_b, path_loss_clipped,
    color="#e74c3c", linewidth=3, zorder=10, label="gradient descent path",
)
ax3d.scatter(
    path_w, path_b, path_loss_clipped,
    color="#e74c3c", s=60, zorder=11, depthshade=False,
)
# Mark the minimum
ax3d.scatter(
    [path_w[-1]], [path_b[-1]], [path_loss[-1]],
    color="#2ecc71", s=120, zorder=12, marker="*", depthshade=False, label="minimum",
)

ax3d.set_xlabel("Weight (w)", fontsize=12, labelpad=10)
ax3d.set_ylabel("Bias (b)", fontsize=12, labelpad=10)
ax3d.set_zlabel("Loss (MSE)", fontsize=12, labelpad=10)
ax3d.set_title(
    "Loss Surface: How Error Changes with Weight and Bias",
    fontsize=14, fontweight="bold", pad=20,
)
ax3d.view_init(elev=30, azim=-50)
ax3d.legend(loc="upper right", fontsize=10)

fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=15, label="Loss")
plt.tight_layout()
plt.savefig("../assets/loss_surface.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: loss_surface.png")

# --- Plot 4: Complex loss landscape with multiple local minima ---
fig = plt.figure(figsize=(14, 7))

# Create a synthetic landscape with multiple valleys and peaks
x_grid = np.linspace(-4, 4, 300)
y_grid = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x_grid, y_grid)

# Combine several Gaussians (negative = valleys) and base curvature
Z = 0.3 * (X**2 + Y**2)  # gentle bowl base
# Deep global minimum
Z -= 6.0 * np.exp(-((X - 1.5)**2 + (Y - 1.2)**2) / 0.8)
# Shallower local minimum
Z -= 3.5 * np.exp(-((X + 1.8)**2 + (Y - 0.5)**2) / 0.6)
# Another local minimum
Z -= 2.8 * np.exp(-((X + 0.5)**2 + (Y + 2.0)**2) / 0.5)
# Small local minimum (trap)
Z -= 2.0 * np.exp(-((X - 2.5)**2 + (Y + 1.5)**2) / 0.4)
# Add a ridge
Z += 1.5 * np.exp(-((X - 0.2)**2) / 0.3) * np.exp(-((Y + 0.3)**2) / 2.0)

ax1 = fig.add_subplot(121, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap="inferno", alpha=0.9, edgecolor="none")

# Mark the minima
minima = [
    (1.5, 1.2, "global\nminimum", "#2ecc71"),
    (-1.8, 0.5, "local\nminimum", "#e74c3c"),
    (-0.5, -2.0, "local\nminimum", "#e67e22"),
    (2.5, -1.5, "local\nminimum", "#e67e22"),
]
for mx, my, mlabel, mcolor in minima:
    mz = 0.3 * (mx**2 + my**2)
    mz -= 6.0 * np.exp(-((mx - 1.5)**2 + (my - 1.2)**2) / 0.8)
    mz -= 3.5 * np.exp(-((mx + 1.8)**2 + (my - 0.5)**2) / 0.6)
    mz -= 2.8 * np.exp(-((mx + 0.5)**2 + (my + 2.0)**2) / 0.5)
    mz -= 2.0 * np.exp(-((mx - 2.5)**2 + (my + 1.5)**2) / 0.4)
    mz += 1.5 * np.exp(-((mx - 0.2)**2) / 0.3) * np.exp(-((my + 0.3)**2) / 2.0)
    ax1.scatter([mx], [my], [mz - 0.3], color=mcolor, s=100, zorder=15,
                depthshade=False, marker="v")

ax1.set_xlabel("Parameter 1", fontsize=11, labelpad=8)
ax1.set_ylabel("Parameter 2", fontsize=11, labelpad=8)
ax1.set_zlabel("Loss", fontsize=11, labelpad=8)
ax1.set_title("Complex Loss Landscape", fontsize=13, fontweight="bold", pad=15)
ax1.view_init(elev=35, azim=-60)

# --- Contour view (top-down) ---
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=40, cmap="inferno")
ax2.contour(X, Y, Z, levels=20, colors="white", linewidths=0.3, alpha=0.4)

for mx, my, mlabel, mcolor in minima:
    ax2.plot(mx, my, marker="v", markersize=12, color=mcolor, markeredgecolor="white",
             markeredgewidth=1.5, zorder=10)
    ax2.annotate(mlabel, xy=(mx, my), xytext=(mx + 0.35, my + 0.35),
                 fontsize=9, color="white", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

ax2.set_xlabel("Parameter 1", fontsize=11)
ax2.set_ylabel("Parameter 2", fontsize=11)
ax2.set_title("Top-Down View (Contour Map)", fontsize=13, fontweight="bold")
fig.colorbar(contour, ax=ax2, shrink=0.85, label="Loss")

plt.suptitle(
    "Why Local Minima Are a Problem in Deep Learning",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("../assets/local_minima.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: local_minima.png")
