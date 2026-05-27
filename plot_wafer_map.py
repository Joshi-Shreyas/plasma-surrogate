"""
plot_wafer_map.py — Visualize predicted etch rate uniformity across the wafer.

Generates a circular heatmap (wafer_uniformity_map.png) at a fixed process
recipe, showing how etch rate varies spatially due to plasma non-uniformity.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import load_model


RECIPE = {"pressure": 40.0, "power": 800.0}
OUTPUT_PATH = "wafer_uniformity_map.png"
GRID_SIZE   = 100   # increased from 50 for smoother output


def build_wafer_grid(grid_size: int, recipe: dict) -> tuple:
    """Create a meshgrid of wafer positions with fixed process conditions."""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    grid = np.stack([
        X.ravel(),
        Y.ravel(),
        np.full(X.size, recipe["pressure"]),
        np.full(X.size, recipe["power"]),
    ], axis=1).astype(np.float32)

    mask = X**2 + Y**2 > 1.0   # outside wafer circle
    return X, Y, grid, mask


def plot_wafer_map(model: torch.nn.Module) -> None:
    X, Y, grid, mask = build_wafer_grid(GRID_SIZE, RECIPE)

    with torch.no_grad():
        Z = model(torch.tensor(grid)).reshape(GRID_SIZE, GRID_SIZE).numpy()

    Z[mask] = np.nan  # clip to wafer boundary

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap="plasma")
    cbar = fig.colorbar(im, ax=ax, label="Etch Rate (Å/min)")
    cbar.ax.tick_params(labelsize=9)

    # Annotate uniformity stats
    valid = Z[~mask]
    nu = (valid.max() - valid.min()) / (2 * valid.mean()) * 100   # non-uniformity %
    ax.set_title(
        f"Wafer Etch Uniformity Map\n"
        f"{RECIPE['pressure']} mTorr | {RECIPE['power']} W  |  Non-uniformity: {nu:.1f}%",
        fontsize=11
    )
    ax.set_xlabel("Wafer X (normalized)")
    ax.set_ylabel("Wafer Y (normalized)")

    # Draw wafer edge
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "w--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"[plot] Wafer uniformity map saved → {OUTPUT_PATH}")
    print(f"[plot] Non-uniformity (range/2mean): {nu:.2f}%")


if __name__ == "__main__":
    model = load_model()
    plot_wafer_map(model)
