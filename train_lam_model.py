"""
train_lam_model.py — Train the plasma etch surrogate model.

Features:
  - 80/20 train/validation split
  - Early stopping on validation loss
  - Loss curve saved as training_loss_curve.png
  - Best model checkpoint saved automatically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PlasmaEtchSurrogate


# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS       = 1500
LR           = 1e-3
BATCH_SIZE   = 256
VAL_SPLIT    = 0.2
PATIENCE     = 100   # early stopping patience
MODEL_PATH   = "lam_plasma_model.pth"
CURVE_PATH   = "training_loss_curve.png"
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str = "etch_data.pt"):
    data = torch.load(path, weights_only=False)
    X = torch.tensor(data["inputs"])
    Y = torch.tensor(data["outputs"])
    return X, Y


def train_val_split(X, Y, val_fraction: float = VAL_SPLIT, seed: int = 42):
    n = len(X)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=rng)
    split = int(n * (1 - val_fraction))
    return X[idx[:split]], Y[idx[:split]], X[idx[split:]], Y[idx[split:]]


def make_batches(X, Y, batch_size: int):
    idx = torch.randperm(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], Y[b]


def plot_loss_curve(train_losses, val_losses, path: str = CURVE_PATH):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(train_losses, label="Train Loss", linewidth=1.8, color="#0ea5e9")
    ax.semilogy(val_losses,   label="Val Loss",   linewidth=1.8, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Plasma Surrogate — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Loss curve saved → {path}")


def train():
    X, Y = load_dataset()
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y)
    print(f"[train] Dataset: {len(X_train)} train / {len(X_val)} val samples")

    model    = PlasmaEtchSurrogate()
    loss_fn  = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    best_val_loss   = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print("[train] Starting training …")
    for epoch in range(1, EPOCHS + 1):
        # ── Training pass ──
        model.train()
        epoch_loss = 0.0
        for xb, yb in make_batches(X_train, Y_train, BATCH_SIZE):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(X_train) // BATCH_SIZE)

        # ── Validation pass ──
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        # ── Checkpoint & early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"[train] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"[train] Best val loss: {best_val_loss:.4f} → model saved to {MODEL_PATH}")
    plot_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    train()
