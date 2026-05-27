"""
model.py — Shared surrogate model architecture.
Import this everywhere instead of redefining the network.
"""

import torch
import torch.nn as nn


class PlasmaEtchSurrogate(nn.Module):
    """
    A fully-connected surrogate model for plasma etch rate prediction.

    Inputs (4 features):
        x, y       : normalized wafer position in [-1, 1]
        pressure   : chamber pressure in mTorr [10, 100]
        power      : RF power in Watts [500, 2000]

    Output (1 value):
        etch_rate  : predicted etch rate in Angstroms/minute
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(path: str = "lam_plasma_model.pth") -> PlasmaEtchSurrogate:
    """Load a trained surrogate model from disk."""
    model = PlasmaEtchSurrogate()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
