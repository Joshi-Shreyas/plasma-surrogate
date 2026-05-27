"""
generate_etch_data.py — Synthetic plasma etch dataset generator.

Simulates a realistic LAM-style plasma etch process with:
  - Spatial non-uniformity (edge effects, center hotspot)
  - Chamber noise
  - Cross-term coupling between pressure and power
"""

import numpy as np
import torch


def generate_data(n_samples: int = 2000, seed: int = 42) -> None:
    """
    Generate synthetic etch rate data and save to etch_data.pt.

    Physics model (simplified):
        etch_rate = power_contribution
                  + pressure_contribution
                  + spatial_nonuniformity
                  + cross_coupling
                  + gaussian_noise
    """
    rng = np.random.default_rng(seed)

    # Process parameters
    pos = rng.uniform(-1, 1, (n_samples, 2))          # (x, y) wafer coords
    pressure = rng.uniform(10, 100, (n_samples, 1))   # mTorr
    power = rng.uniform(500, 2000, (n_samples, 1))    # Watts

    x, y = pos[:, 0:1], pos[:, 1:2]
    r2 = x**2 + y**2  # radial distance squared

    # Core etch rate
    power_term    = power * 0.5
    pressure_term = pressure * 2.0

    # Spatial non-uniformity: edge depletion + center hotspot
    spatial_term  = np.sin(x * 3) * 50 - r2 * 30

    # Cross-coupling: high power amplifies pressure sensitivity
    coupling_term = (power / 1000.0) * (pressure * 0.3)

    # Gaussian process noise (~3% of signal)
    etch_rate = power_term + pressure_term + spatial_term + coupling_term
    noise = rng.normal(0, etch_rate.std() * 0.03, etch_rate.shape)
    etch_rate += noise

    inputs  = np.hstack([pos, pressure, power]).astype(np.float32)
    outputs = etch_rate.astype(np.float32)

    torch.save({"inputs": inputs, "outputs": outputs}, "etch_data.pt")
    print(f"[generate] {n_samples} samples saved to etch_data.pt")
    print(f"[generate] Etch rate range: {outputs.min():.1f} – {outputs.max():.1f} A/min")


if __name__ == "__main__":
    generate_data()
