"""
sensitivity_analysis.py — Quantify how process parameters affect etch rate.

Runs two analyses:
  1. Local sensitivity: +10% perturbation from a baseline recipe
  2. Parameter sweep : etch rate vs. pressure and vs. power across their full ranges
      saved as sensitivity_sweep.png
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import load_model


BASELINE = {"x": 0.0, "y": 0.0, "pressure": 55.0, "power": 1250.0}
SWEEP_PATH = "sensitivity_sweep.png"


def to_tensor(x, y, pressure, power):
    return torch.tensor([[x, y, pressure, power]], dtype=torch.float32)


def local_sensitivity(model):
    """Compute first-order sensitivity via +10% perturbation."""
    b = BASELINE
    base_input = to_tensor(**b)

    with torch.no_grad():
        base_rate = model(base_input).item()

    results = {}
    for param, delta_frac in [("pressure", 0.10), ("power", 0.10)]:
        perturbed = {**b, param: b[param] * (1 + delta_frac)}
        with torch.no_grad():
            perturbed_rate = model(to_tensor(**perturbed)).item()
        sensitivity = (perturbed_rate - base_rate) / base_rate * 100
        results[param] = sensitivity

    print("\n Local Sensitivity Analysis")
    print(f"  Baseline recipe : {b['pressure']} mTorr | {b['power']} W | Center wafer")
    print(f"  Base etch rate  : {base_rate:.2f} Å/min")
    print(f"  +10% Pressure   : {results['pressure']:+.2f}% change in etch rate")
    print(f"  +10% RF Power   : {results['power']:+.2f}% change in etch rate")
    dominant = "Power" if abs(results["power"]) > abs(results["pressure"]) else "Pressure"
    print(f"  → Process is {dominant}-Sensitive")

    return results


def parameter_sweep(model):
    """Sweep pressure and power independently and plot results."""
    b = BASELINE
    pressures = np.linspace(10, 100, 80)
    powers    = np.linspace(500, 2000, 80)

    with torch.no_grad():
        rates_pressure = [
            model(to_tensor(b["x"], b["y"], p, b["power"])).item()
            for p in pressures
        ]
        rates_power = [
            model(to_tensor(b["x"], b["y"], b["pressure"], pw)).item()
            for pw in powers
        ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(pressures, rates_pressure, color="#0ea5e9", linewidth=2)
    ax1.axvline(b["pressure"], color="gray", linestyle="--", alpha=0.6, label=f"Baseline {b['pressure']} mTorr")
    ax1.set_xlabel("Pressure (mTorr)")
    ax1.set_ylabel("Predicted Etch Rate (Å/min)")
    ax1.set_title("Sensitivity to Chamber Pressure")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(powers, rates_power, color="#f97316", linewidth=2)
    ax2.axvline(b["power"], color="gray", linestyle="--", alpha=0.6, label=f"Baseline {b['power']} W")
    ax2.set_xlabel("RF Power (W)")
    ax2.set_ylabel("Predicted Etch Rate (Å/min)")
    ax2.set_title("Sensitivity to RF Power")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Plasma Etch Surrogate — Parameter Sensitivity Sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(SWEEP_PATH, dpi=150)
    plt.close(fig)
    print(f"[sensitivity] Sweep plots saved → {SWEEP_PATH}")


if __name__ == "__main__":
    model = load_model()
    local_sensitivity(model)
    parameter_sweep(model)
