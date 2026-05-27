"""
predict_etch.py — Run inference with the trained plasma surrogate.

Usage:
    python predict_etch.py                    # single test case + accuracy table
    python predict_etch.py --n 20             # evaluate on 20 random samples
"""

import argparse
import torch
from model import load_model


def predict_single(model: torch.nn.Module) -> None:
    """Predict etch rate for a representative process point."""
    # [x=0.0, y=0.0 (wafer center), Pressure=50 mTorr, Power=1200 W]
    test_case = torch.tensor([[0.0, 0.0, 50.0, 1200.0]])

    with torch.no_grad():
        pred = model(test_case).item()

    print(" RESEARCH PLASMA SURROGATE — PREDICTION")
    print(f"  Position  : wafer center (0, 0)")
    print(f"  Pressure  : 50 mTorr")
    print(f"  RF Power  : 1200 W")
    print(f"  Etch Rate : {pred:.2f} Å/min")


def evaluate_samples(model: torch.nn.Module, n: int = 5) -> None:
    """Evaluate model against held-out samples from the dataset."""
    data = torch.load("etch_data.pt", weights_only=False)
    X = torch.tensor(data["inputs"][:n])
    Y = torch.tensor(data["outputs"][:n])

    with torch.no_grad():
        preds = model(X)

    print(f"\n{'Sample':>6} | {'Actual (Å/min)':>14} | {'Predicted':>10} | {'Error %':>8}")
    print("-" * 48)
    for i in range(n):
        a = Y[i].item()
        p = preds[i].item()
        err = abs(a - p) / abs(a) * 100
        print(f"{i+1:>6} | {a:>14.2f} | {p:>10.2f} | {err:>7.2f}%")

    mae  = torch.abs(preds - Y).mean().item()
    mape = (torch.abs(preds - Y) / torch.abs(Y)).mean().item() * 100
    print(f"\n  MAE  : {mae:.2f} Å/min")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  Model Accuracy : {100 - mape:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Plasma etch surrogate inference")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--model", type=str, default="lam_plasma_model.pth")
    args = parser.parse_args()

    model = load_model(args.model)
    predict_single(model)
    evaluate_samples(model, n=args.n)


if __name__ == "__main__":
    main()
