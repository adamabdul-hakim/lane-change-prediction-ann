"""
predict.py — load the saved model and run inference on test samples.

Usage:
    python src/predict.py              # shows 10 random test samples
    python src/predict.py --n 20       # shows 20 random test samples
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing import load_data, clean_data
from model import LaneChangeModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR   = Path(__file__).resolve().parent.parent / "data" / "processed"
THRESHOLD  = 0.5


def load_model():
    """Load the trained ANN from models/ann.pt."""
    path = MODELS_DIR / "ann.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at {path}. Run 'python src/train.py' first."
        )
    model = LaneChangeModel()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_norm_params():
    """Load the normalisation mean and std saved during training."""
    path = MODELS_DIR / "norm_params.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No normalisation params found at {path}. Run 'python src/train.py' first."
        )
    data = np.load(path)
    return data["mean"], data["std"]


def predict(model, X_raw, mean, std):
    """Normalise raw samples and return (predictions, confidence scores)."""
    X = (X_raw - mean) / std
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).squeeze(-1).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples to show")
    args = parser.parse_args()

    # --- load model and norm params -----------------------------------------
    print("Loading model...")
    model      = load_model()
    mean, std  = load_norm_params()

    # --- reconstruct the same test split used during training ----------------
    print("Loading data...")
    X, y = load_data(DATA_DIR)
    X = clean_data(X[:200_000])
    y = y[:200_000]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- pick n random samples -----------------------------------------------
    rng     = np.random.default_rng(seed=42)
    indices = rng.choice(len(X_test), size=args.n, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # --- run inference -------------------------------------------------------
    preds, probs = predict(model, X_sample, mean, std)

    # --- print results table -------------------------------------------------
    label = {0: "No Change  ", 1: "Lane Change"}
    correct = sum(p == a for p, a in zip(preds, y_sample))

    print(f"\n{'='*58}")
    print(f"  {'#':<4} {'Actual':<14} {'Predicted':<14} {'Confidence':<12} {'OK?'}")
    print(f"  {'-'*53}")
    for i, (actual, pred, prob) in enumerate(zip(y_sample, preds, probs)):
        match = "correct" if actual == pred else "WRONG"
        print(f"  {i:<4} {label[actual]:<14} {label[pred]:<14} {prob:.4f}       {match}")
    print(f"  {'-'*53}")
    print(f"  Accuracy on these {args.n} samples: {correct}/{args.n}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
