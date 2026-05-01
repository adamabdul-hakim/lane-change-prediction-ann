import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import (
    load_data,
    clean_data,
    split_data,
    normalize_data,
    oversample,
    compute_pos_weight,
)
from model import LaneChangeModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_SAMPLES  = 200_000   # subset of full dataset used for training
BATCH_SIZE = 512
EPOCHS     = 30
LR         = 0.001
THRESHOLD  = 0.5       # decision threshold for converting probabilities → labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():

    # --- 1. Load and preprocess data ----------------------------------------
    print("Loading data...")
    X, y = load_data(DATA_DIR)

    # use a fixed subset so training stays tractable
    X = X[:N_SAMPLES]
    y = y[:N_SAMPLES]

    # strip any comma-formatted strings; no-op if data is already float32
    X = clean_data(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # normalise using training statistics only (avoids test-set leakage)
    # mean and std are saved so predict.py can normalise new samples identically
    X_train, X_test, norm_mean, norm_std = normalize_data(X_train, X_test)

    # oversample minority class on training data only
    X_train, y_train = oversample(X_train, y_train)
    print(
        f"After resampling: {X_train.shape[0]:,} training samples  "
        f"| class distribution: {np.bincount(y_train)}"
    )

    # pos_weight ≈ 1.0 after oversampling; computed explicitly for correctness
    pos_weight = torch.tensor(compute_pos_weight(y_train), dtype=torch.float32)

    # --- 2. Convert to PyTorch tensors ----------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # --- 3. Model, loss, optimiser -------------------------------------------
    model     = LaneChangeModel()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- 4. Mini-batch training -----------------------------------------------
    print(f"\nTraining ANN ({EPOCHS} epochs, batch size {BATCH_SIZE})...")
    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()   # [batch] raw logits
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>2}/{EPOCHS}  avg loss: {avg_loss:.4f}")

    # --- 5. ANN evaluation ---------------------------------------------------
    model.eval()
    with torch.no_grad():
        ann_probs = torch.sigmoid(model(X_test_t)).squeeze()

    y_true   = y_test_t.numpy()
    ann_preds = (ann_probs >= THRESHOLD).float().numpy()

    ann_precision = precision_score(y_true, ann_preds)
    ann_recall    = recall_score(y_true, ann_preds)
    ann_f1        = f1_score(y_true, ann_preds)

    # --- 6. Logistic regression baseline ------------------------------------
    print("\nTraining Logistic Regression baseline...")
    X_train_np = X_train_t.numpy()
    X_test_np  = X_test_t.numpy()
    y_train_np = y_train_t.numpy()
    y_test_np  = y_test_t.numpy()

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
    log_reg.fit(X_train_np, y_train_np)
    lr_preds = log_reg.predict(X_test_np)

    lr_precision = precision_score(y_test_np, lr_preds)
    lr_recall    = recall_score(y_test_np, lr_preds)
    lr_f1        = f1_score(y_test_np, lr_preds)

    # --- 7. Save model and normalisation parameters --------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "ann.pt"
    torch.save(model.state_dict(), model_path)
    norm_path = MODELS_DIR / "norm_params.npz"
    np.savez(norm_path, mean=norm_mean, std=norm_std)
    print(f"\nModel saved  -> {model_path}")
    print(f"Norm params  -> {norm_path}")

    # --- 8. Print comparison -------------------------------------------------
    print("\n" + "=" * 40)
    print(f"{'Metric':<12} {'ANN':>10} {'Logistic Reg':>14}")
    print("-" * 40)
    print(f"{'Precision':<12} {ann_precision:>10.4f} {lr_precision:>14.4f}")
    print(f"{'Recall':<12} {ann_recall:>10.4f} {lr_recall:>14.4f}")
    print(f"{'F1 Score':<12} {ann_f1:>10.4f} {lr_f1:>14.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
