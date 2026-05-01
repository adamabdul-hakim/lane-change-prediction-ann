import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def load_data(data_dir):
    """Load X and y arrays from .npy files inside data_dir."""
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    return X, y


def clean_data(X):
    """Return a float32 array, stripping any comma-formatted string values if present."""
    if X.dtype == np.float32:
        return X
    # vectorised path for string arrays (e.g. "1,234.56" → 1234.56)
    clean = np.vectorize(lambda v: float(str(v).replace(",", "")))(X)
    return clean.astype(np.float32)


def split_data(X, y, test_size=0.2, random_state=42):
    """80/20 train-test split. Returns (X_train, X_test, y_train, y_test)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def normalize_data(X_train, X_test):
    """Z-score normalise using training statistics only to avoid data leakage.

    Returns (X_train_norm, X_test_norm).
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # constant features: avoid division by zero
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def oversample(X_train, y_train, random_state=42):
    """Balance classes by randomly oversampling the minority class.

    Applied to training data only — never to test data.
    Returns (X_resampled, y_resampled) as numpy arrays.
    """
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    return X_res, y_res.astype(np.int64)


def compute_pos_weight(y_train):
    """Return the negative-to-positive ratio as a float for BCEWithLogitsLoss.

    After oversampling this will be ~1.0. Kept as an explicit function so the
    caller can always pass the correct weight regardless of whether oversampling
    was applied.
    """
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    return neg / pos
