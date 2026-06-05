"""Root mean squared error for conditional-mean predictions."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return scalar RMSE over all samples and output dimensions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"inputs must be 2-D (n, d); got shape {y_true.shape}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
