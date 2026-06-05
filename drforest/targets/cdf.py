"""Weighted empirical CDFs from the DRF weight matrix."""

import numpy as np

from drforest.targets._validation import (
    as_response_matrix,
    as_weight_operator,
    weights_apply,
)


def _validate_thresholds(thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = thresholds[None]
    if thresholds.ndim != 1:
        raise ValueError(f"thresholds must be scalar or 1-D; got shape {thresholds.shape}")
    if thresholds.size == 0:
        raise ValueError("thresholds must not be empty")
    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds contain non-finite values")
    return thresholds


def weighted_cdf(W: object, Y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return ``P(Y_j <= t | X=x)`` for each output and threshold.

    The result has shape ``(n_test, d, n_thresholds)``. The same threshold grid is
    evaluated for every response dimension.
    """
    op = as_weight_operator(W)
    Y = as_response_matrix(Y, op.shape[1])
    ts = _validate_thresholds(thresholds)

    n_test, n_outputs = op.shape[0], Y.shape[1]
    out = np.empty((n_test, n_outputs, ts.size), dtype=np.float64)
    for output in range(n_outputs):
        indicators = (Y[:, output, None] <= ts[None, :]).astype(np.float64)
        out[:, output, :] = weights_apply(op, indicators)
    return out
