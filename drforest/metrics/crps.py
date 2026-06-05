"""Componentwise CRPS for weighted empirical predictive distributions."""

import numpy as np

from drforest.targets._validation import (
    as_csr_weights,
    as_response_matrix,
    normalize_rows,
)


def _weighted_absolute_pairwise(y: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    w_sorted = w[order]
    prefix_w = np.cumsum(w_sorted) - w_sorted
    prefix_yw = np.cumsum(w_sorted * y_sorted) - w_sorted * y_sorted
    return float(2.0 * np.sum(w_sorted * (y_sorted * prefix_w - prefix_yw)))


def componentwise_crps(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Return CRPS for each test row and output dimension.

    For an empirical distribution ``Σ_i w_i δ_{y_i}``, CRPS is
    ``Σ_i w_i |y_i - y| - 0.5 Σ_i Σ_j w_i w_j |y_i - y_j|``.
    """
    W_csr = normalize_rows(as_csr_weights(W))
    Y_train = as_response_matrix(Y_train, W_csr.shape[1])
    Y_true = np.asarray(Y_true, dtype=np.float64)
    if Y_true.shape != (W_csr.shape[0], Y_train.shape[1]):
        raise ValueError(f"Y_true must have shape {(W_csr.shape[0], Y_train.shape[1])}; got {Y_true.shape}")
    if not np.isfinite(Y_true).all():
        raise ValueError("Y_true contains non-finite values")

    out = np.empty_like(Y_true, dtype=np.float64)
    for row in range(W_csr.shape[0]):
        start, end = W_csr.indptr[row], W_csr.indptr[row + 1]
        cols = W_csr.indices[start:end]
        weights = W_csr.data[start:end]
        for output in range(Y_train.shape[1]):
            atoms = Y_train[cols, output]
            first = np.sum(weights * np.abs(atoms - Y_true[row, output]))
            second = 0.5 * _weighted_absolute_pairwise(atoms, weights)
            out[row, output] = first - second
    return out


def mean_crps(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> float:
    """Return mean componentwise CRPS over all test rows and outputs."""
    return float(np.mean(componentwise_crps(W, Y_train, Y_true)))


def energy_score(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Return the multivariate energy score for each test row.

    For a weighted empirical predictive distribution ``Σ_i w_i δ_{y_i}``, the
    energy score is ``Σ_i w_i ||y_i - y|| - 0.5 Σ_i Σ_j w_i w_j ||y_i-y_j||``.
    It reduces to CRPS in one response dimension.
    """
    W_csr = normalize_rows(as_csr_weights(W))
    Y_train = as_response_matrix(Y_train, W_csr.shape[1])
    Y_true = np.asarray(Y_true, dtype=np.float64)
    if Y_true.shape != (W_csr.shape[0], Y_train.shape[1]):
        raise ValueError(f"Y_true must have shape {(W_csr.shape[0], Y_train.shape[1])}; got {Y_true.shape}")
    if not np.isfinite(Y_true).all():
        raise ValueError("Y_true contains non-finite values")

    out = np.empty(W_csr.shape[0], dtype=np.float64)
    for row in range(W_csr.shape[0]):
        start, end = W_csr.indptr[row], W_csr.indptr[row + 1]
        cols = W_csr.indices[start:end]
        weights = W_csr.data[start:end]
        atoms = Y_train[cols]
        first = np.sum(weights * np.linalg.norm(atoms - Y_true[row], axis=1))
        pairwise = np.linalg.norm(atoms[:, None, :] - atoms[None, :, :], axis=2)
        second = 0.5 * np.sum(weights[:, None] * weights[None, :] * pairwise)
        out[row] = first - second
    return out


def mean_energy_score(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> float:
    """Return the mean multivariate energy score over test rows."""
    return float(np.mean(energy_score(W, Y_train, Y_true)))
