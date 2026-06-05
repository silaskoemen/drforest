"""Weighted empirical quantiles from the DRF weight matrix."""

import numpy as np

from drforest.targets._validation import (
    as_materialized_csr_weights,
    as_response_matrix,
)


def _validate_quantiles(quantiles: np.ndarray) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim == 0:
        quantiles = quantiles[None]
    if quantiles.ndim != 1:
        raise ValueError(f"quantiles must be scalar or 1-D; got shape {quantiles.shape}")
    if quantiles.size == 0:
        raise ValueError("quantiles must not be empty")
    if not np.isfinite(quantiles).all():
        raise ValueError("quantiles contain non-finite values")
    if ((quantiles < 0.0) | (quantiles > 1.0)).any():
        raise ValueError("quantiles must lie in [0, 1]")
    return quantiles


def weighted_quantile(W: object, Y: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return weighted empirical quantiles.

    The result has shape ``(n_test, d, n_quantiles)``. Quantiles use the inverse
    empirical CDF convention: the smallest response value whose cumulative weight
    is at least ``q``. A :class:`MixtureWeights` input is materialized first.
    """
    W_csr = as_materialized_csr_weights(W)
    Y = as_response_matrix(Y, W_csr.shape[1])
    qs = _validate_quantiles(quantiles)

    n_test, n_outputs = W_csr.shape[0], Y.shape[1]
    out = np.empty((n_test, n_outputs, qs.size), dtype=np.float64)
    for row in range(n_test):
        start, end = W_csr.indptr[row], W_csr.indptr[row + 1]
        cols = W_csr.indices[start:end]
        weights = W_csr.data[start:end]
        positive = weights > 0.0
        cols = cols[positive]
        weights = weights[positive]
        for output in range(n_outputs):
            atoms = Y[cols, output]
            order = np.argsort(atoms, kind="stable")
            y_sorted = atoms[order]
            cdf = np.cumsum(weights[order])
            indices = np.searchsorted(cdf, qs, side="left")
            indices = np.minimum(indices, y_sorted.size - 1)
            out[row, output, :] = y_sorted[indices]
    return out
