"""Validation helpers for targets computed from the DRF weight matrix."""

import numpy as np
from scipy.sparse import csr_matrix, issparse

from drforest.mixture import MixtureWeights


def as_csr_weights(W: object) -> csr_matrix:
    """Return ``W`` as CSR and reject malformed weight matrices."""
    if not issparse(W):
        W = csr_matrix(np.asarray(W, dtype=np.float64))
    else:
        W = csr_matrix(W)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D; got shape {W.shape}")
    if W.data.size and not np.isfinite(W.data).all():
        raise ValueError("W contains non-finite weights")
    if W.data.size and (W.data < 0.0).any():
        raise ValueError("W contains negative weights")
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    if not (row_sums > 0.0).all():
        raise ValueError("every row of W must have positive total weight")
    return W


def as_response_matrix(Y: np.ndarray, n_rows: int) -> np.ndarray:
    """Return ``Y`` as a finite ``(n, d)`` float array aligned with ``W`` columns."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
    if Y.shape[0] != n_rows:
        raise ValueError(f"W has {n_rows} columns but Y has {Y.shape[0]} rows")
    if Y.shape[1] == 0:
        raise ValueError("Y has zero response dimensions (d == 0)")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains non-finite values")
    return Y


def normalize_rows(W: csr_matrix) -> csr_matrix:
    """Normalize nonnegative CSR rows to sum to one."""
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = 1.0 / row_sums
    return W.multiply(inv[:, None]).tocsr()


def as_weight_operator(W: object) -> MixtureWeights | csr_matrix:
    """Return a ``MixtureWeights`` as-is, else a row-normalized CSR matrix."""
    if isinstance(W, MixtureWeights):
        return W
    return normalize_rows(as_csr_weights(W))


def weights_apply(op: MixtureWeights | csr_matrix, M: np.ndarray) -> np.ndarray:
    """Linear action ``op @ M`` for either representation, without densifying."""
    if isinstance(op, MixtureWeights):
        return op.apply(M)
    return op @ M


def as_materialized_csr_weights(W: object) -> csr_matrix:
    """Return ``W`` as an explicit row-normalized CSR, materializing a mixture."""
    if isinstance(W, MixtureWeights):
        return W.to_csr()
    return normalize_rows(as_csr_weights(W))
