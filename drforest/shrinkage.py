"""Post-hoc shrinkage transforms on the DRF weight simplex."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from drforest.features.rff import GaussianRFF
from drforest.weights import (
    _as_csr_weights,
    _normalize_rows,
    embedding_norm_sq,
    mmd_to_target,
    n_eff,
)


@dataclass(frozen=True)
class ShrinkageResult:
    """Shrunk weights plus the row-wise intensity used to form them."""

    weights: csr_matrix
    alpha: np.ndarray
    target_weights: csr_matrix


def marginal_target(n_train: int) -> csr_matrix:
    """Uniform marginal target on the training atoms, shape ``(1, n_train)``."""
    if isinstance(n_train, bool) or not isinstance(n_train, Integral):
        raise TypeError(f"n_train must be an integer, not {type(n_train).__name__}: {n_train!r}")
    n_train = int(n_train)
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1; got {n_train}")
    return csr_matrix(np.full((1, n_train), 1.0 / n_train, dtype=np.float64))


def _marginal_shrunk_weights(W: csr_matrix, alpha: np.ndarray) -> csr_matrix:
    n_test, n_train = W.shape
    retained = W.multiply((1.0 - alpha)[:, None]).tocsr()

    rows = np.repeat(np.arange(n_test, dtype=np.int64), n_train)
    cols = np.tile(np.arange(n_train, dtype=np.int64), n_test)
    data = np.repeat(alpha / n_train, n_train)
    target_part = csr_matrix((data, (rows, cols)), shape=W.shape)

    out = retained + target_part
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def shrink(
    W: object,
    Y: np.ndarray,
    *,
    rff: GaussianRFF,
    target: str = "marginal",
) -> ShrinkageResult:
    """Shrink rows of ``W`` toward a target distribution on the same atoms.

    The closed-form intensity is

    ``α = (1 - ‖μ̂‖²) / ((1 - ‖μ̂‖²) + n_eff · MMD²(P̂_x, P̂_target))``.

    Only the marginal target is implemented for milestone 1. The kernel geometry
    is fixed by the caller-supplied ``rff`` map; no bandwidth or feature-count
    defaults are chosen inside this transform.
    """
    if target != "marginal":
        raise ValueError(f"unsupported shrinkage target {target!r}; only 'marginal' is implemented")

    W_csr = _normalize_rows(_as_csr_weights(W))
    target_weights = marginal_target(W_csr.shape[1])
    variance = np.maximum(1.0 - embedding_norm_sq(W_csr, Y, rff), 0.0)
    distance = mmd_to_target(W_csr, target_weights, Y, rff)
    denominator = variance + n_eff(W_csr) * distance
    alpha = np.divide(variance, denominator, out=np.zeros_like(variance), where=denominator > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return ShrinkageResult(
        weights=_marginal_shrunk_weights(W_csr, alpha),
        alpha=alpha,
        target_weights=target_weights,
    )
