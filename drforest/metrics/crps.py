"""Componentwise CRPS for weighted empirical predictive distributions."""

from collections.abc import Callable

import numpy as np
from scipy.sparse import csr_matrix

from drforest.mixture import MixtureWeights
from drforest.targets._validation import (
    as_materialized_csr_weights,
    as_response_matrix,
)


def _weighted_absolute_pairwise(y: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    w_sorted = w[order]
    prefix_w = np.cumsum(w_sorted) - w_sorted
    prefix_yw = np.cumsum(w_sorted * y_sorted) - w_sorted * y_sorted
    return float(2.0 * np.sum(w_sorted * (y_sorted * prefix_w - prefix_yw)))


def _abs_deviation_profile(atoms: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Return ``f(q) = Σ_j |q - y_j|`` (vectorized) and ``UU = Σ_{jk}|y_j-y_k|/n²``.

    ``f`` answers absolute-deviation queries against the fixed atom set in
    ``O(m log n)`` via sorted prefix sums; ``UU`` is the uniform-target pairwise
    term, computed once as a by-product.
    """
    y_sorted = np.sort(atoms, kind="stable")
    cum = np.cumsum(y_sorted)
    n = y_sorted.size
    total = float(cum[-1])

    def deviation(q: np.ndarray) -> np.ndarray:
        below_count = np.searchsorted(y_sorted, q, side="right")  # #{y_j <= q}
        sum_below = np.where(below_count > 0, cum[np.clip(below_count - 1, 0, n - 1)], 0.0)
        sum_above = total - sum_below
        return q * below_count - sum_below + sum_above - q * (n - below_count)

    uu = float(deviation(y_sorted).sum()) / (n * n)
    return deviation, uu


def _validate_crps_inputs(
    op: MixtureWeights | csr_matrix, Y_train: np.ndarray, Y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Y_train = as_response_matrix(Y_train, op.shape[1])
    Y_true = np.asarray(Y_true, dtype=np.float64)
    if Y_true.shape != (op.shape[0], Y_train.shape[1]):
        raise ValueError(f"Y_true must have shape {(op.shape[0], Y_train.shape[1])}; got {Y_true.shape}")
    if not np.isfinite(Y_true).all():
        raise ValueError("Y_true contains non-finite values")
    return Y_train, Y_true


def _crps_from_csr(W_csr, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
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


def _crps_from_uniform_mixture(mix: MixtureWeights, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Factored CRPS of ``(1-α)·base + α·uniform`` without materializing weights.

    Expands the pairwise term as ``(1-α)²·PP + 2(1-α)α·PU + α²·UU``, where the
    uniform contributions ``PU`` and ``UU`` reuse a per-dimension sorted
    deviation profile over all training atoms.
    """
    base = mix.base
    alpha = mix.alpha
    n_train = mix.n_train
    out = np.empty_like(Y_true, dtype=np.float64)
    for output in range(Y_train.shape[1]):
        y = Y_train[:, output]
        deviation, uu = _abs_deviation_profile(y)
        target_dev = deviation(Y_true[:, output]) / n_train  # a·E_U|Z-y_true| per row
        for row in range(base.shape[0]):
            start, end = base.indptr[row], base.indptr[row + 1]
            cols = base.indices[start:end]
            p = base.data[start:end]
            atoms = y[cols]
            a = alpha[row]

            first = (1.0 - a) * np.sum(p * np.abs(atoms - Y_true[row, output])) + a * target_dev[row]
            pp = _weighted_absolute_pairwise(atoms, p)
            pu = float(np.sum(p * deviation(atoms))) / n_train
            pairwise = (1.0 - a) ** 2 * pp + 2.0 * (1.0 - a) * a * pu + a**2 * uu
            out[row, output] = first - 0.5 * pairwise
    return out


def componentwise_crps(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Return CRPS for each test row and output dimension.

    For an empirical distribution ``Σ_i w_i δ_{y_i}``, CRPS is
    ``Σ_i w_i |y_i - y| - 0.5 Σ_i Σ_j w_i w_j |y_i - y_j|``. A
    :class:`MixtureWeights` with the uniform marginal target is scored through a
    factored path that avoids materializing the dense weight matrix; any other
    mixture target falls back to the materialized CSR computation.
    """
    if isinstance(W, MixtureWeights) and W.is_uniform_marginal:
        Y_train, Y_true = _validate_crps_inputs(W, Y_train, Y_true)
        return _crps_from_uniform_mixture(W, Y_train, Y_true)

    W_csr = as_materialized_csr_weights(W)
    Y_train, Y_true = _validate_crps_inputs(W_csr, Y_train, Y_true)
    return _crps_from_csr(W_csr, Y_train, Y_true)


def mean_crps(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> float:
    """Return mean componentwise CRPS over all test rows and outputs."""
    return float(np.mean(componentwise_crps(W, Y_train, Y_true)))


def energy_score(W: object, Y_train: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Return the multivariate energy score for each test row.

    For a weighted empirical predictive distribution ``Σ_i w_i δ_{y_i}``, the
    energy score is ``Σ_i w_i ||y_i - y|| - 0.5 Σ_i Σ_j w_i w_j ||y_i-y_j||``.
    It reduces to CRPS in one response dimension. A :class:`MixtureWeights`
    input is materialized first (the multivariate norm does not factor across
    dimensions, so there is no sorted-deviation shortcut yet).
    """
    W_csr = as_materialized_csr_weights(W)
    Y_train, Y_true = _validate_crps_inputs(W_csr, Y_train, Y_true)

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
