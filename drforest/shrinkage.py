"""Post-hoc shrinkage transforms on the DRF weight simplex."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from drforest.features.rff import GaussianRFF
from drforest.mixture import MixtureWeights
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

    weights: MixtureWeights
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


def shrink(
    W: object,
    Y: np.ndarray,
    *,
    rff: GaussianRFF,
    target: str = "marginal",
    parameterization: str = "kmse",
) -> ShrinkageResult:
    """Shrink rows of ``W`` toward a target distribution on the same atoms.

    Both closed forms estimate the bias-variance optimum ``α* = V / (V + D²)``
    with the RFF-pinned variance ``V = (1 - ‖μ̂‖²) / n_eff`` (k(y,y)=1). They
    differ only in how the squared bias ``D² = ‖μ*-μ₀‖²`` is plugged in:

    - ``"kmse"`` uses the raw empirical ``MMD² = ‖μ̂-μ₀‖²`` (kernel-mean
      shrinkage form, Muandet et al. 2016)::

          α = (1 - ‖μ̂‖²) / ((1 - ‖μ̂‖²) + n_eff · MMD²).

    - ``"stein"`` uses the bias-corrected ``D̂² = MMD² - V`` (E[MMD²] = D² + V),
      collapsing to the positive-part James–Stein form::

          α = V / MMD² = (1 - ‖μ̂‖²) / (n_eff · MMD²).

    The two agree when ``MMD² ≫ V`` (strong conditional signal) and diverge only
    when ``MMD² ≈ V`` (weak signal), where ``"stein"`` shrinks more aggressively.

    Only the marginal target is implemented for milestone 1. The kernel geometry
    is fixed by the caller-supplied ``rff`` map; no bandwidth or feature-count
    defaults are chosen inside this transform.
    """
    if target != "marginal":
        raise ValueError(f"unsupported shrinkage target {target!r}; only 'marginal' is implemented")
    if parameterization not in ("kmse", "stein"):
        raise ValueError(f"parameterization must be 'kmse' or 'stein'; got {parameterization!r}")

    W_csr = _normalize_rows(_as_csr_weights(W))
    target_weights = marginal_target(W_csr.shape[1])
    variance = np.maximum(1.0 - embedding_norm_sq(W_csr, Y, rff), 0.0)
    distance = mmd_to_target(W_csr, target_weights, Y, rff)
    scaled_distance = n_eff(W_csr) * distance
    # kmse: V/(V+MMD²) → variance/(variance + n_eff·MMD²);
    # stein: V/MMD²     → variance/(n_eff·MMD²)  (bias-corrected denominator).
    denominator = scaled_distance if parameterization == "stein" else variance + scaled_distance
    # denominator == 0 is a limit, not a guard: with positive variance it means
    # MMD² -> 0 (conditional indistinguishable from target) so α -> 1; with zero
    # variance the row is degenerate and α -> 0. Both reduce to 1{variance > 0}.
    limit = (variance > 0.0).astype(np.float64)
    alpha = np.divide(variance, denominator, out=limit, where=denominator > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return ShrinkageResult(
        weights=MixtureWeights(base=W_csr, alpha=alpha, target=target_weights),
        alpha=alpha,
        target_weights=target_weights,
    )
