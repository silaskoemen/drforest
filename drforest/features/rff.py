"""Gaussian Random Fourier Features in complex form.

We use the complex feature map ``φ̃_ω(y) = exp(i ωᵀ y)`` with frequencies
``ω ~ N(0, σ⁻² I_d)``. Bochner's theorem then gives, in expectation,

    E_ω[ φ̃_ω(u) · conj(φ̃_ω(v)) ] = exp(-‖u−v‖² / (2σ²)) = k(u, v),

the Gaussian kernel of bandwidth ``σ``. The Monte-Carlo estimate over ``B``
frequencies,

    (1/B) Σ_b φ̃_{ω_b}(u) · conj(φ̃_{ω_b}(v))  →  k(u, v)   as B → ∞,

is real in expectation and unbiased — no random phase offset (the usual
``√2 cos(ωᵀy + b)`` trick) is needed. The complex form is also what makes the
streaming MMD split criterion a single running complex sum.

Frequencies are resampled *per node* (a deliberate decorrelation source), so
``sample_rff`` takes an explicit ``Generator``. The bandwidth ``σ`` is a global
property of the response distribution and is chosen by a pluggable
``BandwidthRule`` (default: the median heuristic).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class GaussianRFF:
    """A fixed set of sampled frequencies realising one Gaussian RFF map."""

    omega: np.ndarray  # (n_features, d) real frequencies
    sigma: float

    @property
    def n_features(self) -> int:
        return self.omega.shape[0]

    @property
    def dim(self) -> int:
        return self.omega.shape[1]

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Map responses ``Y`` (n, d) to complex features (n, n_features)."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        if Y.shape[1] != self.dim:
            raise ValueError(f"Y has dimension {Y.shape[1]}, frequencies expect {self.dim}")
        return np.exp(1j * (Y @ self.omega.T))


def sample_rff(dim: int, n_features: int, sigma: float, rng: Generator) -> GaussianRFF:
    """Draw ``n_features`` Gaussian RFF frequencies for bandwidth ``sigma``."""
    if dim <= 0:
        raise ValueError(f"dim must be positive; got {dim}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive; got {n_features}")
    omega = rng.normal(loc=0.0, scale=1.0 / sigma, size=(n_features, dim))
    return GaussianRFF(omega=omega, sigma=float(sigma))


class BandwidthRule(Protocol):
    """Strategy mapping a response sample to a Gaussian kernel bandwidth σ."""

    def __call__(self, Y: np.ndarray) -> float: ...


def median_heuristic(Y: np.ndarray) -> float:
    """σ = median pairwise Euclidean distance of the responses ``Y``."""
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
    if Y.shape[0] < 2:
        raise ValueError("median heuristic needs at least 2 samples")
    sigma = float(np.median(pdist(Y, metric="euclidean")))
    if sigma <= 0:
        raise ValueError("degenerate Y: median pairwise distance is 0")
    return sigma


def fixed_bandwidth(sigma: float) -> BandwidthRule:
    """A constant-σ rule, for tests and for the tunable spectral-measure dial."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    value = float(sigma)

    def rule(Y: np.ndarray) -> float:
        return value

    return rule
