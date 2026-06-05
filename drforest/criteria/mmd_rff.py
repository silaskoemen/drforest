"""Default criterion — MMD via Random Fourier Features.

Scores each split by the scaled MMD of Eq. 12 between the children's response
distributions, estimated with complex Gaussian RFF. Frequencies ``ω_b`` are
resampled per node from the node's RNG (decorrelation source); the bandwidth
``σ`` is fixed for the forest (median heuristic by default).

The ``1/B`` averaging in the MMD estimate is carried as ``scale`` — it is
constant across all splits/features, so it does not affect which split is
chosen, but it keeps reported scores on a kernel-comparable scale.
"""

import numpy as np
from numpy.random import Generator

from drforest.criteria.base import MeanEmbeddingCriterion
from drforest.features.rff import BandwidthRule, sample_rff


class MmdRffCriterion(MeanEmbeddingCriterion):
    def __init__(self, n_features: int, sigma: float, dim: int) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive; got {n_features}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self.n_features = int(n_features)
        self.sigma = float(sigma)
        self.dim = int(dim)

    @classmethod
    def from_data(cls, Y: np.ndarray, n_features: int, bandwidth_rule: BandwidthRule) -> "MmdRffCriterion":
        """Configure the criterion, fixing σ from the training responses."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        return cls(n_features=n_features, sigma=bandwidth_rule(Y), dim=Y.shape[1])

    def embed(self, Y: np.ndarray, rng: Generator) -> np.ndarray:
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        rff = sample_rff(self.dim, self.n_features, self.sigma, rng)
        return rff.transform(Y)

    @property
    def scale(self) -> float:
        return 1.0 / self.n_features
