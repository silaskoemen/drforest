"""Default criterion — MMD via Random Fourier Features.

Scores each split by the scaled MMD of Eq. 12 between the children's response
distributions, estimated with complex Gaussian RFF. Frequencies ``ω_b`` are
resampled per node from the node's RNG (decorrelation source); the bandwidth
``σ`` is fixed for the forest (median heuristic by default).

The ``1/B`` averaging in the MMD estimate is carried as ``scale`` — it is
constant across all splits/features, so it does not affect which split is
chosen, but it keeps reported scores on a kernel-comparable scale.
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

import drforest.criteria._rust as _rust
from drforest.criteria.base import (
    MeanEmbeddingCriterion,
    Split,
    _as_feature_index,
    _best_split_on_feature,
    validate_split_inputs,
)
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

    def best_split(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        features: Sequence[int],
        rng: Generator,
        min_leaf: int,
        threshold_bounds: np.ndarray | None,
        max_cutpoints: int | None = None,
    ) -> Split | None:
        validate_split_inputs(X, Y, features)
        if Y.shape[1] != self.dim:
            raise ValueError(f"Y has {Y.shape[1]} response dimensions; criterion was configured for {self.dim}")
        if min_leaf < 1:
            raise ValueError(f"min_leaf must be >= 1; got {min_leaf}")
        if threshold_bounds is not None:
            threshold_bounds = np.asarray(threshold_bounds, dtype=np.float64)
            if threshold_bounds.shape != (len(features), 2):
                raise ValueError(
                    f"threshold_bounds must have shape ({len(features)}, 2); " f"got {threshold_bounds.shape}"
                )

        psi = self.embed(Y, rng)
        if _rust.rust_available():
            found = _rust.best_complex_embedding_split(
                X,
                psi,
                features,
                scale=self.scale,
                min_leaf=min_leaf,
                threshold_bounds=threshold_bounds,
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                return None
            feature, threshold, score = found
            return Split(feature=feature, threshold=threshold, score=score)

        best: Split | None = None
        for j, f in enumerate(features):
            idx = _as_feature_index(f)
            lo, hi = (-np.inf, np.inf) if threshold_bounds is None else threshold_bounds[j]
            found = _best_split_on_feature(X[:, idx], psi, self.scale, min_leaf, lo, hi, max_cutpoints)
            if found is None:
                continue
            threshold, score = found
            if best is None or score > best.score:
                best = Split(feature=idx, threshold=threshold, score=score)
        return best

    @property
    def scale(self) -> float:
        return 1.0 / self.n_features
