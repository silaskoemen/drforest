"""Anisotropic MMD via diagonal-bandwidth Random Fourier Features."""

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

import drforest.criteria._rust as _rust
from drforest.criteria.base import (
    Criterion,
    Split,
    _as_feature_index,
    _best_split_on_feature,
    validate_split_inputs,
)
from drforest.features.rff import DiagonalBandwidthRule, sample_diagonal_rff


class AnisotropicMmdCriterion(Criterion):
    """MMD criterion with a diagonal Gaussian kernel bandwidth."""

    def __init__(self, n_features: int, bandwidths: np.ndarray) -> None:
        bandwidths = np.asarray(bandwidths, dtype=np.float64)
        if n_features <= 0:
            raise ValueError(f"n_features must be positive; got {n_features}")
        if bandwidths.ndim != 1:
            raise ValueError(f"bandwidths must be 1-D; got shape {bandwidths.shape}")
        if bandwidths.shape[0] == 0:
            raise ValueError("bandwidths must be non-empty")
        if not np.all(bandwidths > 0.0):
            raise ValueError("all bandwidths must be positive")
        self.n_features = int(n_features)
        self.bandwidths = bandwidths.copy()
        self.dim = int(bandwidths.shape[0])

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        n_features: int,
        bandwidth_rule: DiagonalBandwidthRule,
    ) -> "AnisotropicMmdCriterion":
        """Configure the criterion, fixing diagonal bandwidths from training responses."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        bandwidths = np.asarray(bandwidth_rule(Y), dtype=np.float64)
        if bandwidths.shape != (Y.shape[1],):
            raise ValueError(f"bandwidth_rule returned shape {bandwidths.shape}; expected ({Y.shape[1]},)")
        return cls(n_features=n_features, bandwidths=bandwidths)

    def embed(self, Y: np.ndarray, rng: Generator) -> np.ndarray:
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        if Y.shape[1] != self.dim:
            raise ValueError(f"Y has {Y.shape[1]} response dimensions; criterion was configured for {self.dim}")
        rff = sample_diagonal_rff(self.bandwidths, self.n_features, rng)
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
