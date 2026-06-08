"""CART criterion — the identity-kernel correctness anchor.

With the identity kernel ``k(y, y') = ⟨y, y'⟩`` the embedding is just ``Ψ = Y``,
so the shared between-group scatter score becomes

    (n_L n_R / n²) · ‖ ȳ_L − ȳ_R ‖²,

which has the same argmax as multivariate variance reduction (between-group
sum of squares, Eq. 13). This makes CART the reference the RFF criterion is
tested against, and the mean-only baseline.
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

import drforest.criteria._rust as _rust
from drforest.criteria.base import MeanEmbeddingCriterion, Split, validate_split_inputs


class CartCriterion(MeanEmbeddingCriterion):
    def embed(self, Y: np.ndarray, rng: Generator) -> np.ndarray:
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        return np.ascontiguousarray(Y, dtype=np.float64)

    @property
    def scale(self) -> float:
        return 1.0

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
        if min_leaf < 1:
            raise ValueError(f"min_leaf must be >= 1; got {min_leaf}")
        if threshold_bounds is not None:
            threshold_bounds = np.asarray(threshold_bounds, dtype=np.float64)
            if threshold_bounds.shape != (len(features), 2):
                raise ValueError(
                    f"threshold_bounds must have shape ({len(features)}, 2); " f"got {threshold_bounds.shape}"
                )
        if _rust.rust_available():
            found = _rust.best_cart_split(
                X,
                Y,
                features,
                min_leaf=min_leaf,
                threshold_bounds=threshold_bounds,
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                return None
            feature, threshold, score = found
            return Split(feature=feature, threshold=threshold, score=score)
        return super().best_split(X, Y, features, rng, min_leaf, threshold_bounds, max_cutpoints)
