"""CART criterion — the identity-kernel correctness anchor.

With the identity kernel ``k(y, y') = ⟨y, y'⟩`` the embedding is just ``Ψ = Y``,
so the shared between-group scatter score becomes

    (n_L n_R / n²) · ‖ ȳ_L − ȳ_R ‖²,

which has the same argmax as multivariate variance reduction (between-group
sum of squares, Eq. 13). This makes CART the reference the RFF criterion is
tested against, and the mean-only baseline.
"""

import numpy as np
from numpy.random import Generator

from drforest.criteria.base import MeanEmbeddingCriterion


class CartCriterion(MeanEmbeddingCriterion):
    def embed(self, Y: np.ndarray, rng: Generator) -> np.ndarray:
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        return np.ascontiguousarray(Y, dtype=np.float64)

    @property
    def scale(self) -> float:
        return 1.0
