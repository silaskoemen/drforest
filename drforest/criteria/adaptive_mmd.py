"""Adaptive-frequency MMD split criterion.

The ordinary ``mmd_rff`` criterion averages a fixed Gaussian RFF draw uniformly.
This criterion samples an overcomplete node-local frequency pool and scores each
candidate split by the strongest subset of per-frequency MMD contributions.
That makes the kernel geometry adaptive at split-selection time while preserving
the existing criterion contract: all node-local randomness comes from the
passed RNG and no learned frequencies are cached on the criterion instance.
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

from drforest.criteria.base import (
    Criterion,
    Split,
    _as_feature_index,
    split_candidate_positions,
    validate_split_inputs,
)
from drforest.features.rff import BandwidthRule, sample_rff


class AdaptiveMmdCriterion(Criterion):
    """MMD criterion with per-split top-k frequency selection from an RFF pool."""

    def __init__(self, pool_features: int, selected_features: int, sigma: float, dim: int) -> None:
        if pool_features <= 0:
            raise ValueError(f"pool_features must be positive; got {pool_features}")
        if selected_features <= 0:
            raise ValueError(f"selected_features must be positive; got {selected_features}")
        if selected_features > pool_features:
            raise ValueError(f"selected_features must be <= pool_features; got {selected_features} > {pool_features}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self.pool_features = int(pool_features)
        self.selected_features = int(selected_features)
        self.sigma = float(sigma)
        self.dim = int(dim)

    @classmethod
    def from_data(
        cls,
        Y: np.ndarray,
        *,
        pool_features: int,
        selected_features: int,
        bandwidth_rule: BandwidthRule,
    ) -> "AdaptiveMmdCriterion":
        """Configure the criterion, fixing sigma from the training responses."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        return cls(
            pool_features=pool_features,
            selected_features=selected_features,
            sigma=bandwidth_rule(Y),
            dim=Y.shape[1],
        )

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
                raise ValueError(f"threshold_bounds must have shape ({len(features)}, 2); got {threshold_bounds.shape}")

        rff = sample_rff(self.dim, self.pool_features, self.sigma, rng)
        psi = rff.transform(Y)

        best: Split | None = None
        for j, f in enumerate(features):
            idx = _as_feature_index(f)
            lo, hi = (-np.inf, np.inf) if threshold_bounds is None else threshold_bounds[j]
            found = _best_split_on_feature_adaptive(
                X[:, idx],
                psi,
                self.selected_features,
                min_leaf,
                lo,
                hi,
                max_cutpoints,
            )
            if found is None:
                continue
            threshold, score = found
            if best is None or score > best.score:
                best = Split(feature=idx, threshold=threshold, score=score)
        return best


def _best_split_on_feature_adaptive(
    x: np.ndarray,
    psi: np.ndarray,
    selected_features: int,
    min_leaf: int,
    lo: float,
    hi: float,
    max_cutpoints: int | None = None,
) -> tuple[float, float] | None:
    """Best split using the mean of the top-k per-frequency MMD contributions."""
    n = x.shape[0]
    if n < 2 * min_leaf:
        return None
    if selected_features <= 0:
        raise ValueError(f"selected_features must be positive; got {selected_features}")
    if selected_features > psi.shape[1]:
        raise ValueError(f"selected_features must be <= psi columns; got {selected_features} > {psi.shape[1]}")

    order = np.argsort(x, kind="stable")
    xs = x[order]
    psi_sorted = psi[order]
    positions = split_candidate_positions(xs, min_leaf=min_leaf, lo=lo, hi=hi, max_cutpoints=max_cutpoints)
    if positions.shape[0] == 0:
        return None

    prefix = np.cumsum(psi_sorted, axis=0)
    total = prefix[-1]

    n_l = positions + 1
    n_r = n - n_l
    s_l = prefix[positions]
    s_r = total - s_l
    diff = s_l / n_l[:, None] - s_r / n_r[:, None]
    coord_scores = np.abs(diff) ** 2
    if selected_features == coord_scores.shape[1]:
        selected = np.mean(coord_scores, axis=1)
    else:
        top = np.partition(coord_scores, -selected_features, axis=1)[:, -selected_features:]
        selected = np.mean(top, axis=1)
    scores = (n_l * n_r) / (n * n) * selected

    best_idx = int(np.argmax(scores))
    i = int(positions[best_idx])
    lower = max(float(xs[i]), float(lo))
    upper = min(float(xs[i + 1]), float(hi))
    threshold = 0.5 * (lower + upper)
    return float(threshold), float(scores[best_idx])
