"""Sliced-Wasserstein split criterion.

At each node, responses are projected onto random unit directions and candidate
children are scored by the average exact 1-D squared Wasserstein distance
between their projected empirical distributions. The projections are resampled
from the node RNG, mirroring the decorrelation role of RFF frequencies in the
MMD criterion.
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


class SlicedWassersteinCriterion(Criterion):
    """Random-projection sliced-Wasserstein criterion."""

    def __init__(self, n_projections: int, dim: int) -> None:
        if n_projections <= 0:
            raise ValueError(f"n_projections must be positive; got {n_projections}")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self.n_projections = int(n_projections)
        self.dim = int(dim)

    @classmethod
    def from_data(cls, Y: np.ndarray, n_projections: int) -> "SlicedWassersteinCriterion":
        """Configure the criterion from response dimensionality."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        return cls(n_projections=n_projections, dim=Y.shape[1])

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

        if self.dim == 1:
            # In one response dimension every unit projection is either +Y or
            # -Y, and W2 is invariant to a shared sign flip. Averaging many
            # projections repeats the same score, so use the exact equivalent
            # single projected column.
            projected = np.ascontiguousarray(Y, dtype=np.float64)
        else:
            theta = _sample_unit_directions(self.dim, self.n_projections, rng)
            projected = np.ascontiguousarray(Y @ theta.T, dtype=np.float64)
        best: Split | None = None
        for j, f in enumerate(features):
            idx = _as_feature_index(f)
            lo, hi = (-np.inf, np.inf) if threshold_bounds is None else threshold_bounds[j]
            found = _best_split_on_feature_sliced(X[:, idx], projected, min_leaf, lo, hi, max_cutpoints)
            if found is None:
                continue
            threshold, score = found
            if best is None or score > best.score:
                best = Split(feature=idx, threshold=threshold, score=score)
        return best


def _sample_unit_directions(dim: int, n_projections: int, rng: Generator) -> np.ndarray:
    directions = rng.normal(size=(n_projections, dim))
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0.0):
        raise RuntimeError("failed to sample nonzero projection directions")
    return directions / norms[:, None]


def _wasserstein_1d_sq(left: np.ndarray, right: np.ndarray) -> float:
    """Exact W2² between two one-dimensional empirical distributions."""
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("left and right samples must be one-dimensional")
    n_left = left.shape[0]
    n_right = right.shape[0]
    if n_left == 0 or n_right == 0:
        raise ValueError("left and right samples must both be non-empty")

    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    if n_left == n_right:
        return float(np.mean((left_sorted - right_sorted) ** 2))

    left_jumps = np.arange(1, n_left, dtype=np.float64) / n_left
    right_jumps = np.arange(1, n_right, dtype=np.float64) / n_right
    breaks = np.concatenate(([0.0], left_jumps, right_jumps, [1.0]))
    breaks.sort()
    breaks = np.unique(breaks)

    lower = breaks[:-1]
    upper = breaks[1:]
    widths = upper - lower
    mid = 0.5 * (lower + upper)
    left_idx = np.minimum((mid * n_left).astype(np.int64), n_left - 1)
    right_idx = np.minimum((mid * n_right).astype(np.int64), n_right - 1)
    diff = left_sorted[left_idx] - right_sorted[right_idx]
    return float(np.sum(widths * diff**2))


def _sliced_wasserstein_sq(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] == 1:
        return _wasserstein_1d_sq(left[:, 0], right[:, 0])
    values = [_wasserstein_1d_sq(left[:, b], right[:, b]) for b in range(left.shape[1])]
    return float(np.mean(values))


def _best_split_on_feature_sliced(
    x: np.ndarray,
    projected: np.ndarray,
    min_leaf: int,
    lo: float,
    hi: float,
    max_cutpoints: int | None = None,
) -> tuple[float, float] | None:
    n = x.shape[0]
    if n < 2 * min_leaf:
        return None

    order = np.argsort(x, kind="stable")
    xs = x[order]
    z = projected[order]
    positions = split_candidate_positions(xs, min_leaf=min_leaf, lo=lo, hi=hi, max_cutpoints=max_cutpoints)
    if positions.shape[0] == 0:
        return None

    best: tuple[float, float] | None = None
    for i in positions:
        n_left = i + 1
        n_right = n - n_left
        lower = max(float(xs[i]), float(lo))
        upper = min(float(xs[i + 1]), float(hi))

        sw_sq = _sliced_wasserstein_sq(z[:n_left], z[n_left:])
        score = (n_left * n_right) / (n * n) * sw_sq
        threshold = 0.5 * (lower + upper)
        if best is None or score > best[1]:
            best = (threshold, float(score))
    return best
