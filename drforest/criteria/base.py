"""Split-criterion interface and a reusable mean-embedding sweep.

The splitting-geometry axis is pluggable, and the *true* contract is just
``Criterion.best_split(X, Y, features, rng, min_leaf, threshold_bounds, max_cutpoints) -> Split | None``
(spec A.2: evaluate cutoffs of a candidate variable; ``threshold_bounds``
restricts the admissible split threshold per feature, while ``max_cutpoints``
optionally caps the shared split grid for criterion-fair approximate sweeps).

Several — but not all — criteria reduce to one shared computation: scoring a
split (L | R) by the between-group scatter of a per-point embedding
``Ψ ∈ ℂ^{n×m}`` (real for CART, complex for RFF),

    score(L, R) = scale · (n_L n_R / n_P²) · ‖ mean_L Ψ − mean_R Ψ ‖²,

i.e. the scaled MMD of Eq. 12 (and, for the identity kernel, exactly
variance-reduction CART up to ``scale``). Those criteria subclass
``MeanEmbeddingCriterion`` and only supply ``embed`` and ``scale``.

Criteria with a different geometry — e.g. ``sliced_wasserstein``, which needs
per-projection sorting and 1-D quantile gaps, not prefix sums of means —
implement ``best_split`` directly and do not pay for an embedding sweep that
does not fit them.
"""

import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple, SupportsIndex, cast

import numpy as np
from numpy.random import Generator


class Split(NamedTuple):
    feature: int
    threshold: float
    score: float


def _as_feature_index(f: object) -> int:
    """Coerce a feature id to ``int``, rejecting bool and non-integral types.

    ``operator.index`` admits ``bool`` (it is an ``int`` subclass), so reject it
    explicitly: ``True`` masquerading as column 1 is a wiring bug, not an index.
    """
    if isinstance(f, bool | np.bool_):
        raise TypeError(f"feature index must be an integer, not bool: {f!r}")
    try:
        # cast: f is untrusted input; operator.index raises TypeError at runtime
        # for anything that is not actually integral, which we re-raise below.
        return operator.index(cast(SupportsIndex, f))
    except TypeError as exc:
        raise TypeError(f"feature index must be an integer; got {f!r}") from exc


def validate_split_inputs(X: np.ndarray, Y: np.ndarray, features: Sequence[int]) -> None:
    """Fail loudly on mis-wired split inputs (spec guiding principle: no silent defaults)."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
    if Y.shape[1] == 0:
        raise ValueError("Y has zero response dimensions (d == 0)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y disagree on n: {X.shape[0]} rows vs {Y.shape[0]}")
    if len(features) == 0:
        raise ValueError("empty candidate-feature set: an empty mtry is a wiring bug")
    n_p = X.shape[1]
    for f in features:
        idx = _as_feature_index(f)
        if not 0 <= idx < n_p:
            raise ValueError(f"feature index {idx} out of range for p={n_p}")


class Criterion(ABC):
    """A pluggable split criterion (the splitting-geometry axis).

    Implementations must be **stateless / reentrant**: a single configured
    criterion instance is shared across every tree and node of a forest, so
    ``best_split`` must not mutate instance state, and all per-node randomness
    (RFF frequencies, projections) must be drawn from the passed ``rng`` — never
    cached on ``self``.
    """

    @abstractmethod
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
        """Best split over ``features`` (mtry candidates), or None if none valid.

        ``threshold_bounds`` constrains the admissible split threshold per
        candidate: it is aligned with ``features`` (shape ``(len(features), 2)``,
        columns ``(lo, hi)``), and a returned split on ``features[j]`` must
        satisfy ``lo_j <= threshold < hi_j`` (strict upper, matching ``x <= t``
        routing). ``lo >= hi`` means that feature admits no split. ``None`` means
        unconstrained. Honouring the band is part of the contract — every
        axis-aligned criterion searches inside the same threshold domain.

        ``max_cutpoints`` optionally caps the number of candidate thresholds per
        feature. Capped grids are deterministic rank-quantile probes snapped to
        valid split boundaries, then de-duplicated. This search approximation is
        criterion-agnostic: CART, MMD, and sliced-Wasserstein all score the same
        split grid when a cap is active.
        """


def split_candidate_positions(
    xs: np.ndarray,
    *,
    min_leaf: int,
    lo: float,
    hi: float,
    max_cutpoints: int | None,
) -> np.ndarray:
    """Valid split boundary positions, optionally capped by rank probes.

    ``xs`` must already be sorted. A returned position ``i`` means the split
    boundary after ``xs[i]`` and before ``xs[i + 1]``. Capping places probes
    evenly over the bounded-valid rank span, snaps each probe to the nearest
    bounded-valid boundary, and de-duplicates. The result can contain fewer than
    ``max_cutpoints`` positions when multiple probes snap to the same valid
    boundary. It never consumes RNG.
    """
    n = xs.shape[0]
    if min_leaf < 1:
        raise ValueError(f"min_leaf must be >= 1; got {min_leaf}")
    if max_cutpoints is not None:
        if isinstance(max_cutpoints, bool | np.bool_) or not isinstance(max_cutpoints, Integral):
            raise TypeError(
                f"max_cutpoints must be an integer or None, not {type(max_cutpoints).__name__}: {max_cutpoints!r}"
            )
        if max_cutpoints < 1:
            raise ValueError(f"max_cutpoints must be >= 1 or None; got {max_cutpoints}")
        max_cutpoints = int(max_cutpoints)
    if n < 2 * min_leaf:
        return np.empty(0, dtype=np.int64)

    positions = np.arange(n - 1, dtype=np.int64)
    lower = np.maximum(xs[:-1], lo)
    upper = np.minimum(xs[1:], hi)
    valid = (positions >= min_leaf - 1) & (positions <= n - min_leaf - 1) & (xs[:-1] != xs[1:]) & (lower < upper)
    valid_positions = positions[valid]
    if max_cutpoints is None or valid_positions.shape[0] <= max_cutpoints:
        return valid_positions

    low = int(valid_positions[0])
    high = int(valid_positions[-1])
    probe_positions = np.linspace(low, high, num=2 * max_cutpoints + 1)[1::2]
    insertion = np.searchsorted(valid_positions, probe_positions, side="left")
    snapped = np.empty(max_cutpoints, dtype=np.int64)
    for j, pos in enumerate(probe_positions):
        right_idx = insertion[j]
        if right_idx == 0:
            snapped[j] = valid_positions[0]
            continue
        if right_idx == valid_positions.shape[0]:
            snapped[j] = valid_positions[-1]
            continue
        left = valid_positions[right_idx - 1]
        right = valid_positions[right_idx]
        snapped[j] = left if pos - left <= right - pos else right
    return np.unique(snapped)


def _best_split_on_feature(
    x: np.ndarray,
    Psi: np.ndarray,
    scale: float,
    min_leaf: int,
    lo: float,
    hi: float,
    max_cutpoints: int | None = None,
) -> tuple[float, float] | None:
    """Best (threshold, score) for one feature, or None if no valid split.

    A split after sorted positions ``i | i+1`` is the *interval* of thresholds
    ``xs[i] <= t < xs[i+1]`` (any ``t`` there gives the same partition under
    ``x <= t`` routing). It is admissible iff that interval intersects the
    leaf-feasibility band ``[lo, hi)`` — i.e. ``max(xs[i], lo) < min(xs[i+1], hi)``
    — *not* merely when the midpoint lies in the band. The returned threshold is
    the midpoint of the feasible sub-interval ``[lower, upper)``, which reduces
    to the usual gap midpoint when the band does not bind. Scoring uses the split
    sample alone; the band only restricts *which* cutpoints are eligible.

    Phase-1 implementation note: ``cumsum`` materialises every left-prefix sum
    ``S_L`` at once. That is a deliberate NumPy vectorisation, *not* the
    conceptual contract — the Rust port carries ``S_L += φ̃(y)`` incrementally
    in ``O(B)`` per moved point, with ``O(B)`` working memory rather than the
    ``O(n · B)`` prefix matrix built here.
    """
    n = x.shape[0]
    if n < 2 * min_leaf:
        return None

    order = np.argsort(x, kind="stable")
    xs = x[order]
    psi = Psi[order]
    positions = split_candidate_positions(xs, min_leaf=min_leaf, lo=lo, hi=hi, max_cutpoints=max_cutpoints)
    if positions.shape[0] == 0:
        return None

    # The cap restricts which scores enter the argmax. In this NumPy path the
    # full prefix score array is still cheap to materialise; sliced-W is where
    # reducing the candidate set changes runtime materially.
    prefix = np.cumsum(psi, axis=0)  # prefix[i] = Σ_{k≤i} ψ_k  -> S_L at split i
    total = prefix[-1]

    n_l = np.arange(1, n)  # left size for the split after position i (i = 0..n-2)
    n_r = n - n_l
    s_l = prefix[:-1]
    s_r = total - s_l

    diff = s_l / n_l[:, None] - s_r / n_r[:, None]
    sq_norm = np.abs(diff)
    sq_norm = np.einsum("ij,ij->i", sq_norm, sq_norm)  # Σ_b |Δ_b|²
    scores = scale * (n_l * n_r) / (n * n) * sq_norm

    selected_scores = scores[positions]
    i = int(positions[int(np.argmax(selected_scores))])
    lower = max(float(xs[i]), float(lo))
    upper = min(float(xs[i + 1]), float(hi))
    threshold = 0.5 * (lower + upper)  # interior of [lower, upper): keeps both bands
    return float(threshold), float(scores[i])


class MeanEmbeddingCriterion(Criterion):
    """Criteria scored by between-group scatter of a per-point embedding ``Ψ``."""

    @abstractmethod
    def embed(self, Y: np.ndarray, rng: Generator) -> np.ndarray:
        """Per-point embedding ``Ψ`` (n, m) of the node responses ``Y``.

        May resample randomness (RFF frequencies, projections) from ``rng``;
        called once per node before sweeping all candidate features.
        """

    @property
    @abstractmethod
    def scale(self) -> float:
        """Constant score multiplier (e.g. 1/B for RFF). Argmax-invariant."""

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
        Psi = self.embed(Y, rng)
        best: Split | None = None
        for j, f in enumerate(features):
            idx = _as_feature_index(f)
            lo, hi = (-np.inf, np.inf) if threshold_bounds is None else threshold_bounds[j]
            found = _best_split_on_feature(X[:, idx], Psi, self.scale, min_leaf, lo, hi, max_cutpoints)
            if found is None:
                continue
            threshold, score = found
            if best is None or score > best.score:
                best = Split(feature=idx, threshold=threshold, score=score)
        return best
