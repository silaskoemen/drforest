"""Single honest decision tree (Phase-1 pure Python).

One tree is grown target-free: each split maximises the chosen ``Criterion``'s
two-sample statistic on the responses in the candidate children. The tree is
the first consumer of ``Criterion.best_split`` and the unit the forest will
later weight and (in Phase 3) hand to Rust.

Design choices (see also the project config note):

- **Subsampling.** Each tree is grown on a subsample of ``subsample_size`` rows
  drawn *without replacement* from the full training set (spec A.4: not
  bootstrap). All stored row indices are **global** (into the training ``X``),
  so the forest assembles the weight matrix without any row-mapping retrofit.

- **Honesty.** The subsample is partitioned into a disjoint *split-sample*
  ``S`` (chooses structure) and *leaf-sample* ``L`` (populates the leaves that
  drive the weights). ``honesty_fraction`` is the share going to ``S``; ``0.0``
  is the opt-in "fast" mode (``S = L =`` subsample), which is *not*
  inference-valid. The criterion searches for the best split **within an X-only
  leaf-feasibility band** (``threshold_bounds``, computed from ``X_L`` so the
  responses ``Y_L`` are never consulted): only thresholds keeping >=
  ``min_samples_leaf`` rows of ``L`` on each side are admissible. Scoring still
  uses ``S`` only. So the criterion returns the best split that also keeps both
  honest leaves populated — empty honest leaves are impossible and we no longer
  stump just because the unconstrained best-on-``S`` split was leaf-infeasible.

- **colsample.** A fraction of *features* is resampled per node as split
  candidates; for each candidate the criterion sweep evaluates every cut point.

- **Independent per-node RNG axes.** Each node spawns two sibling streams from
  its node generator: one for structure (colsampling) and one for the criterion
  (RFF / projection resampling). They are decoupled, so the criterion's
  randomness does not depend on ``colsample``, ``p``, or candidate draw order.

- **alpha-regularity + min_samples_leaf.** ``min_child = max(min_samples_leaf,
  ceil(alpha * n_node_S))``. ``min_samples_leaf`` is an absolute floor
  (estimation variance); ``alpha`` is a relative floor that forces geometric
  node shrinkage with depth — the assumption behind the inference-valid claim.

- **Struct-of-arrays.** Nodes are contiguous ``int32``/``float64`` arrays (no
  per-node Python objects in the routing hot path); routing is a module-level
  function, so the Phase-3 Rust swap touches only the builder.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from drforest.criteria.base import Criterion, Split

NodeRng = Callable[[int], Generator]


@dataclass(frozen=True)
class TreeParams:
    """Growth hyperparameters for a single tree."""

    min_samples_leaf: int = 5
    alpha: float = 0.05
    honesty_fraction: float = 0.5
    colsample: float = 0.7

    def __post_init__(self) -> None:
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1; got {self.min_samples_leaf}")
        if not 0.0 <= self.honesty_fraction < 1.0:
            raise ValueError(f"honesty_fraction must be in [0, 1); got {self.honesty_fraction}")
        if not 0.0 < self.colsample <= 1.0:
            raise ValueError(f"colsample must be in (0, 1]; got {self.colsample}")
        if not 0.0 <= self.alpha <= 0.5:
            raise ValueError(f"alpha must be in [0, 0.5]; got {self.alpha}")

    @property
    def honest(self) -> bool:
        return self.honesty_fraction > 0.0

    def n_candidates(self, n_features: int) -> int:
        """Number of candidate features per node (>= 1)."""
        return max(1, round(self.colsample * n_features))


def _route(
    feature: np.ndarray,
    threshold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    leaf_id: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Route each row of ``X`` to its leaf id, vectorised across rows.

    Loops over depth (not rows): each iteration advances every still-internal
    point one level. The Rust port keeps this exact shape.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    node = np.zeros(X.shape[0], dtype=np.int64)
    while True:
        internal = feature[node] >= 0
        if not internal.any():
            break
        rows = np.nonzero(internal)[0]
        cur = node[rows]
        go_left = X[rows, feature[cur]] <= threshold[cur]
        node[rows] = np.where(go_left, left[cur], right[cur])
    return leaf_id[node].astype(np.int32, copy=False)


@dataclass(frozen=True)
class DecisionTree:
    """A grown tree as struct-of-arrays plus its honest leaf populations.

    ``*_rows`` are **global** indices into the training ``X`` passed to
    :func:`build_tree`.
    """

    feature: np.ndarray  # int32 (n_nodes,)  split feature, -1 at leaves
    threshold: np.ndarray  # float64 (n_nodes,)  split threshold, NaN at leaves
    left: np.ndarray  # int32 (n_nodes,)  left child, -1 at leaves
    right: np.ndarray  # int32 (n_nodes,)  right child, -1 at leaves
    leaf_id: np.ndarray  # int32 (n_nodes,)  compact leaf index, -1 at internals
    n_leaves: int
    n_features_in: int  # p the tree was fitted on; apply() must match
    split_sample_rows: np.ndarray  # int32  global rows used to grow the structure
    leaf_sample_rows: np.ndarray  # int32  global rows used to populate leaves
    leaf_sample_leaf: np.ndarray  # int32  leaf index of each leaf-sample row

    @property
    def n_nodes(self) -> int:
        return self.feature.shape[0]

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf id (0..n_leaves-1) for each row of ``X``."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}")
        if X.shape[1] != self.n_features_in:
            raise ValueError(f"X has {X.shape[1]} features; tree was fitted on {self.n_features_in}")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values; missing-value handling is not implemented")
        return _route(self.feature, self.threshold, self.left, self.right, self.leaf_id, X)


def fold_sizes(m: int, honesty_fraction: float) -> tuple[int, int]:
    """Deterministic (split-sample, leaf-sample) sizes for a subsample of ``m`` rows.

    ``honesty_fraction == 0`` is fast mode: both folds are the whole subsample.
    """
    if honesty_fraction == 0.0:
        return m, m
    n_split = min(max(round(honesty_fraction * m), 1), m - 1)
    return n_split, m - n_split


def _honesty_split(rows: np.ndarray, honesty_fraction: float, rng: Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split ``rows`` into (split-sample, leaf-sample) global index arrays."""
    if honesty_fraction == 0.0:
        return rows, rows
    n_split, _ = fold_sizes(rows.shape[0], honesty_fraction)
    shuffled = rows[rng.permutation(rows.shape[0])]
    return shuffled[:n_split], shuffled[n_split:]


def _threshold_bounds(X: np.ndarray, l_idx: np.ndarray, candidates: np.ndarray, min_samples_leaf: int) -> np.ndarray:
    """Per-candidate (lo, hi) admissible split-threshold band, aligned with ``candidates``.

    This is an **X-only support constraint**: it uses the leaf-sample covariates
    ``X_L`` (never the responses ``Y_L``), so a split keeps >= ``min_samples_leaf``
    leaf atoms on each side without ever looking at responses — leaving response
    honesty intact. For a feature, ``lo = v_{k-1}`` and ``hi = v_{n_L - k}`` of the
    sorted leaf-sample column (``k = min_samples_leaf``); thresholds ``t`` with
    ``lo <= t < hi`` keep both leaf children >= k. ``lo >= hi`` (ties) marks the
    feature as admitting no leaf-feasible split. Requires ``n_L >= 2k`` (the
    caller's eligibility gate guarantees it).
    """
    n_l = l_idx.shape[0]
    k = min_samples_leaf
    bounds = np.empty((candidates.shape[0], 2), dtype=np.float64)
    for j in range(candidates.shape[0]):
        col = X[l_idx, candidates[j]]
        part = np.partition(col, (k - 1, n_l - k))
        bounds[j, 0] = part[k - 1]
        bounds[j, 1] = part[n_l - k]
    return bounds


def _validate_split_contract(
    split: Split,
    n_features: int,
    min_child: int,
    X: np.ndarray,
    s_idx: np.ndarray,
    l_idx: np.ndarray,
    min_samples_leaf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Revalidate an untrusted criterion ``Split`` (the protocol allows any impl).

    Structural checks (feature in range, finite threshold) run *before* indexing
    with the feature. The split sample must keep ``min_child`` per child, and —
    since honouring ``threshold_bounds`` is now part of the contract — the leaf
    sample must keep ``min_samples_leaf`` per child. Any breach is a criterion
    bug and raises. Returns the left-child membership masks ``(s_left, l_left)``.
    """
    if not 0 <= split.feature < n_features:
        raise ValueError(f"criterion returned feature {split.feature} out of range [0, {n_features})")
    if not math.isfinite(split.threshold):
        raise ValueError(f"criterion returned non-finite threshold {split.threshold}")
    s_left = X[s_idx, split.feature] <= split.threshold
    n_sl = int(s_left.sum())
    n_sr = s_idx.shape[0] - n_sl
    if n_sl < min_child or n_sr < min_child:
        raise ValueError(f"criterion split violates min_child={min_child}: split-sample children are ({n_sl}, {n_sr})")
    l_left = X[l_idx, split.feature] <= split.threshold
    n_ll = int(l_left.sum())
    n_lr = l_idx.shape[0] - n_ll
    if n_ll < min_samples_leaf or n_lr < min_samples_leaf:
        raise ValueError(
            f"criterion ignored threshold_bounds: leaf-sample children ({n_ll}, {n_lr}) "
            f"below min_samples_leaf={min_samples_leaf}"
        )
    return s_left, l_left


def build_tree(
    X: np.ndarray,
    Y: np.ndarray,
    criterion: Criterion,
    params: TreeParams,
    *,
    subsample_size: int,
    bootstrap: bool = False,
    tree_rng: Generator,
    node_rng: NodeRng,
) -> DecisionTree:
    """Grow one honest tree on a row subsample of ``(X, Y)``.

    The subsample is ``subsample_size`` rows drawn without replacement, or with
    replacement when ``bootstrap=True`` (the opt-in, inference-invalid fast
    path). ``tree_rng`` drives the subsample draw and the honesty fold split;
    ``node_rng(node_id)`` yields the per-node stream, from which two independent
    sibling streams are spawned (structure vs criterion).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y disagree on n: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[1] == 0:
        raise ValueError("X has zero features (p == 0)")
    if Y.shape[1] == 0:
        raise ValueError("Y has zero response dimensions (d == 0)")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values; missing-value handling is not implemented")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains non-finite values; missing-value handling is not implemented")
    n, p = X.shape
    if not 1 <= subsample_size <= n:
        raise ValueError(f"subsample_size must be in [1, {n}]; got {subsample_size}")

    # Feasibility (deterministic, RNG-independent): both honest folds must be able
    # to hold min_samples_leaf, else even the root leaf would violate the floor.
    n_s_size, n_l_size = fold_sizes(subsample_size, params.honesty_fraction)
    if min(n_s_size, n_l_size) < params.min_samples_leaf:
        raise ValueError(
            f"infeasible honest folds (S={n_s_size}, L={n_l_size}) from "
            f"subsample_size={subsample_size}, honesty_fraction={params.honesty_fraction}: "
            f"each fold must hold min_samples_leaf={params.min_samples_leaf}; "
            "increase subsample_size or lower honesty_fraction/min_samples_leaf"
        )

    sample_rows = tree_rng.choice(n, size=subsample_size, replace=bootstrap)
    s_rows, l_rows = _honesty_split(sample_rows, params.honesty_fraction, tree_rng)

    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []

    def add_node() -> int:
        idx = len(feature)
        feature.append(-1)
        threshold.append(np.nan)
        left.append(-1)
        right.append(-1)
        return idx

    def grow(s_idx: np.ndarray, l_idx: np.ndarray) -> int:
        idx = add_node()
        n_s = s_idx.shape[0]
        n_l = l_idx.shape[0]
        min_child = max(params.min_samples_leaf, math.ceil(params.alpha * n_s))
        eligible = n_s >= 2 * min_child and n_l >= 2 * params.min_samples_leaf
        if eligible:
            g_struct, g_crit = node_rng(idx).spawn(2)
            candidates = g_struct.choice(p, size=params.n_candidates(p), replace=False)
            # X-only leaf-feasibility band: the criterion searches the best split
            # that also keeps both honest leaves populated. No empty leaves result.
            bounds = _threshold_bounds(X, l_idx, candidates, params.min_samples_leaf)
            split = criterion.best_split(X[s_idx], Y[s_idx], candidates, g_crit, min_child, bounds)
            if split is not None:
                s_left, l_left = _validate_split_contract(split, p, min_child, X, s_idx, l_idx, params.min_samples_leaf)
                feature[idx] = split.feature
                threshold[idx] = split.threshold
                left[idx] = grow(s_idx[s_left], l_idx[l_left])
                right[idx] = grow(s_idx[~s_left], l_idx[~l_left])
        return idx

    grow(s_rows, l_rows)

    feature_arr = np.asarray(feature, dtype=np.int32)
    threshold_arr = np.asarray(threshold, dtype=np.float64)
    left_arr = np.asarray(left, dtype=np.int32)
    right_arr = np.asarray(right, dtype=np.int32)

    is_leaf = feature_arr < 0
    n_leaves = int(is_leaf.sum())
    leaf_id = np.full(feature_arr.shape, -1, dtype=np.int32)
    leaf_id[is_leaf] = np.arange(n_leaves, dtype=np.int32)

    leaf_sample_leaf = _route(feature_arr, threshold_arr, left_arr, right_arr, leaf_id, X[l_rows])

    return DecisionTree(
        feature=feature_arr,
        threshold=threshold_arr,
        left=left_arr,
        right=right_arr,
        leaf_id=leaf_id,
        n_leaves=n_leaves,
        n_features_in=p,
        split_sample_rows=s_rows.astype(np.int32),
        leaf_sample_rows=l_rows.astype(np.int32),
        leaf_sample_leaf=leaf_sample_leaf,
    )
