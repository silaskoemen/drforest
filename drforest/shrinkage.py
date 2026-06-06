"""Post-hoc shrinkage transforms on the DRF weight simplex."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from drforest.features.rff import GaussianRFF
from drforest.mixture import MixtureWeights
from drforest.tree import DecisionTree
from drforest.weights import (
    _as_csr_weights,
    _leaf_atoms,
    _normalize_rows,
    _ragged_leaf_contributions,
    embedding_norm_sq,
    mmd_to_target,
    n_eff,
)


@dataclass(frozen=True)
class ShrinkageResult:
    """Shrunk weights plus the row-wise intensity used to form them."""

    weights: MixtureWeights
    alpha: np.ndarray
    target_weights: csr_matrix


def marginal_target(n_train: int) -> csr_matrix:
    """Uniform marginal target on the training atoms, shape ``(1, n_train)``."""
    if isinstance(n_train, bool) or not isinstance(n_train, Integral):
        raise TypeError(f"n_train must be an integer, not {type(n_train).__name__}: {n_train!r}")
    n_train = int(n_train)
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1; got {n_train}")
    return csr_matrix(np.full((1, n_train), 1.0 / n_train, dtype=np.float64))


def parent_target(trees: Sequence[DecisionTree], X_test: np.ndarray, n_train: int) -> csr_matrix:
    """Per-test parent-node target averaged over the same trees as the forest.

    For a test point routed to a leaf, each tree contributes the empirical
    distribution of that leaf's parent node, populated by the honest leaf sample.
    A root leaf has no parent, so it contributes its own root distribution; this
    makes stump targets identical to stump weights.
    """
    if len(trees) == 0:
        raise ValueError("parent target needs at least one tree")
    if isinstance(n_train, bool) or not isinstance(n_train, Integral):
        raise TypeError(f"n_train must be an integer, not {type(n_train).__name__}: {n_train!r}")
    n_train = int(n_train)
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1; got {n_train}")

    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2-D (n, p); got shape {X_test.shape}")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    n_test = X_test.shape[0]
    for tree in trees:
        atoms_by_parent_leaf, parent_leaf_ptr = _parent_atoms_by_leaf(tree)
        leaf_of_test = tree.apply(X_test)
        r, c, d = _ragged_leaf_contributions(
            leaf_of_test,
            atoms_by_parent_leaf,
            parent_leaf_ptr,
            len(trees),
            empty_message="empty parent target encountered: tree violates the no-empty-leaf guarantee",
        )
        rows.append(r)
        cols.append(c)
        data.append(d)

    target = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_test, n_train))
    target.sum_duplicates()
    return target


def _parent_atoms_by_leaf(tree: DecisionTree) -> tuple[np.ndarray, np.ndarray]:
    atoms_by_leaf, leaf_ptr, _ = _leaf_atoms(tree)
    parent = np.full(tree.n_nodes, -1, dtype=np.int32)
    internal = np.nonzero(tree.feature >= 0)[0]
    parent[tree.left[internal]] = internal
    parent[tree.right[internal]] = internal

    leaf_node = np.empty(tree.n_leaves, dtype=np.int32)
    leaf_nodes = np.nonzero(tree.leaf_id >= 0)[0]
    leaf_node[tree.leaf_id[leaf_nodes]] = leaf_nodes
    leaf_start, leaf_end = _subtree_leaf_ranges(tree, 0)

    segments = []
    parent_leaf_ptr = np.zeros(tree.n_leaves + 1, dtype=np.int64)
    for leaf in range(tree.n_leaves):
        parent_node = int(parent[leaf_node[leaf]])
        if parent_node < 0:
            start, end = leaf, leaf + 1
        else:
            start, end = int(leaf_start[parent_node]), int(leaf_end[parent_node])
        segment = atoms_by_leaf[leaf_ptr[start] : leaf_ptr[end]]
        segments.append(segment)
        parent_leaf_ptr[leaf + 1] = parent_leaf_ptr[leaf] + segment.shape[0]
    return np.concatenate(segments), parent_leaf_ptr


def _subtree_leaf_ranges(tree: DecisionTree, node: int) -> tuple[np.ndarray, np.ndarray]:
    """Leaf-id half-open ranges for each subtree.

    This relies on the builder assigning leaf ids in depth-first order, which
    makes each subtree's leaves contiguous. The explicit adjacency check catches
    any future builder/Rust backend that changes leaf numbering without updating
    parent-target assembly.
    """
    start = np.zeros(tree.n_nodes, dtype=np.int32)
    end = np.zeros(tree.n_nodes, dtype=np.int32)

    def visit(cur: int) -> tuple[int, int]:
        leaf = tree.leaf_id[cur]
        if leaf >= 0:
            start[cur] = leaf
            end[cur] = leaf + 1
        else:
            left_start, left_end = visit(int(tree.left[cur]))
            right_start, right_end = visit(int(tree.right[cur]))
            if left_end != right_start:
                raise ValueError(
                    "leaf ids must be depth-first contiguous for parent-target shrinkage; "
                    f"node {cur} has left range ending at {left_end} and right range starting at {right_start}"
                )
            start[cur] = left_start
            end[cur] = right_end
        return int(start[cur]), int(end[cur])

    visit(node)
    return start, end


def shrink_to_target(
    W: object,
    Y: np.ndarray,
    *,
    rff: GaussianRFF,
    target_weights: object,
    parameterization: str = "kmse",
) -> ShrinkageResult:
    """Shrink rows of ``W`` toward precomputed target weights."""
    if parameterization not in ("kmse", "stein"):
        raise ValueError(f"parameterization must be 'kmse' or 'stein'; got {parameterization!r}")

    W_csr = _normalize_rows(_as_csr_weights(W))
    target_csr = _normalize_rows(_as_csr_weights(target_weights))
    variance = np.maximum(1.0 - embedding_norm_sq(W_csr, Y, rff), 0.0)
    distance = mmd_to_target(W_csr, target_csr, Y, rff)
    scaled_distance = n_eff(W_csr) * distance
    # kmse: V/(V+MMD²) → variance/(variance + n_eff·MMD²);
    # stein: V/MMD²     → variance/(n_eff·MMD²)  (bias-corrected denominator).
    denominator = scaled_distance if parameterization == "stein" else variance + scaled_distance
    # denominator == 0 is a limit, not a guard: with positive variance it means
    # MMD² -> 0 (conditional indistinguishable from target) so α -> 1; with zero
    # variance the row is degenerate and α -> 0. Both reduce to 1{variance > 0}.
    limit = (variance > 0.0).astype(np.float64)
    alpha = np.divide(variance, denominator, out=limit, where=denominator > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return ShrinkageResult(
        weights=MixtureWeights(base=W_csr, alpha=alpha, target=target_csr),
        alpha=alpha,
        target_weights=target_csr,
    )


def shrink(
    W: object,
    Y: np.ndarray,
    *,
    rff: GaussianRFF,
    target: str = "marginal",
    parameterization: str = "kmse",
    trees: Sequence[DecisionTree] | None = None,
    X_test: np.ndarray | None = None,
) -> ShrinkageResult:
    """Shrink rows of ``W`` toward a target distribution on the same atoms.

    Both closed forms estimate the bias-variance optimum ``α* = V / (V + D²)``
    with the RFF-pinned variance ``V = (1 - ‖μ̂‖²) / n_eff`` (k(y,y)=1). They
    differ only in how the squared bias ``D² = ‖μ*-μ₀‖²`` is plugged in:

    - ``"kmse"`` uses the raw empirical ``MMD² = ‖μ̂-μ₀‖²`` (kernel-mean
      shrinkage form, Muandet et al. 2016)::

          α = (1 - ‖μ̂‖²) / ((1 - ‖μ̂‖²) + n_eff · MMD²).

    - ``"stein"`` uses the bias-corrected ``D̂² = MMD² - V`` (E[MMD²] = D² + V),
      collapsing to the positive-part James–Stein form::

          α = V / MMD² = (1 - ‖μ̂‖²) / (n_eff · MMD²).

    The two agree when ``MMD² ≫ V`` (strong conditional signal) and diverge only
    when ``MMD² ≈ V`` (weak signal), where ``"stein"`` shrinks more aggressively.

    ``target="marginal"`` uses the global empirical marginal. ``target="parent"``
    requires ``trees`` and ``X_test`` and shrinks each routed leaf toward its
    parent node's honest empirical distribution before averaging over trees. The
    kernel geometry is fixed by the caller-supplied ``rff`` map; no bandwidth or
    feature-count defaults are chosen inside this transform.
    """
    if target not in ("marginal", "parent"):
        raise ValueError(f"unsupported shrinkage target {target!r}; expected 'marginal' or 'parent'")

    W_csr = _normalize_rows(_as_csr_weights(W))
    if target == "marginal":
        target_weights = marginal_target(W_csr.shape[1])
    else:
        if trees is None or X_test is None:
            raise ValueError("target='parent' requires both trees and X_test")
        target_weights = parent_target(trees, X_test, W_csr.shape[1])
    return shrink_to_target(W_csr, Y, rff=rff, target_weights=target_weights, parameterization=parameterization)
