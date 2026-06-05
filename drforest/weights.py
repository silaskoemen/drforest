"""Assembly of the central object: the row-stochastic weight matrix ``W``.

The induced DRF weighting (spec A.1) is

    w_x(x_i) = (1/N) Σ_k  1(x_i ∈ L_k(x)) / |L_k(x)|,

a distribution on the training atoms ``{x_i}``. Everything downstream
(quantiles, CDF, mean, shrinkage, ...) is a plug-in on ``W ∈ ℝ^{n_test×n_train}``.

Because the tree builder guarantees every leaf holds at least
``min_samples_leaf`` leaf-sample atoms, ``|L_k(x)| ≥ 1`` always: each tree
contributes total weight exactly 1 to every test point, so rows of ``W`` sum to
exactly 1 with no empty-leaf special-casing.

Layout note: trees expose only compact integer arrays (per-test leaf ids, and
per-leaf-sample leaf assignment), exactly what the Rust builder will return, so
this assembly stays pure-Python/SciPy across the Phase-3 swap.
"""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from drforest.tree import DecisionTree


def _leaf_atoms(tree: DecisionTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group a tree's leaf-sample atoms by leaf.

    Returns ``(atoms_by_leaf, leaf_ptr, leaf_sizes)`` where ``atoms_by_leaf`` are
    the global training rows ordered by leaf and ``leaf_ptr`` are CSR-style
    offsets, so leaf ``l``'s atoms are ``atoms_by_leaf[leaf_ptr[l]:leaf_ptr[l+1]]``.
    """
    order = np.argsort(tree.leaf_sample_leaf, kind="stable")
    atoms_by_leaf = tree.leaf_sample_rows[order].astype(np.int64, copy=False)
    leaf_sizes = np.bincount(tree.leaf_sample_leaf, minlength=tree.n_leaves)
    leaf_ptr = np.zeros(tree.n_leaves + 1, dtype=np.int64)
    np.cumsum(leaf_sizes, out=leaf_ptr[1:])
    return atoms_by_leaf, leaf_ptr, leaf_sizes


def _tree_contributions(
    tree: DecisionTree, X_test: np.ndarray, n_trees: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO (rows, cols, data) of one tree's weight contribution to ``W``."""
    leaf_of_test = tree.apply(X_test)  # (n_test,), validates n_features_in
    atoms_by_leaf, leaf_ptr, leaf_sizes = _leaf_atoms(tree)

    counts = leaf_sizes[leaf_of_test]  # atoms each test point inherits
    if not (counts > 0).all():
        raise ValueError("empty leaf encountered: tree violates the no-empty-leaf guarantee")

    n_test = X_test.shape[0]
    total = int(counts.sum())
    # Ragged gather: for test point i in leaf l, emit its atoms with weight
    # 1 / (n_trees * |l|). within-block offset = position - block start.
    block_start = np.zeros(n_test, dtype=np.int64)
    np.cumsum(counts[:-1], out=block_start[1:])
    within = np.arange(total) - np.repeat(block_start, counts)
    atom_index = np.repeat(leaf_ptr[leaf_of_test], counts) + within

    rows = np.repeat(np.arange(n_test), counts)
    cols = atoms_by_leaf[atom_index]
    data = np.repeat(1.0 / (n_trees * counts), counts)
    return rows, cols, data


def assemble_weights(trees: Sequence[DecisionTree], X_test: np.ndarray, n_train: int) -> csr_matrix:
    """Row-stochastic weight matrix ``W`` (n_test × n_train) for the forest."""
    if len(trees) == 0:
        raise ValueError("no trees to assemble weights from")
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2-D (n, p); got shape {X_test.shape}")

    n_trees = len(trees)
    rows, cols, data = [], [], []
    for tree in trees:
        r, c, d = _tree_contributions(tree, X_test, n_trees)
        rows.append(r)
        cols.append(c)
        data.append(d)

    W = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(X_test.shape[0], n_train),
    )
    W.sum_duplicates()  # the same atom across trees accumulates into one entry
    return W
