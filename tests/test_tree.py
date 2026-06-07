import math

import numpy as np
import pytest

from drforest.criteria.base import Criterion, Split
from drforest.criteria.cart import CartCriterion
from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.features.rff import fixed_bandwidth
from drforest.rng import RngStreams
from drforest.tree import (
    DecisionTree,
    TreeParams,
    _threshold_bounds,
    _validate_split_contract,
    build_tree,
)


class _FixedSplitCriterion(Criterion):
    """A misbehaving plugin that always returns the same (untrusted) split."""

    def __init__(self, split: Split):
        self._split = split

    def best_split(self, X, Y, features, rng, min_leaf, threshold_bounds, max_cutpoints=None):
        return self._split


def _node_rng(seed: int):
    streams = RngStreams(seed)
    return lambda node_id: streams.node(0, node_id)


def _build(X, Y, criterion, params, *, seed=0, subsample_size=None):
    return build_tree(
        X,
        Y,
        criterion,
        params,
        subsample_size=X.shape[0] if subsample_size is None else subsample_size,
        tree_rng=np.random.default_rng(seed),
        node_rng=_node_rng(seed),
    )


def _planted_data(seed=7, n=200, p=3, d=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, d)) + (X[:, 0] > 0)[:, None] * np.array([3.0, -2.0])
    return X, Y


# ---- params validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples_leaf": 0},
        {"alpha": 0.6},
        {"alpha": -0.1},
        {"honesty_fraction": 1.0},
        {"honesty_fraction": -0.1},
        {"colsample": 0.0},
        {"colsample": 1.5},
        {"max_cutpoints": 0},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        TreeParams(**kwargs)


def test_n_candidates_rounds_and_floors():
    assert TreeParams(colsample=0.7).n_candidates(10) == 7
    assert TreeParams(colsample=0.7).n_candidates(1) == 1  # max(1, round(0.7))
    assert TreeParams(colsample=1.0).n_candidates(4) == 4


def test_invalid_subsample_size_rejected():
    X, Y = _planted_data(n=20)
    with pytest.raises(ValueError, match="subsample_size"):
        _build(X, Y, CartCriterion(), TreeParams(), subsample_size=21)
    with pytest.raises(ValueError, match="subsample_size"):
        _build(X, Y, CartCriterion(), TreeParams(), subsample_size=0)


def test_non_finite_inputs_rejected():
    X, Y = _planted_data(n=40)
    X_bad = X.copy()
    X_bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _build(X_bad, Y, CartCriterion(), TreeParams())
    Y_bad = Y.copy()
    Y_bad[5, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _build(X, Y_bad, CartCriterion(), TreeParams())


def test_zero_dim_inputs_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="zero features"):
        _build(np.zeros((10, 0)), rng.normal(size=(10, 2)), CartCriterion(), TreeParams())
    with pytest.raises(ValueError, match="zero response"):
        _build(rng.normal(size=(10, 2)), np.zeros((10, 0)), CartCriterion(), TreeParams())


@pytest.mark.parametrize(
    "subsample_size,honesty_fraction",
    [(5, 0.5), (10, 0.99), (4, 0.0)],  # L (or fast fold) < min_samples_leaf=5
)
def test_infeasible_fold_sizes_rejected(subsample_size, honesty_fraction):
    X, Y = _planted_data(n=40)
    params = TreeParams(min_samples_leaf=5, honesty_fraction=honesty_fraction)
    with pytest.raises(ValueError, match="infeasible honest folds"):
        _build(X, Y, CartCriterion(), params, subsample_size=subsample_size)


def test_apply_rejects_wrong_feature_count():
    X, Y = _planted_data(n=60, p=3)
    tree = _build(X, Y, CartCriterion(), TreeParams())
    assert tree.n_features_in == 3
    with pytest.raises(ValueError, match="features"):
        tree.apply(np.zeros((4, 2)))


@pytest.mark.parametrize(
    "bad_split,match",
    [
        (Split(feature=99, threshold=0.0, score=1.0), "out of range"),
        (Split(feature=0, threshold=float("nan"), score=1.0), "non-finite"),
        (Split(feature=0, threshold=-1e9, score=1.0), "min_child"),  # left child empty
    ],
)
def test_misbehaving_criterion_split_is_revalidated(bad_split, match):
    X, Y = _planted_data(n=120, p=3)
    with pytest.raises(ValueError, match=match):
        _build(X, Y, _FixedSplitCriterion(bad_split), TreeParams(honesty_fraction=0.0))


# ---- subsampling -------------------------------------------------------------


def test_subsampling_is_without_replacement_and_global():
    X, Y = _planted_data(n=100)
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.5), subsample_size=50)
    used = np.concatenate([tree.split_sample_rows, tree.leaf_sample_rows])
    assert used.shape[0] == 50
    assert np.unique(used).shape[0] == 50  # no replacement
    assert used.min() >= 0 and used.max() < 100  # global indices


# ---- structure invariants ----------------------------------------------------


def test_alpha_regularity_and_min_samples_leaf_on_split_sample():
    X, Y = _planted_data()
    params = TreeParams(min_samples_leaf=5, alpha=0.2, honesty_fraction=0.0, colsample=1.0)
    tree = _build(X, Y, CartCriterion(), params)
    X_s = X[tree.split_sample_rows]

    def check(node, rows):
        if tree.feature[node] < 0:
            return
        min_child = max(params.min_samples_leaf, math.ceil(params.alpha * len(rows)))
        mask = X_s[rows, tree.feature[node]] <= tree.threshold[node]
        assert mask.sum() >= min_child
        assert (~mask).sum() >= min_child
        check(tree.left[node], rows[mask])
        check(tree.right[node], rows[~mask])

    check(0, np.arange(len(X_s)))


def test_leaf_node_consistency():
    X, Y = _planted_data()
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.0))
    is_leaf = tree.feature < 0
    assert np.all(tree.left[is_leaf] == -1)
    assert np.all(tree.right[is_leaf] == -1)
    assert np.all(tree.left[~is_leaf] >= 0)
    assert np.all(tree.right[~is_leaf] >= 0)
    assert tree.n_leaves == int(is_leaf.sum())
    assert set(tree.leaf_id[is_leaf].tolist()) == set(range(tree.n_leaves))


# ---- the empty-honest-leaf guarantee (finding 1 / 5) -------------------------


def test_threshold_bounds_are_x_only_support_constraint():
    # lo = v_{k-1}, hi = v_{n_L-k} of each candidate's leaf-sample column.
    X = np.arange(20, dtype=float).reshape(20, 1)  # X[:,0] = 0..19
    l_idx = np.arange(20)
    bounds = _threshold_bounds(X, l_idx, np.array([0]), min_samples_leaf=5)
    assert bounds.shape == (1, 2)
    assert bounds[0, 0] == 4.0  # v_4
    assert bounds[0, 1] == 15.0  # v_{20-5}


def test_contract_check_rejects_leaf_infeasible_split():
    # Split sample balanced at t=0 (3|3), but leaf sample 1|3 -> contract breach.
    X = np.zeros((10, 1))
    X[[0, 1, 2], 0] = -1.0
    X[[3, 4, 5], 0] = 1.0  # s_idx rows
    X[[6], 0] = -1.0
    X[[7, 8, 9], 0] = 1.0  # l_idx rows
    split = Split(feature=0, threshold=0.0, score=1.0)
    with pytest.raises(ValueError, match="leaf-sample children"):
        _validate_split_contract(
            split,
            1,
            min_child=3,
            X=X,
            s_idx=np.array([0, 1, 2, 3, 4, 5]),
            l_idx=np.array([6, 7, 8, 9]),
            min_samples_leaf=3,
        )


def test_honest_tree_splits_on_clear_signal():
    # With the leaf-feasibility band, a clear planted shift is not stumped away.
    X, Y = _planted_data(seed=32, n=160)
    tree = _build(X, Y, CartCriterion(), TreeParams())  # honest defaults
    assert tree.n_leaves >= 2


@pytest.mark.parametrize("seed", [0, 7, 32, 101, 2024])
def test_no_empty_honest_leaves(seed):
    # Honest structure is grown on S but populated by L; every leaf must still
    # receive at least min_samples_leaf leaf-sample atoms (else 1/|L_k| is
    # undefined in weight assembly). seed=32, n=80 previously produced count 0.
    X, Y = _planted_data(seed=seed, n=80)
    params = TreeParams()  # honest defaults
    tree = _build(X, Y, CartCriterion(), params, seed=seed)
    counts = np.bincount(tree.leaf_sample_leaf, minlength=tree.n_leaves)
    assert counts.shape[0] == tree.n_leaves
    assert counts.min() >= params.min_samples_leaf
    assert (counts > 0).all()


# ---- routing -----------------------------------------------------------------


def test_apply_matches_independent_routing():
    X, Y = _planted_data()
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.0))

    def route_one(x):
        node = 0
        while tree.feature[node] >= 0:
            node = tree.left[node] if x[tree.feature[node]] <= tree.threshold[node] else tree.right[node]
        return tree.leaf_id[node]

    expected = np.array([route_one(x) for x in X], dtype=np.int32)
    assert np.array_equal(tree.apply(X), expected)


def test_cart_root_split_finds_the_step():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-1.0, 1.0, size=(n, 1))
    Y = np.where(X[:, 0] > 0.0, 1.0, -1.0)[:, None] + 0.01 * rng.normal(size=(n, 1))
    params = TreeParams(min_samples_leaf=5, alpha=0.0, honesty_fraction=0.0, colsample=1.0)
    tree = _build(X, Y, CartCriterion(), params)
    assert tree.feature[0] == 0
    assert abs(tree.threshold[0]) < 0.1


# ---- honesty -----------------------------------------------------------------


def test_honest_folds_are_disjoint_and_cover_the_subsample():
    X, Y = _planted_data(n=100)
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.5))
    s = set(tree.split_sample_rows.tolist())
    leaf = set(tree.leaf_sample_rows.tolist())
    assert s.isdisjoint(leaf)
    assert s | leaf == set(range(100))  # subsample_size == n here
    assert abs(len(s) - 50) <= 1


def test_fast_mode_uses_all_rows_for_both_folds():
    X, Y = _planted_data(n=100)
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.0))
    assert np.array_equal(np.sort(tree.split_sample_rows), np.arange(100))
    assert np.array_equal(tree.split_sample_rows, tree.leaf_sample_rows)


def test_leaf_populations_match_routing():
    X, Y = _planted_data(n=120)
    tree = _build(X, Y, CartCriterion(), TreeParams(honesty_fraction=0.5))
    assert tree.leaf_sample_rows.shape == tree.leaf_sample_leaf.shape
    assert tree.leaf_sample_leaf.min() >= 0
    assert tree.leaf_sample_leaf.max() < tree.n_leaves
    assert np.array_equal(tree.leaf_sample_leaf, tree.apply(X[tree.leaf_sample_rows]))


# ---- RNG: determinism & decoupled axes (finding 4) ---------------------------


def test_build_is_deterministic():
    X, Y = _planted_data()
    a = _build(X, Y, CartCriterion(), TreeParams(), seed=42)
    b = _build(X, Y, CartCriterion(), TreeParams(), seed=42)
    assert np.array_equal(a.feature, b.feature)
    assert np.allclose(a.threshold, b.threshold, equal_nan=True)
    assert np.array_equal(a.left, b.left)
    assert np.array_equal(a.leaf_sample_leaf, b.leaf_sample_leaf)


def test_node_stream_spawns_independent_structure_and_criterion_rngs():
    # The builder relies on spawning two independent sibling streams per node.
    gen = RngStreams(0).node(0, 0)
    g_struct, g_crit = gen.spawn(2)
    a = g_struct.normal(size=1000)
    b = g_crit.normal(size=1000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1
    # ...and reproducibly so.
    g2_struct, g2_crit = RngStreams(0).node(0, 0).spawn(2)
    assert np.array_equal(a, g2_struct.normal(size=1000))
    assert np.array_equal(b, g2_crit.normal(size=1000))


def test_mmd_rff_criterion_grows_a_tree():
    X, Y = _planted_data(n=150, d=2)
    crit = MmdRffCriterion.from_data(Y, n_features=64, bandwidth_rule=fixed_bandwidth(1.0))
    tree = _build(X, Y, crit, TreeParams(honesty_fraction=0.5), seed=1)
    assert isinstance(tree, DecisionTree)
    assert tree.n_leaves >= 2  # the planted shift should induce at least one split
    assert tree.apply(X).shape == (150,)
