from typing import cast

import numpy as np
import pytest

from drforest.criteria.base import _best_split_on_feature, split_candidate_positions
from drforest.criteria.cart import CartCriterion
from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.criteria.sliced_wasserstein import (
    SlicedWassersteinCriterion,
    _best_split_on_feature_sliced,
    _sample_unit_directions,
    _sliced_wasserstein_sq,
    _wasserstein_1d_sq,
)
from drforest.features.rff import fixed_bandwidth, sample_rff


def brute_best_on_feature(x, Psi, scale, min_leaf):
    """O(n²) reference: evaluate every distinct-value split directly."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs, psi = x[order], Psi[order]
    best = None
    for i in range(n - 1):
        n_l = i + 1
        n_r = n - n_l
        if xs[i] == xs[i + 1] or n_l < min_leaf or n_r < min_leaf:
            continue
        diff = psi[:n_l].mean(0) - psi[n_l:].mean(0)
        score = scale * (n_l * n_r) / (n * n) * np.sum(np.abs(diff) ** 2)
        threshold = 0.5 * (xs[i] + xs[i + 1])
        if best is None or score > best[1]:
            best = (threshold, score)
    return best


def brute_wasserstein_1d_sq(left, right):
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    n_left = left_sorted.shape[0]
    n_right = right_sorted.shape[0]
    i = 0
    j = 0
    u = 0.0
    total = 0.0
    while i < n_left and j < n_right:
        next_left = (i + 1) / n_left
        next_right = (j + 1) / n_right
        u_next = min(next_left, next_right)
        total += (u_next - u) * float((left_sorted[i] - right_sorted[j]) ** 2)
        u = u_next
        if next_left <= u:
            i += 1
        if next_right <= u:
            j += 1
    return total


def test_streaming_matches_brute_force_mmd():
    rng = np.random.default_rng(0)
    n, dim = 60, 2
    x = rng.normal(size=n)
    Y = rng.normal(size=(n, dim))
    rff = sample_rff(dim, 64, 1.0, rng)
    Psi = rff.transform(Y)
    scale = 1.0 / rff.n_features

    streamed = _best_split_on_feature(x, Psi, scale, 5, -np.inf, np.inf)
    brute = brute_best_on_feature(x, Psi, scale, min_leaf=5)
    assert streamed is not None and brute is not None
    assert np.isclose(streamed[0], brute[0])
    assert np.isclose(streamed[1], brute[1])


def test_scaled_score_equals_kernel_mmd():
    """The RFF score equals the O(n²) kernel-matrix MMD for the *same* ω."""
    rng = np.random.default_rng(3)
    n, dim = 40, 2
    Y = rng.normal(size=(n, dim))
    rff = sample_rff(dim, 128, 0.8, rng)
    Psi = rff.transform(Y)
    B = rff.n_features

    # Empirical RFF kernel matrix K_ij = (1/B) Σ_b φ̃(y_i) conj(φ̃(y_j)).
    K = (Psi @ Psi.conj().T).real / B
    split = n // 2
    left, right = slice(0, split), slice(split, n)
    n_l, n_r = split, n - split
    mmd_sq = K[left, left].mean() + K[right, right].mean() - 2 * K[left, right].mean()
    expected = (n_l * n_r) / (n * n) * mmd_sq

    diff = Psi[left].mean(0) - Psi[right].mean(0)
    got = (1.0 / B) * (n_l * n_r) / (n * n) * np.sum(np.abs(diff) ** 2)
    assert np.isclose(got, expected)


def test_cart_reproduces_variance_reduction_split():
    """Identity-kernel criterion must pick the variance-reduction CART split."""
    rng = np.random.default_rng(7)
    n, p, dim = 80, 4, 3
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, dim))
    # Plant signal so a unique best split exists.
    Y += (X[:, 2] > 0.0)[:, None] * np.array([3.0, -2.0, 1.0])

    min_leaf = 5
    features = list(range(p))
    best = CartCriterion().best_split(
        X, Y, features, rng=np.random.default_rng(0), min_leaf=min_leaf, threshold_bounds=None
    )

    # Independent between-group-SS argmax over all features and thresholds.
    total_ss = np.sum((Y - Y.mean(0)) ** 2)
    ref_feat, ref_thr, ref_red = None, None, -np.inf
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], Y[order]
        for i in range(n - 1):
            n_l = i + 1
            if xs[i] == xs[i + 1] or n_l < min_leaf or n - n_l < min_leaf:
                continue
            within = np.sum((ys[:n_l] - ys[:n_l].mean(0)) ** 2) + np.sum((ys[n_l:] - ys[n_l:].mean(0)) ** 2)
            reduction = total_ss - within
            if reduction > ref_red:
                ref_feat, ref_thr, ref_red = f, 0.5 * (xs[i] + xs[i + 1]), reduction

    assert best is not None
    assert ref_feat is not None and ref_thr is not None
    assert best.feature == ref_feat
    assert np.isclose(best.threshold, ref_thr)


def test_no_valid_split_returns_none():
    crit = CartCriterion()
    X = np.zeros((6, 2))  # all identical feature values -> no distinct cut
    Y = np.random.default_rng(0).normal(size=(6, 2))
    assert crit.best_split(X, Y, [0, 1], np.random.default_rng(0), min_leaf=1, threshold_bounds=None) is None


def test_empty_features_is_a_wiring_error():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValueError, match="empty"):
        crit.best_split(X, Y, [], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_shape_mismatch_is_rejected():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.zeros((5, 2))
    with pytest.raises(ValueError, match="disagree on n"):
        crit.best_split(X, Y, [0], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_out_of_range_feature_is_rejected():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.zeros((6, 2))
    with pytest.raises(ValueError, match="out of range"):
        crit.best_split(X, Y, [0, 5], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_bool_feature_index_is_rejected():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.zeros((6, 2))
    with pytest.raises(TypeError, match="not bool"):
        crit.best_split(X, Y, [True], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_non_integer_feature_index_is_rejected():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.zeros((6, 2))
    # cast: deliberately feed a non-integer id to exercise the runtime guard.
    with pytest.raises(TypeError, match="must be an integer"):
        crit.best_split(X, Y, cast(list[int], [1.5]), np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_zero_dimensional_response_is_rejected():
    crit = CartCriterion()
    X = np.zeros((6, 2))
    Y = np.zeros((6, 0))
    with pytest.raises(ValueError, match="zero response dimensions"):
        crit.best_split(X, Y, [0], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)


def test_zero_dim_rff_is_rejected():
    with pytest.raises(ValueError, match="dim must be positive"):
        sample_rff(0, 16, 1.0, np.random.default_rng(0))


def test_threshold_bounds_restrict_to_band_and_pick_best_in_band():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    X = x[:, None]
    Y = rng.normal(size=(n, 2))
    lo, hi = -0.3, 0.4
    bounds = np.array([[lo, hi]], dtype=float)
    split = CartCriterion().best_split(X, Y, [0], np.random.default_rng(0), min_leaf=3, threshold_bounds=bounds)
    assert split is not None
    assert lo <= split.threshold < hi  # strict upper bound

    # Brute: best CART (scale=1) score among splits whose gap interval intersects
    # the band [lo, hi); threshold = midpoint of the feasible sub-interval.
    order = np.argsort(x)
    xs, ys = x[order], Y[order]
    best = None
    for i in range(n - 1):
        nl, nr = i + 1, n - (i + 1)
        lower, upper = max(xs[i], lo), min(xs[i + 1], hi)
        if xs[i] == xs[i + 1] or nl < 3 or nr < 3 or not (lower < upper):
            continue
        diff = ys[:nl].mean(0) - ys[nl:].mean(0)
        score = (nl * nr) / (n * n) * np.sum(diff**2)
        if best is None or score > best[1]:
            best = (0.5 * (lower + upper), score)
    assert best is not None
    assert np.isclose(split.threshold, best[0])


def test_bounded_sweep_uses_interval_intersection_not_midpoint():
    # S values {0, 10}, leaf band [9, 10): the split is feasible via t in [9, 10),
    # even though the gap midpoint 5 is outside the band (the old bug returned None).
    x = np.array([0.0, 10.0])
    Psi = np.array([[0.0], [1.0]])
    res = _best_split_on_feature(x, Psi, scale=1.0, min_leaf=1, lo=9.0, hi=10.0)
    assert res is not None
    threshold, _ = res
    assert 9.0 <= threshold < 10.0


def test_split_candidate_positions_exact_mode_returns_all_valid_boundaries():
    xs = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 5.0])
    positions = split_candidate_positions(xs, min_leaf=2, lo=-np.inf, hi=np.inf, max_cutpoints=None)
    assert positions.tolist() == [2, 3]


def test_split_candidate_positions_uses_all_boundaries_within_budget():
    xs = np.array([1.0] * 100 + [2.0, 3.0])
    positions = split_candidate_positions(xs, min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=5)
    assert positions.tolist() == [99, 100]


def test_split_candidate_positions_caps_by_rank_probes_and_repairs_plateaus():
    xs = np.array([1.0] * 100 + list(range(2, 22)), dtype=float)
    positions = split_candidate_positions(xs, min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=5)
    assert positions.tolist() == [101, 105, 108, 112, 116]


def test_split_candidate_positions_respects_bounds_before_snapping():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    positions = split_candidate_positions(xs, min_leaf=1, lo=1.5, hi=2.5, max_cutpoints=3)
    assert positions.tolist() == [1, 2]


def test_split_candidate_positions_caps_over_bounded_valid_span():
    xs = np.arange(200.0)
    positions = split_candidate_positions(xs, min_leaf=1, lo=50.0, hi=120.0, max_cutpoints=8)
    assert positions.tolist() == [54, 63, 72, 80, 89, 97, 106, 115]


def test_capped_mean_embedding_scores_only_shared_grid():
    x = np.arange(20.0)
    psi = np.zeros((20, 1), dtype=float)
    allowed = split_candidate_positions(np.sort(x, kind="stable"), min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=4)
    disallowed = np.setdiff1d(np.arange(19), allowed)
    best_disallowed = int(disallowed[len(disallowed) // 2])
    psi[best_disallowed + 1 :] = 100.0
    result = _best_split_on_feature(x, psi, scale=1.0, min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=4)
    assert result is not None
    threshold, _ = result
    assert int(np.floor(threshold)) in allowed


def test_capped_sliced_wasserstein_scores_only_shared_grid():
    x = np.arange(20.0)
    projected = np.zeros((20, 1), dtype=float)
    allowed = split_candidate_positions(np.sort(x, kind="stable"), min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=4)
    disallowed = np.setdiff1d(np.arange(19), allowed)
    best_disallowed = int(disallowed[len(disallowed) // 2])
    projected[best_disallowed + 1 :] = 100.0
    result = _best_split_on_feature_sliced(x, projected, min_leaf=1, lo=-np.inf, hi=np.inf, max_cutpoints=4)
    assert result is not None
    threshold, _ = result
    assert int(np.floor(threshold)) in allowed


def test_empty_band_yields_no_split():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 1))
    Y = rng.normal(size=(40, 2))
    bounds = np.array([[0.5, 0.5]])  # lo >= hi -> no admissible threshold
    assert CartCriterion().best_split(X, Y, [0], np.random.default_rng(0), min_leaf=2, threshold_bounds=bounds) is None


def test_threshold_bounds_shape_is_validated():
    X = np.zeros((10, 2))
    Y = np.random.default_rng(0).normal(size=(10, 2))
    bad = np.zeros((1, 2))  # only 1 band for 2 features
    with pytest.raises(ValueError, match="threshold_bounds"):
        CartCriterion().best_split(X, Y, [0, 1], np.random.default_rng(0), min_leaf=2, threshold_bounds=bad)


def test_mmd_from_data_sets_sigma():
    Y = np.random.default_rng(0).normal(size=(30, 2))
    crit = MmdRffCriterion.from_data(Y, n_features=32, bandwidth_rule=fixed_bandwidth(1.3))
    assert crit.sigma == 1.3
    assert crit.dim == 2
    assert crit.scale == 1.0 / 32


def test_wasserstein_1d_sq_matches_quantile_integral():
    left = np.array([0.0, 2.0])
    right = np.array([1.0, 3.0])
    assert np.isclose(_wasserstein_1d_sq(left, right), 1.0)

    unequal = _wasserstein_1d_sq(np.array([0.0]), np.array([2.0, 4.0]))
    # ∫_0^0.5 (0 - 2)^2 du + ∫_0.5^1 (0 - 4)^2 du
    assert np.isclose(unequal, 10.0)


def test_wasserstein_1d_sq_matches_reference_for_random_unequal_sizes():
    rng = np.random.default_rng(20)
    size_pairs = [(1, 2), (2, 7), (3, 11), (7, 19), (13, 29), (31, 47)]
    for n_left, n_right in size_pairs:
        for _ in range(50):
            left = rng.normal(size=n_left)
            right = rng.normal(size=n_right)
            assert np.isclose(
                _wasserstein_1d_sq(left, right),
                brute_wasserstein_1d_sq(left, right),
                rtol=1e-12,
                atol=1e-12,
            )


def test_sliced_wasserstein_one_dimensional_projection_is_not_repeated():
    rng = np.random.default_rng(21)
    n, p = 60, 3
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, 1))
    Y += (X[:, 2] > 0.0)[:, None] * 2.0

    many = SlicedWassersteinCriterion(n_projections=64, dim=1).best_split(
        X, Y, [0, 1, 2], np.random.default_rng(0), min_leaf=4, threshold_bounds=None
    )
    one = SlicedWassersteinCriterion(n_projections=1, dim=1).best_split(
        X, Y, [0, 1, 2], np.random.default_rng(1), min_leaf=4, threshold_bounds=None
    )
    ref = None
    for feature in [0, 1, 2]:
        found = _best_split_on_feature_sliced(X[:, feature], Y, min_leaf=4, lo=-np.inf, hi=np.inf)
        if found is None:
            continue
        threshold, score = found
        if ref is None or score > ref[2]:
            ref = (feature, threshold, score)

    assert many is not None
    assert one is not None
    assert ref is not None
    assert many.feature == one.feature == ref[0]
    assert np.isclose(many.threshold, one.threshold)
    assert np.isclose(many.threshold, ref[1])
    assert np.isclose(many.score, one.score)
    assert np.isclose(many.score, ref[2])


def test_sliced_wasserstein_sq_one_column_matches_1d_wasserstein():
    left = np.array([[0.0], [2.0]])
    right = np.array([[1.0], [3.0]])
    assert np.isclose(_sliced_wasserstein_sq(left, right), _wasserstein_1d_sq(left[:, 0], right[:, 0]))


def test_sliced_wasserstein_best_split_matches_projected_reference():
    rng = np.random.default_rng(11)
    n, p, dim = 50, 3, 2
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, dim))
    Y += (X[:, 1] > 0.0)[:, None] * np.array([0.0, 3.0])
    features = [0, 1, 2]
    seed = 123
    n_projections = 8

    split = SlicedWassersteinCriterion(n_projections=n_projections, dim=dim).best_split(
        X, Y, features, np.random.default_rng(seed), min_leaf=4, threshold_bounds=None
    )

    theta = _sample_unit_directions(dim, n_projections, np.random.default_rng(seed))
    projected = Y @ theta.T
    ref = None
    for feature in features:
        found = _best_split_on_feature_sliced(X[:, feature], projected, min_leaf=4, lo=-np.inf, hi=np.inf)
        if found is None:
            continue
        threshold, score = found
        if ref is None or score > ref[2]:
            ref = (feature, threshold, score)

    assert split is not None
    assert ref is not None
    assert split.feature == ref[0]
    assert np.isclose(split.threshold, ref[1])
    assert np.isclose(split.score, ref[2])


def test_sliced_wasserstein_honors_threshold_bounds():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 1))
    Y = rng.normal(size=(40, 2))
    bounds = np.array([[-0.2, 0.3]])
    split = SlicedWassersteinCriterion(n_projections=4, dim=2).best_split(
        X, Y, [0], np.random.default_rng(0), min_leaf=3, threshold_bounds=bounds
    )
    assert split is not None
    assert -0.2 <= split.threshold < 0.3


def test_sliced_wasserstein_from_data_sets_dimensions():
    Y = np.random.default_rng(0).normal(size=(30, 3))
    crit = SlicedWassersteinCriterion.from_data(Y, n_projections=7)
    assert crit.n_projections == 7
    assert crit.dim == 3


def test_sliced_wasserstein_rejects_response_dimension_mismatch():
    X = np.zeros((8, 1))
    Y = np.zeros((8, 2))
    crit = SlicedWassersteinCriterion(n_projections=3, dim=3)
    with pytest.raises(ValueError, match="response dimensions"):
        crit.best_split(X, Y, [0], np.random.default_rng(0), min_leaf=1, threshold_bounds=None)
