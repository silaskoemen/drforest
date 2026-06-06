from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.datasets import make_shrinkage_toy
from drforest.features.rff import fixed_bandwidth, sample_rff
from drforest.forest import DistributionalRandomForest
from drforest.metrics import mean_crps, rmse
from drforest.shrinkage import marginal_target, shrink
from drforest.targets import weighted_mean
from drforest.tree import TreeParams
from drforest.weights import embedding_norm_sq, mmd_to_target, n_eff


def _rff(Y, n_features=512, sigma=1.0, seed=0):
    return sample_rff(Y.shape[1], n_features, fixed_bandwidth(sigma)(Y), np.random.default_rng(seed))


def test_n_eff_matches_participation_ratio():
    W = csr_matrix([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25]])

    assert np.allclose(n_eff(W), np.array([1.0, 1.0 / (0.5**2 + 0.25**2 + 0.25**2)]))


def test_embedding_norm_and_mmd_to_identical_target():
    Y = np.array([[-1.0], [0.0], [1.0]])
    W = csr_matrix([[0.2, 0.3, 0.5]])
    rff = _rff(Y)

    norm_sq = embedding_norm_sq(W, Y, rff)
    distance = mmd_to_target(W, W, Y, rff)

    assert norm_sq.shape == (1,)
    assert 0.0 <= norm_sq[0] <= 1.0
    assert distance[0] == pytest.approx(0.0, abs=1e-14)


def test_marginal_target_is_uniform_distribution():
    target = marginal_target(4)

    assert target.shape == (1, 4)
    assert np.allclose(target.toarray(), np.full((1, 4), 0.25))


def test_marginal_target_rejects_invalid_size():
    with pytest.raises(ValueError, match=">= 1"):
        marginal_target(0)
    with pytest.raises(TypeError, match="not bool"):
        marginal_target(True)
    with pytest.raises(TypeError, match="not float"):
        marginal_target(cast(Any, 3.7))


def test_shrink_preserves_simplex_and_uses_marginal_convex_combination():
    Y = np.array([[-1.0], [0.0], [1.0], [2.0]])
    W = csr_matrix([[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    rff = _rff(Y)

    result = shrink(W, Y, rff=rff)
    got = result.weights.to_csr().toarray()
    expected = (1.0 - result.alpha)[:, None] * W.toarray() + result.alpha[:, None] * 0.25

    assert got.shape == W.shape
    assert np.all(got >= 0.0)
    assert np.allclose(got.sum(axis=1), 1.0)
    assert np.allclose(got, expected)
    assert np.all((0.0 <= result.alpha) & (result.alpha <= 1.0))


def test_shrink_identical_conditional_and_target_sets_alpha_to_one():
    Y = np.array([[-1.0], [0.0], [1.0], [2.0]])
    W = marginal_target(Y.shape[0])
    rff = _rff(Y)

    result = shrink(W, Y, rff=rff)

    assert result.alpha[0] == pytest.approx(1.0)
    assert np.allclose(result.weights.to_csr().toarray(), W.toarray())


def test_shrink_large_effective_sample_drives_alpha_down():
    rng = np.random.default_rng(3)
    left = rng.normal(-2.0, 0.15, size=(100, 1))
    right = rng.normal(2.0, 0.15, size=(100, 1))
    Y = np.vstack([left, right])
    rff = _rff(Y, n_features=1024, sigma=0.8, seed=4)

    W_small = csr_matrix([np.r_[np.full(5, 0.2), np.zeros(195)]])
    W_large = csr_matrix([np.r_[np.full(100, 0.01), np.zeros(100)]])

    small = shrink(W_small, Y, rff=rff)
    large = shrink(W_large, Y, rff=rff)

    assert n_eff(W_large)[0] > n_eff(W_small)[0]
    assert large.alpha[0] < small.alpha[0]
    assert large.alpha[0] < 0.1


def test_shrink_rejects_unknown_target():
    Y = np.array([[0.0], [1.0]])
    W = csr_matrix([[0.5, 0.5]])
    rff = _rff(Y)

    with pytest.raises(ValueError, match="only 'marginal'"):
        shrink(W, Y, rff=rff, target="parent")  # type: ignore[arg-type]


def test_shrink_rejects_unknown_parameterization():
    Y = np.array([[0.0], [1.0]])
    W = csr_matrix([[0.5, 0.5]])
    rff = _rff(Y)

    with pytest.raises(ValueError, match="kmse.*stein"):
        shrink(W, Y, rff=rff, parameterization="oracle")


def test_stein_parameterization_shrinks_at_least_as_much_as_kmse():
    # stein uses the bias-corrected D̂² = MMD² - V, so its α = V/MMD² is always
    # >= the kmse α = V/(V+MMD²); they coincide only as V/MMD² -> 0.
    Y = np.array([[-1.0], [0.0], [1.0], [2.0]])
    W = csr_matrix([[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    rff = _rff(Y)

    kmse = shrink(W, Y, rff=rff, parameterization="kmse").alpha
    stein = shrink(W, Y, rff=rff, parameterization="stein").alpha

    assert np.all(stein >= kmse - 1e-12)
    assert np.all((0.0 <= stein) & (stein <= 1.0))


def test_stein_alpha_is_one_when_conditional_equals_target():
    Y = np.array([[-1.0], [0.0], [1.0], [2.0]])
    W = marginal_target(Y.shape[0])
    rff = _rff(Y)

    # MMD² -> 0 with conditional == target, so V/MMD² clips up to 1.
    assert shrink(W, Y, rff=rff, parameterization="stein").alpha[0] == pytest.approx(1.0)


def test_shrunk_weights_recompute_metrics_on_distribution_shift_toy():
    rng = np.random.default_rng(19)
    X = rng.uniform(-1.0, 1.0, size=(300, 1))
    scale = np.where(X[:, 0] < 0.0, 0.25, 1.5)
    Y = rng.normal(scale=scale[:, None], size=(300, 1))
    W = csr_matrix(
        [
            np.where(X[:, 0] < 0.0, 1.0 / np.sum(X[:, 0] < 0.0), 0.0),
            np.where(X[:, 0] >= 0.0, 1.0 / np.sum(X[:, 0] >= 0.0), 0.0),
        ]
    )
    Y_true = np.array([[0.0], [1.5]])
    rff = _rff(Y, n_features=512, sigma=1.0, seed=9)

    raw_rmse = rmse(Y_true, weighted_mean(W, Y))
    raw_crps = mean_crps(W, Y, Y_true)
    result = shrink(W, Y, rff=rff)
    shrunk_rmse = rmse(Y_true, weighted_mean(result.weights, Y))
    shrunk_crps = mean_crps(result.weights, Y, Y_true)

    assert np.isfinite(raw_rmse)
    assert np.isfinite(raw_crps)
    assert np.isfinite(shrunk_rmse)
    assert np.isfinite(shrunk_crps)


def test_step7_mmd_rff_forest_raw_vs_marginal_shrinkage_signal():
    dataset = make_shrinkage_toy(n=260, seed=29)
    X, Y = dataset.X, dataset.Y

    forest = DistributionalRandomForest(
        criterion_factory=lambda Y: MmdRffCriterion.from_data(
            Y,
            n_features=96,
            bandwidth_rule=fixed_bandwidth(1.2),
        ),
        seed=12,
        n_trees=18,
        subsample=0.7,
        tree_params=TreeParams(
            min_samples_leaf=6,
            alpha=0.02,
            honesty_fraction=0.5,
            colsample=1.0,
        ),
    ).fit(X, Y)
    X_test, Y_test = X[:36], Y[:36]
    W = forest.weights(X_test)
    rff = _rff(Y, n_features=384, sigma=1.2, seed=44)

    raw = {
        "rmse": rmse(Y_test, weighted_mean(W, Y)),
        "crps": mean_crps(W, Y, Y_test),
    }
    result = shrink(W, Y, rff=rff)
    shrunk = {
        "rmse": rmse(Y_test, weighted_mean(result.weights, Y)),
        "crps": mean_crps(result.weights, Y, Y_test),
    }

    assert W.shape == result.weights.shape == (Y_test.shape[0], Y.shape[0])
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(np.asarray(result.weights.to_csr().sum(axis=1)).ravel(), 1.0)
    assert np.all((0.0 <= result.alpha) & (result.alpha <= 1.0))
    assert result.alpha.mean() > 0.0
    assert set(raw) == set(shrunk) == {"rmse", "crps"}
    assert all(np.isfinite(score) and score >= 0.0 for score in raw.values())
    assert all(np.isfinite(score) and score >= 0.0 for score in shrunk.values())
    assert any(not np.isclose(raw[name], shrunk[name]) for name in raw)
