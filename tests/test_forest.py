from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from drforest import DistributionalRandomForest
from drforest.criteria import (
    AdaptiveMmdCriterion,
    AnisotropicMmdCriterion,
    CartCriterion,
    SlicedWassersteinCriterion,
)
from drforest.criteria.mmd_rff import MmdRffCriterion
from drforest.features.rff import median_heuristic
from drforest.metrics import mean_crps, mean_energy_score, rmse
from drforest.targets import weighted_cdf, weighted_mean, weighted_quantile
from drforest.tree import TreeParams, build_tree
from drforest.weights import assemble_weights


def _planted_data(seed=7, n=400, p=3, d=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, d)) + (X[:, 0] > 0)[:, None] * np.array([3.0, -2.0])
    return X, Y


def _cart_forest(**kw):
    return DistributionalRandomForest(
        criterion="cart",
        random_state=kw.pop("random_state", 0),
        n_estimators=kw.pop("n_estimators", 25),
        **kw,
    )


def _symmetric_noise(rng, n, scale):
    z = rng.normal(scale=scale, size=(n // 2, 1))
    y = np.vstack([z, -z])
    return y[rng.permutation(n)]


def _distribution_shift_toys(seed=31, n_per_regime=300):
    """Three same-mean regimes: tight, high-variance, and bimodal."""
    rng = np.random.default_rng(seed)
    x_low = rng.uniform(-1.0, -0.5, size=(n_per_regime, 1))
    x_high = rng.uniform(-0.25, 0.25, size=(n_per_regime, 1))
    x_bimodal = rng.uniform(0.5, 1.0, size=(n_per_regime, 1))

    y_low = _symmetric_noise(rng, n_per_regime, scale=0.15)
    y_high = _symmetric_noise(rng, n_per_regime, scale=2.0)
    z = rng.normal(scale=0.10, size=(n_per_regime // 2, 1))
    y_bimodal = np.vstack([-2.0 + z, 2.0 - z])
    y_bimodal = y_bimodal[rng.permutation(n_per_regime)]

    X = np.vstack([x_low, x_high, x_bimodal])
    Y = np.vstack([y_low, y_high, y_bimodal])
    order = rng.permutation(X.shape[0])
    return X[order], Y[order], y_high, y_bimodal


def _mmd_toy_forest():
    return DistributionalRandomForest(
        criterion_factory=lambda Y: MmdRffCriterion.from_data(Y, n_features=256, bandwidth_rule=median_heuristic),
        random_state=17,
        n_estimators=80,
        subsample=0.8,
        min_samples_leaf=8,
        alpha=0.02,
        honesty_fraction=0.5,
        colsample=1.0,
    )


# ---- construction validation -------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"n_estimators": 0}, {"subsample": 0.0}, {"subsample": 1.5}])
def test_invalid_forest_params_rejected(kwargs):
    with pytest.raises(ValueError):
        DistributionalRandomForest(criterion="cart", random_state=0, **kwargs)


def test_builtin_criterion_overrides_factory():
    called = False

    def factory(Y):
        nonlocal called
        called = True
        return CartCriterion()

    X, Y = _planted_data(n=80)
    forest = DistributionalRandomForest(
        criterion="cart",
        criterion_factory=factory,
        n_estimators=2,
        random_state=0,
    ).fit(X, Y)

    assert isinstance(forest.criterion_, CartCriterion)
    assert not called


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("mmd", MmdRffCriterion),
        ("mmd_rff", MmdRffCriterion),
        ("cart", CartCriterion),
        ("anisotropic_mmd", AnisotropicMmdCriterion),
        ("adaptive_mmd", AdaptiveMmdCriterion),
        ("sliced_wasserstein", SlicedWassersteinCriterion),
    ],
)
def test_builtin_criterion_names(name, expected_type):
    X, Y = _planted_data(n=80)
    forest = DistributionalRandomForest(
        criterion=name,
        n_estimators=1,
        max_cutpoints=2,
        random_state=0,
    ).fit(X, Y)

    assert isinstance(forest.criterion_, expected_type)


def test_default_criterion_is_mmd():
    X, Y = _planted_data(n=80)
    forest = DistributionalRandomForest(n_estimators=1, max_cutpoints=2, random_state=0).fit(X, Y)
    assert isinstance(forest.criterion_, MmdRffCriterion)


def test_unknown_builtin_criterion_rejected_at_fit():
    X, Y = _planted_data(n=80)
    forest = DistributionalRandomForest(criterion="unknown", n_estimators=2, random_state=0)
    with pytest.raises(ValueError, match="unknown criterion"):
        forest.fit(X, Y)


def test_non_string_criterion_rejected_at_fit():
    X, Y = _planted_data(n=80)
    forest = DistributionalRandomForest(criterion=cast(Any, 42), n_estimators=1, random_state=0)
    with pytest.raises(TypeError, match="criterion must be a string or None"):
        forest.fit(X, Y)


def test_criterion_factory_must_return_criterion():
    X, Y = _planted_data(n=80)
    factory = cast(Any, lambda y: object())
    forest = DistributionalRandomForest(criterion_factory=factory, n_estimators=1, random_state=0)
    with pytest.raises(TypeError, match="criterion_factory must return a Criterion"):
        forest.fit(X, Y)


def test_flattened_tree_parameters_are_forwarded():
    forest = DistributionalRandomForest(
        criterion="cart",
        min_samples_leaf=7,
        alpha=0.2,
        honesty_fraction=0.4,
        colsample=0.6,
        max_cutpoints=17,
    )

    assert forest.tree_params == TreeParams(
        min_samples_leaf=7,
        alpha=0.2,
        honesty_fraction=0.4,
        colsample=0.6,
        max_cutpoints=17,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples_leaf": 0},
        {"alpha": -0.1},
        {"alpha": 0.6},
        {"honesty_fraction": -0.1},
        {"honesty_fraction": 1.0},
        {"colsample": 0.0},
        {"colsample": 1.1},
        {"max_cutpoints": 0},
    ],
)
def test_invalid_flattened_tree_parameters_rejected(kwargs):
    forest = DistributionalRandomForest(criterion="cart", **kwargs)
    with pytest.raises((TypeError, ValueError)):
        _ = forest.tree_params


def test_inference_valid_flag_reflects_mode():
    cf = lambda Y: CartCriterion()  # noqa: E731
    # Fixed-fraction subsampling is not sublinear -> never inference-valid yet.
    honest = DistributionalRandomForest(criterion_factory=cf, random_state=0)
    assert honest.inference_valid is False
    # Fast/bootstrap/no-alpha configs are also clearly not inference-valid.
    no_honesty = DistributionalRandomForest(criterion_factory=cf, random_state=0, honesty_fraction=0.0)
    no_alpha = DistributionalRandomForest(criterion_factory=cf, random_state=0, alpha=0.0)
    assert no_honesty.inference_valid is False
    assert no_alpha.inference_valid is False
    boot = DistributionalRandomForest(criterion_factory=cf, random_state=0, bootstrap=True)
    assert boot.inference_valid is False


def test_weights_before_fit_raises():
    forest = _cart_forest()
    with pytest.raises(RuntimeError, match="not fitted"):
        forest.weights(np.zeros((3, 3)))


@pytest.mark.parametrize("method", ["predict", "predict_quantiles", "predict_cdf"])
def test_prediction_methods_before_fit_raise(method):
    forest = _cart_forest()
    X = np.zeros((3, 3))
    with pytest.raises(RuntimeError, match="not fitted"):
        if method == "predict":
            forest.predict(X)
        elif method == "predict_quantiles":
            forest.predict_quantiles(X, [0.5])
        else:
            forest.predict_cdf(X, [0.0])


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "match"),
    [
        (lambda X: X[:, 0], lambda Y: Y, "X must be 2-D"),
        (lambda X: X, lambda Y: Y[:, :, None], "y must be 1-D or 2-D"),
        (lambda X: X[:-1], lambda Y: Y, "X and y disagree"),
        (lambda X: np.empty((X.shape[0], 0)), lambda Y: Y, "zero features"),
        (lambda X: X, lambda Y: np.empty((Y.shape[0], 0)), "zero response dimensions"),
    ],
)
def test_fit_rejects_invalid_public_input_shapes(X_transform, y_transform, match):
    X, Y = _planted_data(n=80)
    with pytest.raises(ValueError, match=match):
        _cart_forest(n_estimators=1).fit(X_transform(X), y_transform(Y))


def test_fit_rejects_non_finite_predictors_before_configuring_criterion():
    X, Y = _planted_data(n=80)
    X[0, 0] = np.nan
    called = False

    def factory(y):
        nonlocal called
        called = True
        return CartCriterion()

    forest = DistributionalRandomForest(criterion_factory=factory, n_estimators=1, random_state=0)
    with pytest.raises(ValueError, match="X contains non-finite"):
        forest.fit(X, Y)
    assert not called


def test_non_finite_test_points_rejected():
    X, Y = _planted_data()
    forest = _cart_forest().fit(X, Y)
    X_bad = X[:10].copy()
    X_bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        forest.weights(X_bad)
    X_bad[2, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        forest.weights(X_bad)


def test_fit_validates_data_before_configuring_criterion():
    # A non-finite Y must raise the data error, not reach the criterion factory.
    called = {"factory": False}

    def factory(Y):
        called["factory"] = True
        return CartCriterion()

    X, Y = _planted_data(n=120)
    Y_bad = Y.copy()
    Y_bad[0, 0] = np.nan
    forest = DistributionalRandomForest(criterion_factory=factory, random_state=0, n_estimators=5, subsample=0.5)
    with pytest.raises(ValueError, match="non-finite"):
        forest.fit(X, Y_bad)
    assert not called["factory"]


def test_infeasible_subsample_raises_with_guidance():
    X, Y = _planted_data(n=40)  # ceil(0.1*40)=4 -> honest folds below min_samples_leaf=5
    forest = _cart_forest(subsample=0.1)
    with pytest.raises(ValueError, match="subsample too small"):
        forest.fit(X, Y)


def test_subsample_size_is_ceil_fraction_times_n():
    forest = _cart_forest(subsample=0.5)
    assert forest.subsample_size(100) == 50
    assert forest.subsample_size(201) == 101  # ceil(100.5)


# ---- W correctness -----------------------------------------------------------


def test_weights_are_row_stochastic():
    X, Y = _planted_data()
    forest = _cart_forest().fit(X, Y)
    W = forest.weights(X[:50])
    assert isinstance(W, csr_matrix)
    assert W.shape == (50, X.shape[0])
    assert W.min() >= 0.0
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)


def test_weights_row_stochastic_even_for_stumps():
    # Tiny subsample (s_n = ceil(0.04*200) = 8 < 2*min_child) -> trees cannot
    # split -> every leaf is the root; rows must still sum to 1.
    X, Y = _planted_data(n=200)
    forest = _cart_forest(subsample=0.04, honesty_fraction=0.0).fit(X, Y)
    assert max(t.n_leaves for t in forest.trees) == 1  # all stumps
    W = forest.weights(X[:20])
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)


def test_bootstrap_draws_with_replacement_and_stays_row_stochastic():
    X, Y = _planted_data(n=300)
    forest = _cart_forest(bootstrap=True, random_state=4).fit(X, Y)
    # With replacement, the per-tree subsample (S + L global rows) has duplicates.
    used = np.concatenate([forest.trees[0].split_sample_rows, forest.trees[0].leaf_sample_rows])
    assert np.unique(used).shape[0] < used.shape[0]
    W = forest.weights(X[:20])
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
    assert W.min() >= 0.0


def test_bootstrap_false_has_no_duplicate_rows():
    X, Y = _planted_data(n=300)
    forest = _cart_forest(bootstrap=False, random_state=4).fit(X, Y)
    used = np.concatenate([forest.trees[0].split_sample_rows, forest.trees[0].leaf_sample_rows])
    assert np.unique(used).shape[0] == used.shape[0]


def test_single_tree_weights_match_manual():
    X, Y = _planted_data(n=300)
    tree = build_tree(
        X,
        Y,
        CartCriterion(),
        TreeParams(honesty_fraction=0.5),
        subsample_size=300,
        tree_rng=np.random.default_rng(3),
        node_rng=lambda nid: np.random.default_rng(1000 + nid),
    )
    X_test = X[:10]
    W = assemble_weights([tree], X_test, n_train=300).toarray()

    leaf_of_test = tree.apply(X_test)
    for i, leaf in enumerate(leaf_of_test):
        atoms = tree.leaf_sample_rows[tree.leaf_sample_leaf == leaf]
        expected = np.zeros(300)
        expected[atoms] = 1.0 / atoms.shape[0]
        assert np.allclose(W[i], expected)


def test_weights_average_over_trees():
    # Two-tree forest: each atom's weight is the mean of the two trees' weights.
    X, Y = _planted_data(n=300)
    p = X.shape[1]
    trees = [
        build_tree(
            X,
            Y,
            CartCriterion(),
            TreeParams(honesty_fraction=0.5),
            subsample_size=300,
            tree_rng=np.random.default_rng(s),
            node_rng=lambda nid, s=s: np.random.default_rng(10 * s + nid),
        )
        for s in (1, 2)
    ]
    X_test = X[:5]
    combined = assemble_weights(trees, X_test, 300).toarray()
    w0 = assemble_weights([trees[0]], X_test, 300).toarray()
    w1 = assemble_weights([trees[1]], X_test, 300).toarray()
    assert np.allclose(combined, 0.5 * (w0 + w1))
    assert p == 3  # guard the planted-data shape


# ---- determinism & criteria --------------------------------------------------


def test_fit_is_deterministic():
    X, Y = _planted_data()
    w_a = _cart_forest(random_state=11).fit(X, Y).weights(X[:30]).toarray()
    w_b = _cart_forest(random_state=11).fit(X, Y).weights(X[:30]).toarray()
    assert np.array_equal(w_a, w_b)
    w_c = _cart_forest(random_state=12).fit(X, Y).weights(X[:30]).toarray()
    assert not np.array_equal(w_a, w_c)


def test_random_state_none_uses_generated_seed(monkeypatch):
    X, Y = _planted_data(n=120)
    monkeypatch.setattr("drforest.forest.secrets.randbits", lambda bits: 23)

    generated = _cart_forest(random_state=None, n_estimators=4).fit(X, Y)
    explicit = _cart_forest(random_state=23, n_estimators=4).fit(X, Y)

    assert np.array_equal(
        generated.predict_weights(X[:20]).toarray(),
        explicit.predict_weights(X[:20]).toarray(),
    )


def test_mmd_rff_forest_weights_are_valid():
    X, Y = _planted_data(n=400, d=2)
    forest = DistributionalRandomForest(
        criterion_factory=lambda Y: MmdRffCriterion.from_data(Y, n_features=128, bandwidth_rule=median_heuristic),
        random_state=0,
        n_estimators=20,
        subsample=0.5,
    ).fit(X, Y)
    W = forest.weights(X[:40])
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
    assert W.min() >= 0.0


def test_conditional_mean_tracks_planted_shift():
    # Sanity that W is informative: E[Y|X] via W should separate the two regimes.
    X, Y = _planted_data(n=600)
    forest = _cart_forest(n_estimators=50, subsample=0.5).fit(X, Y)
    pos = X[X[:, 0] > 0.5][:30]
    neg = X[X[:, 0] < -0.5][:30]
    mean_pos = forest.weights(pos) @ Y
    mean_neg = forest.weights(neg) @ Y
    # Planted shift adds [3, -2] when x0 > 0.
    assert mean_pos[:, 0].mean() - mean_neg[:, 0].mean() > 1.5
    assert mean_pos[:, 1].mean() - mean_neg[:, 1].mean() < -1.0


def test_step6_targets_and_metrics_run_on_forest_weights():
    X, Y = _planted_data(n=250)
    forest = _cart_forest(n_estimators=20, subsample=0.6).fit(X, Y)
    X_test, Y_test = X[:30], Y[:30]
    W = forest.weights(X_test)

    mean_pred = weighted_mean(W, Y)
    quantiles = weighted_quantile(W, Y, np.array([0.1, 0.5, 0.9]))
    score_rmse = rmse(Y_test, mean_pred)
    score_crps = mean_crps(W, Y, Y_test)
    score_energy = mean_energy_score(W, Y, Y_test)

    assert mean_pred.shape == Y_test.shape
    assert quantiles.shape == (Y_test.shape[0], Y_test.shape[1], 3)
    assert np.isfinite(score_rmse) and score_rmse >= 0.0
    assert np.isfinite(score_crps) and score_crps >= 0.0
    assert np.isfinite(score_energy) and score_energy >= 0.0


def test_estimator_predictions_store_targets_and_preserve_univariate_shapes():
    X, Y = _planted_data(n=180, d=1)
    y = Y[:, 0]
    forest = _cart_forest(n_estimators=8, subsample=0.6).fit(X, y)
    X_test = X[:12]

    W = forest.predict_weights(X_test)
    mean = forest.predict(X_test)
    quantiles = forest.predict_quantiles(X_test, np.array([0.1, 0.5, 0.9]))
    cdf = forest.predict_cdf(X_test, np.array([-1.0, 0.0, 1.0]))

    assert forest.y_train_.shape == (y.shape[0], 1)
    assert not np.shares_memory(forest.y_train_, y)
    assert forest.n_features_in_ == X.shape[1]
    assert forest.n_outputs_ == 1
    assert forest.estimators_ is forest.trees
    assert np.array_equal(W.toarray(), forest.weights(X_test).toarray())
    assert mean.shape == (X_test.shape[0],)
    assert quantiles.shape == (X_test.shape[0], 3)
    assert cdf.shape == (X_test.shape[0], 3)
    assert np.allclose(mean, weighted_mean(W, Y)[:, 0])
    assert np.array_equal(quantiles, weighted_quantile(W, Y, np.array([0.1, 0.5, 0.9]))[:, 0, :])
    assert np.allclose(cdf, weighted_cdf(W, Y, np.array([-1.0, 0.0, 1.0]))[:, 0, :])
    assert forest.predict_quantiles(X_test, 0.5).shape == (X_test.shape[0], 1)
    assert forest.predict_cdf(X_test, 0.0).shape == (X_test.shape[0], 1)


def test_multivariate_prediction_shapes():
    X, Y = _planted_data(n=180, d=2)
    forest = _cart_forest(n_estimators=8, subsample=0.6).fit(X, Y)
    X_test = X[:12]
    W = forest.predict_weights(X_test)
    quantiles = forest.predict_quantiles(X_test, np.array([0.25, 0.75]))
    cdf = forest.predict_cdf(X_test, np.array([0.0]))

    assert forest.predict(X_test).shape == (12, 2)
    assert quantiles.shape == (12, 2, 2)
    assert cdf.shape == (12, 2, 1)
    assert np.array_equal(quantiles, weighted_quantile(W, Y, np.array([0.25, 0.75])))
    assert np.allclose(cdf, weighted_cdf(W, Y, np.array([0.0])))


def test_refit_replaces_target_shape_and_fitted_state():
    X, Y = _planted_data(n=180, d=2)
    forest = _cart_forest(n_estimators=4, subsample=0.6)

    assert not forest.is_fitted
    forest.fit(X, Y[:, 0])
    assert forest.is_fitted
    assert forest.predict(X[:5]).shape == (5,)

    forest.fit(X, Y)
    assert forest.n_outputs_ == 2
    assert forest.y_train_.shape == Y.shape
    assert forest.predict(X[:5]).shape == (5, 2)


def test_prediction_rejects_wrong_feature_count():
    X, Y = _planted_data(n=120)
    forest = _cart_forest(n_estimators=2).fit(X, Y)
    with pytest.raises(ValueError, match="features"):
        forest.predict(X[:5, :-1])


def test_step6_qrf_scenario1_quantiles_widen_under_variance_shift():
    X, Y, _, _ = _distribution_shift_toys()
    forest = _mmd_toy_forest().fit(X, Y)

    low = np.full((40, 1), -0.75)
    high = np.full((40, 1), 0.0)
    q_low = weighted_quantile(forest.weights(low), Y, np.array([0.1, 0.9]))
    q_high = weighted_quantile(forest.weights(high), Y, np.array([0.1, 0.9]))

    low_width = np.mean(q_low[:, 0, 1] - q_low[:, 0, 0])
    high_width = np.mean(q_high[:, 0, 1] - q_high[:, 0, 0])
    assert high_width > 3.0 * low_width


def test_step6_qrf_scenario2_crps_prefers_matching_variance_regime():
    X, Y, y_high, _ = _distribution_shift_toys()
    forest = _mmd_toy_forest().fit(X, Y)
    y_true = y_high[:80]

    high = np.full((y_true.shape[0], 1), 0.0)
    low = np.full((y_true.shape[0], 1), -0.75)
    local_score = mean_crps(forest.weights(high), Y, y_true)
    wrong_regime_score = mean_crps(forest.weights(low), Y, y_true)

    assert local_score < 0.8 * wrong_regime_score


def test_step6_qrf_scenario3_quantiles_track_bimodal_shift_with_centered_mean():
    X, Y, _, y_bimodal = _distribution_shift_toys()
    forest = _mmd_toy_forest().fit(X, Y)

    bimodal = np.full((80, 1), 0.75)
    W_bimodal = forest.weights(bimodal)
    q_bimodal = weighted_quantile(W_bimodal, Y, np.array([0.1, 0.9]))
    mean_bimodal = weighted_mean(W_bimodal, Y)

    assert np.mean(q_bimodal[:, 0, 0]) < -1.0
    assert np.mean(q_bimodal[:, 0, 1]) > 1.0
    assert abs(np.mean(mean_bimodal)) < 0.35

    local_score = mean_crps(W_bimodal, Y, y_bimodal[:80])
    low_score = mean_crps(forest.weights(np.full((80, 1), -0.75)), Y, y_bimodal[:80])
    assert local_score < 0.8 * low_score
