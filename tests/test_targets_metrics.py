import numpy as np
import pytest
from scipy.sparse import csr_matrix

from drforest.metrics import (
    componentwise_crps,
    energy_score,
    mean_crps,
    mean_energy_score,
    rmse,
)
from drforest.targets import weighted_cdf, weighted_mean, weighted_quantile


def test_weighted_mean_matches_manual_sparse_product():
    W = csr_matrix([[1.0, 0.0, 0.0], [0.25, 0.25, 0.5]])
    Y = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 40.0]])

    got = weighted_mean(W, Y)

    expected = np.array([[1.0, 10.0], [3.5, 27.5]])
    assert np.allclose(got, expected)


def test_weighted_mean_normalizes_small_row_sum_drift():
    W = csr_matrix([[2.0, 0.0], [1.0, 1.0]])
    Y = np.array([[1.0], [5.0]])

    assert np.allclose(weighted_mean(W, Y), np.array([[1.0], [3.0]]))


def test_weighted_quantile_uses_inverse_empirical_cdf_per_output():
    W = csr_matrix([[0.2, 0.3, 0.5], [0.0, 0.6, 0.4]])
    Y = np.array([[10.0, 0.0], [20.0, -1.0], [30.0, 2.0]])

    got = weighted_quantile(W, Y, np.array([0.5, 0.9]))

    assert got.shape == (2, 2, 2)
    assert np.allclose(got[0, 0], [20.0, 30.0])
    assert np.allclose(got[1, 0], [20.0, 30.0])
    assert np.allclose(got[0, 1], [0.0, 2.0])
    assert np.allclose(got[1, 1], [-1.0, 2.0])


def test_weighted_quantile_rejects_invalid_quantiles():
    W = csr_matrix([[1.0]])
    Y = np.array([[1.0]])

    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        weighted_quantile(W, Y, np.array([-0.1]))
    with pytest.raises(ValueError, match="must not be empty"):
        weighted_quantile(W, Y, np.array([]))


def test_weighted_quantile_ignores_zero_weight_atoms_at_q_zero():
    W = csr_matrix([[0.0, 1.0, 0.0]])
    Y = np.array([[-100.0], [3.0], [100.0]])

    got = weighted_quantile(W, Y, np.array([0.0, 1.0]))

    assert np.allclose(got[0, 0], [3.0, 3.0])


def test_weighted_cdf_matches_manual_indicators():
    W = csr_matrix([[0.25, 0.75], [1.0, 0.0]])
    Y = np.array([[1.0, 10.0], [3.0, 5.0]])

    got = weighted_cdf(W, Y, np.array([2.0, 10.0]))

    expected = np.array(
        [
            [[0.25, 1.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
        ]
    )
    assert got.shape == (2, 2, 2)
    assert np.allclose(got, expected)


def test_targets_reject_malformed_weights_and_responses():
    Y = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="negative"):
        weighted_mean(csr_matrix([[1.0, -0.1]]), Y)
    with pytest.raises(ValueError, match="positive total"):
        weighted_mean(csr_matrix([[0.0, 0.0]]), Y)
    with pytest.raises(ValueError, match="W has 3 columns"):
        weighted_mean(csr_matrix([[1.0, 0.0, 0.0]]), Y)
    with pytest.raises(ValueError, match="non-finite"):
        weighted_mean(csr_matrix([[1.0, 0.0]]), np.array([[1.0], [np.nan]]))


def test_rmse_matches_numpy_definition():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 4.0], [1.0, 4.0]])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.0))


def test_rmse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.zeros((2, 1)), np.zeros((2, 2)))


def test_componentwise_crps_for_one_hot_distribution_is_absolute_error():
    W = csr_matrix([[0.0, 1.0]])
    Y_train = np.array([[1.0, 10.0], [4.0, 20.0]])
    Y_true = np.array([[2.5, 17.0]])

    got = componentwise_crps(W, Y_train, Y_true)

    assert np.allclose(got, np.array([[1.5, 3.0]]))
    assert mean_crps(W, Y_train, Y_true) == pytest.approx(2.25)


def test_componentwise_crps_matches_manual_two_atom_formula():
    W = csr_matrix([[0.25, 0.75]])
    Y_train = np.array([[0.0], [2.0]])
    Y_true = np.array([[1.0]])

    # Σ w_i |x_i-y| - 0.5 Σ_ij w_i w_j |x_i-x_j|
    expected = 1.0 - 0.5 * (2.0 * 0.25 * 0.75 * 2.0)
    assert componentwise_crps(W, Y_train, Y_true)[0, 0] == pytest.approx(expected)


def test_componentwise_crps_rejects_bad_truth_shape():
    W = csr_matrix([[1.0, 0.0]])
    Y_train = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="Y_true must have shape"):
        componentwise_crps(W, Y_train, np.array([[1.0, 2.0]]))


def test_energy_score_for_one_hot_distribution_is_euclidean_error():
    W = csr_matrix([[0.0, 1.0]])
    Y_train = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y_true = np.array([[0.0, 0.0]])

    got = energy_score(W, Y_train, Y_true)

    assert got == pytest.approx(np.array([5.0]))
    assert mean_energy_score(W, Y_train, Y_true) == pytest.approx(5.0)


def test_univariate_energy_score_matches_crps():
    W = csr_matrix([[0.25, 0.75]])
    Y_train = np.array([[0.0], [2.0]])
    Y_true = np.array([[1.0]])

    assert energy_score(W, Y_train, Y_true)[0] == pytest.approx(componentwise_crps(W, Y_train, Y_true)[0, 0])


def test_energy_score_matches_manual_two_atom_formula():
    W = csr_matrix([[0.25, 0.75]])
    Y_train = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y_true = np.array([[0.0, 4.0]])

    first = 0.25 * 4.0 + 0.75 * 3.0
    second = 0.5 * (2.0 * 0.25 * 0.75 * 5.0)
    assert energy_score(W, Y_train, Y_true)[0] == pytest.approx(first - second)
