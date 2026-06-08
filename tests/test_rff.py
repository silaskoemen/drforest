import numpy as np
import pytest

from drforest.features.rff import (
    coordinatewise_median_heuristic,
    fixed_bandwidth,
    fixed_bandwidths,
    median_heuristic,
    sample_diagonal_rff,
    sample_rff,
)


def gaussian_kernel(u, v, sigma):
    return np.exp(-np.sum((u - v) ** 2) / (2.0 * sigma**2))


def test_rff_approximates_gaussian_kernel():
    # (1/B) Σ_b φ̃(u) conj(φ̃(v)) -> k(u, v) as B grows.
    rng = np.random.default_rng(0)
    dim, sigma, n_features = 3, 1.5, 400_000
    rff = sample_rff(dim, n_features, sigma, rng)

    u = rng.normal(size=dim)
    v = rng.normal(size=dim)
    phi = rff.transform(np.stack([u, v]))  # (2, B) complex
    approx = np.mean(phi[0] * np.conj(phi[1]))

    assert abs(approx.imag) < 1e-2  # estimate is real in expectation
    assert abs(approx.real - gaussian_kernel(u, v, sigma)) < 1e-2


def test_rff_diagonal_is_one():
    rng = np.random.default_rng(1)
    rff = sample_rff(2, 1000, 0.7, rng)
    y = rng.normal(size=(5, 2))
    phi = rff.transform(y)
    self_inner = np.mean(phi * np.conj(phi), axis=1)  # k(y, y) = 1 exactly
    assert np.allclose(self_inner.real, 1.0)
    assert np.allclose(self_inner.imag, 0.0)


def test_median_heuristic_positive_and_matches_definition():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(50, 3))
    sigma = median_heuristic(y)
    dists = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(-1))
    expected = np.median(dists[np.triu_indices(len(y), k=1)])
    assert sigma > 0
    assert np.isclose(sigma, expected)


def test_fixed_bandwidth_is_constant():
    rule = fixed_bandwidth(2.0)
    assert rule(np.zeros((10, 2))) == 2.0
    assert rule(np.ones((3, 5))) == 2.0


def test_coordinatewise_median_heuristic_matches_columns():
    y = np.array([[0.0, 0.0], [2.0, 10.0], [4.0, 20.0]])
    got = coordinatewise_median_heuristic(y)

    assert np.allclose(got, [2.0, 10.0])


def test_fixed_bandwidths_validates_dimension():
    rule = fixed_bandwidths(np.array([1.0, 2.0]))

    assert np.allclose(rule(np.zeros((5, 2))), [1.0, 2.0])
    with pytest.raises(ValueError, match="response dimensions"):
        rule(np.zeros((5, 3)))


def test_diagonal_rff_matches_scalar_rff_in_one_dimension():
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    y = np.linspace(-1.0, 1.0, 9)[:, None]

    scalar = sample_rff(1, 16, 1.7, rng_a)
    diagonal = sample_diagonal_rff(np.array([1.7]), 16, rng_b)

    assert np.allclose(diagonal.omega, scalar.omega)
    assert np.allclose(diagonal.transform(y), scalar.transform(y))
