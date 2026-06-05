import numpy as np
import pytest
from scipy.sparse import csr_matrix

from drforest.metrics.crps import _abs_deviation_profile, componentwise_crps
from drforest.mixture import MixtureWeights
from drforest.targets import weighted_cdf, weighted_mean


def _random_base(rng, n_test, n_train, support=5):
    base = np.zeros((n_test, n_train))
    for r in range(n_test):
        cols = rng.choice(n_train, min(support, n_train), replace=False)
        w = rng.random(cols.size)
        base[r, cols] = w / w.sum()
    return csr_matrix(base)


def _uniform_target(n_train):
    return csr_matrix(np.full((1, n_train), 1.0 / n_train))


def test_shape_and_uniform_marginal_detection():
    rng = np.random.default_rng(0)
    base = _random_base(rng, 4, 10)
    alpha = rng.random(4)
    mix = MixtureWeights(base=base, alpha=alpha, target=_uniform_target(10))

    assert mix.shape == (4, 10)
    assert mix.n_test == 4 and mix.n_train == 10
    assert mix.is_uniform_marginal

    perturbed = np.full((1, 10), 1.0 / 10)
    perturbed[0, 0] += 0.05
    perturbed[0, 1] -= 0.05
    assert not MixtureWeights(base=base, alpha=alpha, target=csr_matrix(perturbed)).is_uniform_marginal


def test_to_csr_matches_manual_convex_combination():
    rng = np.random.default_rng(1)
    base = _random_base(rng, 5, 12)
    alpha = rng.random(5)
    mix = MixtureWeights(base=base, alpha=alpha, target=_uniform_target(12))

    manual = (1.0 - alpha)[:, None] * base.toarray() + alpha[:, None] * (1.0 / 12)
    assert np.allclose(mix.to_csr().toarray(), manual)
    assert np.allclose(mix.to_csr().sum(axis=1), 1.0)


def test_apply_matches_to_csr_matmul_for_broadcast_and_full_targets():
    rng = np.random.default_rng(2)
    base = _random_base(rng, 6, 9)
    alpha = rng.random(6)
    M = rng.normal(size=(9, 3))

    broadcast = MixtureWeights(base=base, alpha=alpha, target=_uniform_target(9))
    full_target = _random_base(rng, 6, 9, support=4)
    per_row = MixtureWeights(base=base, alpha=alpha, target=full_target)

    assert np.allclose(broadcast.apply(M), broadcast.to_csr() @ M)
    assert np.allclose(per_row.apply(M), per_row.to_csr() @ M)


def test_apply_rejects_bad_operand():
    mix = MixtureWeights(
        base=_random_base(np.random.default_rng(3), 2, 4), alpha=np.zeros(2), target=_uniform_target(4)
    )
    with pytest.raises(ValueError, match="M must be 2-D"):
        mix.apply(np.zeros(4))
    with pytest.raises(ValueError, match="n_train"):
        mix.apply(np.zeros((3, 2)))


def test_deviation_profile_matches_brute_force_and_uu():
    rng = np.random.default_rng(4)
    atoms = rng.normal(size=37)
    deviation, uu = _abs_deviation_profile(atoms)

    queries = rng.normal(size=11)
    brute = np.array([np.sum(np.abs(q - atoms)) for q in queries])
    assert np.allclose(deviation(queries), brute)

    brute_uu = np.mean(np.abs(atoms[:, None] - atoms[None, :]))
    assert uu == pytest.approx(brute_uu)


def test_weighted_mean_and_cdf_match_dense_materialization():
    rng = np.random.default_rng(5)
    n_train, d = 30, 2
    Y = rng.normal(size=(n_train, d))
    base = _random_base(rng, 7, n_train)
    alpha = rng.random(7)
    mix = MixtureWeights(base=base, alpha=alpha, target=_uniform_target(n_train))
    dense = mix.to_csr()

    assert np.allclose(weighted_mean(mix, Y), weighted_mean(dense, Y))

    thresholds = np.linspace(-1.5, 1.5, 6)
    assert np.allclose(weighted_cdf(mix, Y, thresholds), weighted_cdf(dense, Y, thresholds))


def test_componentwise_crps_uniform_mixture_matches_dense():
    rng = np.random.default_rng(6)
    n_train, d = 40, 3
    Y = rng.normal(size=(n_train, d))
    base = _random_base(rng, 8, n_train)
    alpha = rng.random(8)
    mix = MixtureWeights(base=base, alpha=alpha, target=_uniform_target(n_train))
    Y_true = rng.normal(size=(8, d))

    assert mix.is_uniform_marginal
    assert np.allclose(componentwise_crps(mix, Y, Y_true), componentwise_crps(mix.to_csr(), Y, Y_true))


def test_componentwise_crps_non_uniform_target_falls_back_correctly():
    rng = np.random.default_rng(7)
    n_train, d = 25, 2
    Y = rng.normal(size=(n_train, d))
    base = _random_base(rng, 5, n_train)
    alpha = rng.random(5)
    target = _random_base(rng, 1, n_train, support=6)
    mix = MixtureWeights(base=base, alpha=alpha, target=target)
    Y_true = rng.normal(size=(5, d))

    assert not mix.is_uniform_marginal
    # Non-uniform target must materialize and match the CSR computation exactly.
    assert np.allclose(componentwise_crps(mix, Y, Y_true), componentwise_crps(mix.to_csr(), Y, Y_true))


def test_uniform_mixture_targets_do_not_materialize(monkeypatch):
    rng = np.random.default_rng(8)
    n_train, d = 20, 2
    Y = rng.normal(size=(n_train, d))
    mix = MixtureWeights(
        base=_random_base(rng, 4, n_train),
        alpha=rng.random(4),
        target=_uniform_target(n_train),
    )
    Y_true = rng.normal(size=(4, d))

    def _no_densify(self):
        raise AssertionError("to_csr was called: the factored path densified")

    monkeypatch.setattr(MixtureWeights, "to_csr", _no_densify)
    weighted_mean(mix, Y)
    weighted_cdf(mix, Y, np.array([0.0, 1.0]))
    componentwise_crps(mix, Y, Y_true)


def test_duplicate_index_target_is_not_uniform_and_scores_correctly():
    # Uncanonicalized CSR with indices [0, 0, 1, 2] is dense [0.5, 0.25, 0.25, 0],
    # not uniform; the fast path must not be taken.
    target = csr_matrix((np.full(4, 0.25), np.array([0, 0, 1, 2]), np.array([0, 4])), shape=(1, 4))
    assert not target.has_canonical_format
    base = csr_matrix(np.array([[1.0, 0.0, 0.0, 0.0]]))
    mix = MixtureWeights(base=base, alpha=np.array([0.3]), target=target)

    assert not mix.is_uniform_marginal
    Y = np.array([[0.0], [1.0], [2.0], [3.0]])
    Y_true = np.array([[1.5]])
    assert np.allclose(componentwise_crps(mix, Y, Y_true), componentwise_crps(mix.to_csr(), Y, Y_true))


def test_constructor_coerces_non_csr_inputs():
    rng = np.random.default_rng(11)
    base = _random_base(rng, 3, 5)
    alpha = rng.random(3)
    coo_target = _uniform_target(5).tocoo()
    mix = MixtureWeights(base=base.tocsc(), alpha=alpha, target=coo_target)

    assert mix.base.format == "csr" and mix.target.format == "csr"
    assert mix.is_uniform_marginal
    M = rng.normal(size=(5, 2))
    assert np.allclose(mix.apply(M), mix.to_csr() @ M)


def test_constructor_copies_alpha_against_later_mutation():
    rng = np.random.default_rng(12)
    alpha = rng.random(3)
    mix = MixtureWeights(base=_random_base(rng, 3, 5), alpha=alpha, target=_uniform_target(5))

    alpha[:] = 5.0  # would violate the [0, 1] invariant if stored by reference
    assert np.all((0.0 <= mix.alpha) & (mix.alpha <= 1.0))


def test_validation_rejects_malformed_mixtures():
    rng = np.random.default_rng(9)
    base = _random_base(rng, 3, 6)

    with pytest.raises(ValueError, match="alpha must lie in"):
        MixtureWeights(base=base, alpha=np.array([0.0, 1.2, 0.5]), target=_uniform_target(6))
    with pytest.raises(ValueError, match="alpha must have shape"):
        MixtureWeights(base=base, alpha=np.zeros(2), target=_uniform_target(6))

    bad_target = csr_matrix(np.full((1, 6), 0.1))  # rows sum to 0.6, not 1
    with pytest.raises(ValueError, match="rows must sum to 1"):
        MixtureWeights(base=base, alpha=np.zeros(3), target=bad_target)

    with pytest.raises(ValueError, match="target has 5 columns"):
        MixtureWeights(base=base, alpha=np.zeros(3), target=_uniform_target(5))
