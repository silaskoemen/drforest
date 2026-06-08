import importlib

import numpy as np
import pytest

from drforest.criteria.adaptive_mmd import _best_split_on_feature_adaptive
from drforest.criteria.base import _best_split_on_feature, split_candidate_positions
from drforest.criteria.sliced_wasserstein import _best_split_on_feature_sliced

rust_core = pytest.importorskip("drforest._drforest_core", reason="Rust extension is not built")


def test_rust_split_candidate_positions_matches_python_random_cases():
    rng = np.random.default_rng(90)
    for _ in range(100):
        n = int(rng.integers(8, 80))
        min_leaf = int(rng.integers(1, max(2, n // 4)))
        xs = np.sort(rng.normal(size=n), kind="stable")
        lo, hi = np.quantile(xs, [0.1, 0.9])
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 16))

        expected = split_candidate_positions(
            xs,
            min_leaf=min_leaf,
            lo=float(lo),
            hi=float(hi),
            max_cutpoints=max_cutpoints,
        )
        got = rust_core.split_candidate_positions(xs, min_leaf, float(lo), float(hi), max_cutpoints)

        assert got == expected.tolist()


def test_rust_cart_split_matches_python_embedding_sweep_random_cases():
    rng = np.random.default_rng(91)
    for _ in range(50):
        n = int(rng.integers(20, 100))
        dim = int(rng.integers(1, 5))
        x = rng.normal(size=n)
        Y = rng.normal(size=(n, dim))
        min_leaf = int(rng.integers(2, 8))
        lo, hi = -np.inf, np.inf
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 20))

        expected = _best_split_on_feature(
            x,
            Y,
            scale=1.0,
            min_leaf=min_leaf,
            lo=lo,
            hi=hi,
            max_cutpoints=max_cutpoints,
        )
        got = rust_core.best_cart_split_one_feature(
            x,
            Y,
            dim,
            min_leaf,
            lo,
            hi,
            max_cutpoints,
        )

        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert np.isclose(got[0], expected[0])
            assert np.isclose(got[1], expected[1])


def test_rust_cart_split_matches_python_multi_feature_random_cases():
    rng = np.random.default_rng(915)
    for _ in range(50):
        n = int(rng.integers(20, 100))
        p = int(rng.integers(2, 8))
        dim = int(rng.integers(1, 5))
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=(n, dim))
        features = rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist()
        min_leaf = int(rng.integers(2, 8))
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 15))
        bounds = np.column_stack(
            [
                rng.normal(loc=-0.5, scale=0.3, size=len(features)),
                rng.normal(loc=0.5, scale=0.3, size=len(features)),
            ]
        )
        bounds.sort(axis=1)

        expected = None
        for j, feature in enumerate(features):
            found = _best_split_on_feature(
                X[:, feature],
                Y,
                scale=1.0,
                min_leaf=min_leaf,
                lo=float(bounds[j, 0]),
                hi=float(bounds[j, 1]),
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                continue
            threshold, score = found
            if expected is None or score > expected[2]:
                expected = (feature, threshold, score)

        got = rust_core.best_cart_split(
            X,
            Y,
            features,
            min_leaf,
            bounds,
            max_cutpoints,
        )

        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == expected[0]
            assert np.isclose(got[1], expected[1])
            assert np.isclose(got[2], expected[2])


def test_rust_complex_embedding_split_matches_python_random_cases():
    rng = np.random.default_rng(92)
    for _ in range(50):
        n = int(rng.integers(20, 90))
        p = int(rng.integers(2, 8))
        n_embed = int(rng.integers(3, 20))
        X = rng.normal(size=(n, p))
        psi = rng.normal(size=(n, n_embed)) + 1j * rng.normal(size=(n, n_embed))
        features = rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist()
        min_leaf = int(rng.integers(2, 8))
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 15))
        bounds = np.column_stack(
            [
                rng.normal(loc=-0.5, scale=0.3, size=len(features)),
                rng.normal(loc=0.5, scale=0.3, size=len(features)),
            ]
        )
        bounds.sort(axis=1)
        scale = 1.0 / n_embed

        expected = None
        for j, feature in enumerate(features):
            found = _best_split_on_feature(
                X[:, feature],
                psi,
                scale=scale,
                min_leaf=min_leaf,
                lo=float(bounds[j, 0]),
                hi=float(bounds[j, 1]),
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                continue
            threshold, score = found
            if expected is None or score > expected[2]:
                expected = (feature, threshold, score)

        got = rust_core.best_complex_embedding_split(
            X,
            np.ascontiguousarray(np.real(psi)),
            np.ascontiguousarray(np.imag(psi)),
            features,
            scale,
            min_leaf,
            bounds,
            max_cutpoints,
        )

        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == expected[0]
            assert np.isclose(got[1], expected[1])
            assert np.isclose(got[2], expected[2])


def test_rust_adaptive_complex_embedding_split_matches_python_random_cases():
    rng = np.random.default_rng(916)
    for _ in range(50):
        n = int(rng.integers(20, 90))
        p = int(rng.integers(2, 8))
        n_embed = int(rng.integers(3, 20))
        selected_features = int(rng.integers(1, n_embed + 1))
        X = rng.normal(size=(n, p))
        psi = rng.normal(size=(n, n_embed)) + 1j * rng.normal(size=(n, n_embed))
        features = rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist()
        min_leaf = int(rng.integers(2, 8))
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 15))
        bounds = np.column_stack(
            [
                rng.normal(loc=-0.5, scale=0.3, size=len(features)),
                rng.normal(loc=0.5, scale=0.3, size=len(features)),
            ]
        )
        bounds.sort(axis=1)

        expected = None
        for j, feature in enumerate(features):
            found = _best_split_on_feature_adaptive(
                X[:, feature],
                psi,
                selected_features=selected_features,
                min_leaf=min_leaf,
                lo=float(bounds[j, 0]),
                hi=float(bounds[j, 1]),
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                continue
            threshold, score = found
            if expected is None or score > expected[2]:
                expected = (feature, threshold, score)

        got = rust_core.best_adaptive_complex_embedding_split(
            X,
            np.ascontiguousarray(np.real(psi)),
            np.ascontiguousarray(np.imag(psi)),
            selected_features,
            features,
            min_leaf,
            bounds,
            max_cutpoints,
        )

        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == expected[0]
            assert np.isclose(got[1], expected[1])
            assert np.isclose(got[2], expected[2])


def test_rust_sliced_wasserstein_split_matches_python_random_cases():
    rng = np.random.default_rng(917)
    for _ in range(30):
        n = int(rng.integers(20, 70))
        p = int(rng.integers(2, 7))
        n_projections = int(rng.integers(1, 8))
        X = rng.normal(size=(n, p))
        projected = rng.normal(size=(n, n_projections))
        features = rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist()
        min_leaf = int(rng.integers(2, 8))
        max_cutpoints = None if rng.uniform() < 0.4 else int(rng.integers(1, 12))
        bounds = np.column_stack(
            [
                rng.normal(loc=-0.5, scale=0.3, size=len(features)),
                rng.normal(loc=0.5, scale=0.3, size=len(features)),
            ]
        )
        bounds.sort(axis=1)

        expected = None
        for j, feature in enumerate(features):
            found = _best_split_on_feature_sliced(
                X[:, feature],
                projected,
                min_leaf=min_leaf,
                lo=float(bounds[j, 0]),
                hi=float(bounds[j, 1]),
                max_cutpoints=max_cutpoints,
            )
            if found is None:
                continue
            threshold, score = found
            if expected is None or score > expected[2]:
                expected = (feature, threshold, score)

        got = rust_core.best_sliced_wasserstein_split(
            X,
            projected,
            features,
            min_leaf,
            bounds,
            max_cutpoints,
        )

        if expected is None:
            assert got is None
        else:
            assert got is not None
            assert got[0] == expected[0]
            assert np.isclose(got[1], expected[1])
            assert np.isclose(got[2], expected[2])


def test_rust_binding_reports_invalid_min_leaf_like_python():
    with pytest.raises(ValueError, match="min_leaf"):
        rust_core.split_candidate_positions(np.array([0.0, 1.0]), 0, -np.inf, np.inf, None)


def test_rust_extension_module_name_is_importable_when_built():
    assert importlib.import_module("drforest._drforest_core") is rust_core
