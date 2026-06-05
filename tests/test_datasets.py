from typing import Any, cast

import numpy as np
import pytest

from drforest.datasets import (
    list_datasets,
    load_dataset,
    make_gaussian_copula,
    make_heterogeneous_regression,
    make_quantile_scenario,
)


def test_dataset_registry_lists_synthetic_and_external_entries():
    all_names = {info.name for info in list_datasets()}
    synthetic_names = {info.name for info in list_datasets(kind="synthetic")}
    external_names = {info.name for info in list_datasets(kind="external")}

    assert {"paper_quantile_1", "paper_quantile_2", "paper_quantile_3", "paper_copula"} <= synthetic_names
    assert {"jura", "slump", "wq", "enb", "birth1", "wage", "air"} <= external_names
    assert synthetic_names | external_names == all_names


def test_load_dataset_rejects_unknown_and_unimplemented_external_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("missing")
    with pytest.raises(NotImplementedError, match="external dataset fetching"):
        load_dataset("jura")


def test_quantile_scenarios_are_deterministic_and_shaped():
    first = make_quantile_scenario(2, n=100, p=7, seed=4)
    second = load_dataset("paper_quantile_2", n=100, p=7, seed=4)

    assert first.name == "paper_quantile_2"
    assert first.X.shape == (100, 7)
    assert first.Y.shape == (100, 1)
    assert first.feature_names == tuple(f"x{j + 1}" for j in range(7))
    assert first.response_names == ("y",)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.Y, second.Y)


def test_quantile_scenario_3_changes_shape_not_first_two_moments():
    dataset = make_quantile_scenario(3, n=6000, p=4, seed=8)
    left = dataset.Y[dataset.X[:, 0] <= 0.0, 0]
    right = dataset.Y[dataset.X[:, 0] > 0.0, 0]

    assert abs(np.mean(left) - np.mean(right)) < 0.08
    assert abs(np.var(left) - np.var(right)) < 0.12
    assert np.mean(left < 0.0) > 0.1
    assert np.mean(right < 0.0) == 0.0


def test_gaussian_copula_has_x1_dependent_correlation():
    dataset = make_gaussian_copula(n=4000, p=5, d=3, seed=11)
    low = dataset.Y[dataset.X[:, 0] < 0.2]
    high = dataset.Y[dataset.X[:, 0] > 0.8]

    low_corr = np.corrcoef(low[:, 0], low[:, 1])[0, 1]
    high_corr = np.corrcoef(high[:, 0], high[:, 1])[0, 1]
    assert low_corr < 0.3
    assert high_corr > 0.7


def test_heterogeneous_regression_returns_treatment_outcome_response():
    dataset = make_heterogeneous_regression(n=120, p=3, seed=5)

    assert dataset.X.shape == (120, 3)
    assert dataset.Y.shape == (120, 2)
    assert dataset.response_names == ("treatment", "outcome")
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.Y).all()


def test_dataset_loaders_reject_invalid_sizes():
    with pytest.raises(ValueError, match="scenario"):
        make_quantile_scenario(cast(Any, 4), n=10, p=2, seed=0)
    with pytest.raises(ValueError, match="p must be >= 2"):
        make_heterogeneous_regression(n=10, p=1, seed=0)
    with pytest.raises(ValueError, match="d must be >= 2"):
        make_gaussian_copula(n=10, p=2, d=1, seed=0)
