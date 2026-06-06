import importlib.util
import urllib.error
from pathlib import Path
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
from drforest.datasets.external import (
    is_fetchable,
    processed_path,
    read_mtr_dataset,
    write_processed,
)


def _load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "fetch_mtr.py"
    spec = importlib.util.spec_from_file_location("fetch_mtr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Network/checksum/ARFF-parsing logic lives in the fetch script, not the package.
_FETCH = _load_fetch_script()
parse_mtr_arff = _FETCH.parse_mtr_arff

_TOY_ARFF = (
    b"@relation toy\n"
    b"@attribute a numeric\n"
    b"@attribute b numeric\n"
    b"@attribute t1 numeric\n"
    b"@attribute t2 numeric\n"
    b"@data\n"
    b"1.0,2.0,3.0,4.0\n"
    b"5.0,6.0,7.0,8.0\n"
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


def test_parse_mtr_arff_splits_trailing_targets():
    X, Y, feature_names, response_names = parse_mtr_arff(_TOY_ARFF, n_targets=2)

    assert np.array_equal(X, [[1.0, 2.0], [5.0, 6.0]])
    assert np.array_equal(Y, [[3.0, 4.0], [7.0, 8.0]])
    assert feature_names == ("a", "b")
    assert response_names == ("t1", "t2")
    assert X.flags["C_CONTIGUOUS"] and Y.flags["C_CONTIGUOUS"]


def test_parse_mtr_arff_rejects_too_many_targets():
    with pytest.raises(ValueError, match="no feature columns"):
        parse_mtr_arff(_TOY_ARFF, n_targets=4)


def test_enb_is_fetchable_and_registered_with_shapes():
    assert is_fetchable("enb")
    assert not is_fetchable("jura")  # registered but no fetcher wired
    (enb_info,) = [info for info in list_datasets(kind="external") if info.name == "enb"]
    assert enb_info.n_features == 8
    assert enb_info.n_responses == 2


def test_load_dataset_external_rejects_generation_kwargs():
    with pytest.raises(TypeError, match="takes no generation kwargs"):
        load_dataset("enb", n=100)


def test_read_mtr_dataset_missing_points_at_fetch_script(tmp_path, monkeypatch):
    monkeypatch.setenv("DRFOREST_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="benchmarks/data/fetch_mtr.py enb"):
        read_mtr_dataset("enb")


def test_load_dataset_external_missing_raises_offline(tmp_path, monkeypatch):
    # load_dataset never reaches the network: an unmaterialised dataset fails loudly.
    monkeypatch.setenv("DRFOREST_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_mtr.py enb"):
        load_dataset("enb")


def test_processed_round_trip_reads_back_identically(tmp_path, monkeypatch):
    # write_processed -> read_mtr_dataset round-trips bit-for-bit, reading only
    # the processed npz (use the wired 'enb' name with toy arrays so no network).
    monkeypatch.setenv("DRFOREST_DATA_DIR", str(tmp_path))

    X, Y, feature_names, response_names = parse_mtr_arff(_TOY_ARFF, n_targets=2)
    write_processed("enb", X, Y, feature_names, response_names)

    rx, ry, rfn, rrn = read_mtr_dataset("enb")
    assert np.array_equal(rx, X) and np.array_equal(ry, Y)
    assert rfn == feature_names and rrn == response_names
    assert processed_path("enb") == tmp_path / "processed" / "enb.npz"


@pytest.mark.network
def test_fetch_enb_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DRFOREST_DATA_DIR", str(tmp_path))
    try:
        _FETCH.fetch("enb")
    except urllib.error.URLError as exc:
        pytest.skip(f"network unavailable: {exc}")

    assert (tmp_path / "raw" / "enb.arff").exists()
    assert (tmp_path / "processed" / "enb.npz").exists()

    X, Y, _, response_names = read_mtr_dataset("enb")
    assert X.shape == (768, 8)
    assert Y.shape == (768, 2)
    assert response_names == ("Y1", "Y2")
    assert np.isfinite(X).all() and np.isfinite(Y).all()


def test_fetch_recipes_cover_advertised_fetchable_names():
    # the fetch script's drift guard must keep recipes and EXTERNAL_MTR_NAMES aligned
    from drforest.datasets.external import EXTERNAL_MTR_NAMES

    assert set(_FETCH.MTR_SOURCES) == set(EXTERNAL_MTR_NAMES)
