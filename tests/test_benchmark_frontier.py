"""Offline smoke test for the Milestone-1 shrinkage-frontier benchmark.

Exercises the full benchmark pipeline (forest fit -> weights -> shrink -> metric
table) on a synthetic dataset, so it runs in CI without network access.
"""

import importlib.util
from pathlib import Path

import numpy as np

from benchmarks.real_datasets import RealDataset, split_arrays
from benchmarks.results_io import write_json_result
from drforest.datasets import make_gaussian_copula

_STUDIES = Path(__file__).resolve().parents[1] / "benchmarks" / "studies"


def _load_module(filename: str = "run_shrinkage_frontier.py"):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), _STUDIES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_run_on_synthetic_dataset(capsys, monkeypatch, tmp_path):
    module = _load_module()
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    payload = module.run(
        dataset="copula_small",
        seed=0,
        repeats=2,
        n_trees=8,
        n_features=32,
        shrink_features=64,
        results_dir=tmp_path,
    )

    out = capsys.readouterr().out
    assert "RMSE" in out
    assert "CRPS mean" in out
    assert "+marginal" in out
    assert payload["study"] == "run_shrinkage_frontier"
    assert list(tmp_path.glob("*.json"))


def test_benchmark_split_is_disjoint_and_covers_all_rows():
    module = _load_module()
    train, test = module._split(100, test_fraction=0.25, seed=3)

    assert test.size == 25
    assert train.size == 75
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(100))


def test_result_writer_does_not_require_seed_and_avoids_collisions(tmp_path):
    first = write_json_result("study", {"params": {}, "value": 1}, tmp_path)
    second = write_json_result("study", {"value": 2}, tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_ablation_grid_runs_on_synthetic_dataset(monkeypatch, tmp_path):
    module = _load_module("run_ablation.py")
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    payload = module.run(
        datasets=["copula_small"],
        seed=0,
        repeats=1,
        n_trees=8,
        n_features=32,
        shrink_features=64,
        max_cutpoints=8,
        results_dir=tmp_path,
    )

    runs = payload["datasets"][0]["runs"]
    assert {run["criterion"] for run in runs} == {"cart", "mmd_rff", "sliced_wasserstein"}
    for run in runs:
        assert {"raw", "marginal_kmse", "parent_stein"} <= set(run["variants"])
        assert "washout" in run
    washout = payload["datasets"][0]["washout_summary"]["cart"]
    assert washout["rho_bar_mean"] <= 1.0
    assert "single_tree_std_mean" in washout
    assert list(tmp_path.glob("*.json"))


def test_oracle_alpha_sweep_runs_on_synthetic_dataset(monkeypatch, tmp_path):
    module = _load_module("run_oracle_alpha.py")
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    payload = module.run(
        datasets=["copula_small"],
        criteria=["mmd_rff"],
        targets=["marginal", "parent"],
        alphas=[0.0, 0.25],
        seed=0,
        repeats=1,
        n_trees=8,
        n_features=32,
        max_cutpoints=8,
        results_dir=tmp_path,
    )

    runs = payload["datasets"][0]["runs"]
    assert payload["study"] == "run_oracle_alpha"
    assert set(runs[0]["variants"]) == {
        "raw",
        "marginal_alpha_0",
        "marginal_alpha_0.25",
        "parent_alpha_0",
        "parent_alpha_0.25",
    }
    assert {row["variant"] for row in payload["datasets"][0]["summary"]} >= {"raw", "parent_alpha_0.25"}
    assert list(tmp_path.glob("*.json"))


def test_synthetic_splitting_suite_runs_on_synthetic_dataset(monkeypatch, tmp_path):
    module = _load_module("run_synthetic_splitting.py")
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    payload = module.run(
        datasets=["copula_small"],
        criteria=["cart", "mmd_rff"],
        seed=0,
        repeats=1,
        n_trees=8,
        n_features=32,
        max_cutpoints=8,
        honesty_fractions=[0.5],
        results_dir=tmp_path,
    )

    assert payload["study"] == "run_synthetic_splitting"
    summary = payload["datasets"][0]["summary"]
    assert {row["criterion"] for row in summary} == {"cart", "mmd_rff"}
    assert "rho_bar_mean" in summary[0]["washout"]
    assert list(tmp_path.glob("*.json"))


def test_synthetic_splitting_suite_runs_adaptive_mmd(monkeypatch, tmp_path):
    module = _load_module("run_synthetic_splitting.py")
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    payload = module.run(
        datasets=["copula_small"],
        criteria=["adaptive_mmd"],
        seed=0,
        repeats=1,
        n_trees=8,
        n_features=16,
        max_cutpoints=8,
        honesty_fractions=[0.5, 0.0],
        adaptive_pool_features=32,
        adaptive_selected_features=4,
        results_dir=tmp_path,
    )

    summary = payload["datasets"][0]["summary"]
    assert {row["criterion"] for row in summary} == {"adaptive_mmd"}
    assert {row["honesty_fraction"] for row in summary} == {0.5, 0.0}
    assert len(summary[0]["split_feature_counts"]) == small.X.shape[1]
    assert sum(summary[0]["split_feature_counts"]) > 0
    assert payload["params"]["adaptive_pool_features"] == 32
    assert payload["params"]["adaptive_selected_features"] == 4


def test_real_dataset_split_arrays_is_deterministic():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    Y = np.arange(20, dtype=np.float64)

    first = split_arrays("toy", X, Y, n_train=12, n_test=5, seed=3)
    second = split_arrays("toy", X, Y, n_train=12, n_test=5, seed=3)

    assert isinstance(first, RealDataset)
    assert first.X_train.shape == (12, 2)
    assert first.Y_train.shape == (12, 1)
    assert first.X_test.shape == (5, 2)
    assert first.Y_test.shape == (5, 1)
    assert first.name == "toy"
    assert (first.X_train == second.X_train).all()
    assert (first.Y_test == second.Y_test).all()


def test_real_benchmark_runs_on_monkeypatched_dataset(monkeypatch, tmp_path):
    module = _load_module("run_real_benchmark.py")
    X = np.linspace(-1.0, 1.0, 180).reshape(90, 2)
    Y = (X[:, :1] ** 2) + 0.1 * X[:, 1:]

    def load_small(name, *, n_train, n_test, seed):
        return split_arrays(name, X, Y, n_train=n_train, n_test=n_test, seed=seed)

    monkeypatch.setattr(module, "make_real_dataset", load_small)

    payload = module.run(
        datasets=["toy_real"],
        criteria=["cart", "mmd_rff"],
        honesty_fractions=[0.5, 0.0],
        seed=0,
        repeats=1,
        n_train=50,
        n_test=20,
        n_trees=5,
        n_features=16,
        max_cutpoints=8,
        results_dir=tmp_path,
    )

    assert payload["study"] == "run_real_benchmark"
    assert len(payload["runs"]) == 4
    assert {row["honesty_fraction"] for row in payload["summary"]} == {0.5, 0.0}
    assert all(row["fit_time"] >= 0.0 for row in payload["runs"])
    assert all(row["weight_time"] >= 0.0 for row in payload["runs"])
    assert list(tmp_path.glob("*.json"))
