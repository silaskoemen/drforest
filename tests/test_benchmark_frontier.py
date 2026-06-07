"""Offline smoke test for the Milestone-1 shrinkage-frontier benchmark.

Exercises the full benchmark pipeline (forest fit -> weights -> shrink -> metric
table) on a synthetic dataset, so it runs in CI without network access.
"""

import importlib.util
from pathlib import Path

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
