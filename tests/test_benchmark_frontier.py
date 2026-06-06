"""Offline smoke test for the Milestone-1 shrinkage-frontier benchmark.

Exercises the full benchmark pipeline (forest fit -> weights -> shrink -> metric
table) on a synthetic dataset, so it runs in CI without network access.
"""

import importlib.util
from pathlib import Path

from drforest.datasets import make_gaussian_copula

_STUDIES = Path(__file__).resolve().parents[1] / "benchmarks" / "studies"


def _load_module(filename: str = "run_shrinkage_frontier.py"):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), _STUDIES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_run_on_synthetic_dataset(capsys, monkeypatch):
    module = _load_module()
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    module.run(dataset="copula_small", seed=0, repeats=2, n_trees=8, n_features=32, shrink_features=64)

    out = capsys.readouterr().out
    assert "RMSE" in out
    assert "CRPS mean" in out
    assert "+marginal" in out


def test_benchmark_split_is_disjoint_and_covers_all_rows():
    module = _load_module()
    train, test = module._split(100, test_fraction=0.25, seed=3)

    assert test.size == 25
    assert train.size == 75
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(100))


def test_ablation_grid_runs_on_synthetic_dataset(capsys, monkeypatch):
    module = _load_module("run_ablation.py")
    small = make_gaussian_copula(n=120, p=4, d=2, seed=0)
    monkeypatch.setattr(module, "load_dataset", lambda name: small)

    module.run(
        datasets=["copula_small"],
        seed=0,
        repeats=1,
        n_trees=8,
        n_features=32,
        shrink_features=64,
    )

    out = capsys.readouterr().out
    for token in ("cart", "mmd_rff", "raw", "kmse", "stein"):
        assert token in out
