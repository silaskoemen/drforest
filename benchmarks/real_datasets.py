"""Small real-data loader layer for benchmark studies.

The loaders are intentionally lazy about optional dependencies. Offline tests
can monkeypatch ``REAL_DATASETS`` without importing scikit-learn, while real
benchmark runs fail with a clear message when the dependency or remote dataset
is unavailable.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealDataset:
    name: str
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


RealDatasetLoader = Callable[[int, int, int], RealDataset]


def make_real_dataset(name: str, *, n_train: int, n_test: int, seed: int) -> RealDataset:
    if name not in REAL_DATASETS:
        available = ", ".join(sorted(REAL_DATASETS))
        raise ValueError(f"unknown real benchmark dataset {name!r}; available: {available}")
    return REAL_DATASETS[name](n_train, n_test, seed)


def split_arrays(name: str, X: np.ndarray, Y: np.ndarray, *, n_train: int, n_test: int, seed: int) -> RealDataset:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y disagree on n: {X.shape[0]} vs {Y.shape[0]}")
    if n_train < 1 or n_test < 1:
        raise ValueError(f"n_train and n_test must both be positive; got {n_train}, {n_test}")
    n_total = n_train + n_test
    if n_total > X.shape[0]:
        raise ValueError(f"dataset {name!r} has {X.shape[0]} rows, but {n_total} were requested")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"dataset {name!r} contains non-finite values")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])[:n_total]
    X = np.ascontiguousarray(X[idx], dtype=np.float64)
    Y = np.ascontiguousarray(Y[idx], dtype=np.float64)
    return RealDataset(
        name=name,
        X_train=X[:n_train],
        Y_train=Y[:n_train],
        X_test=X[n_train:],
        Y_test=Y[n_train:],
    )


def _require_sklearn():
    try:
        from sklearn.datasets import (
            fetch_california_housing,
            fetch_openml,
            load_diabetes,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "real-data benchmarks require scikit-learn; install it in the pixi environment first"
        ) from exc
    return fetch_california_housing, fetch_openml, load_diabetes


def _diabetes(n_train: int, n_test: int, seed: int) -> RealDataset:
    _, _, load_diabetes = _require_sklearn()
    data = load_diabetes()
    return split_arrays("diabetes", data.data, data.target, n_train=n_train, n_test=n_test, seed=seed)


def _california_housing(n_train: int, n_test: int, seed: int) -> RealDataset:
    fetch_california_housing, _, _ = _require_sklearn()
    data = fetch_california_housing()
    return split_arrays(
        "california_housing",
        data.data,
        data.target,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )


def _kin8nm(n_train: int, n_test: int, seed: int) -> RealDataset:
    _, fetch_openml, _ = _require_sklearn()
    data = fetch_openml(name="kin8nm", version=1, as_frame=True, parser="auto")
    X = data.data.to_numpy(dtype=np.float64)
    Y = data.target.to_numpy(dtype=np.float64)
    return split_arrays("kin8nm", X, Y, n_train=n_train, n_test=n_test, seed=seed)


def _wine_quality_white(n_train: int, n_test: int, seed: int) -> RealDataset:
    _, fetch_openml, _ = _require_sklearn()
    data = fetch_openml(name="wine-quality-white", version=1, as_frame=True, parser="auto")
    X = data.data.to_numpy(dtype=np.float64)
    Y = data.target.astype(int).to_numpy(dtype=np.float64)
    return split_arrays("wine_quality_white", X, Y, n_train=n_train, n_test=n_test, seed=seed)


REAL_DATASETS: dict[str, RealDatasetLoader] = {
    "diabetes": _diabetes,
    "california_housing": _california_housing,
    "kin8nm": _kin8nm,
    "wine_quality_white": _wine_quality_white,
}
