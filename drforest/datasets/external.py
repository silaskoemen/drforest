"""Offline read-side access and on-disk persistence for external MTR benchmarks.

This module is strictly offline: it owns **no** URLs, checksums, downloads, or
ARFF parsing. Datasets are acquired out of band by the run-once script
``benchmarks/data/fetch_mtr.py`` — the sole owner of the source registry, the
network fetch, and checksum verification — which writes the raw payload under
``data/raw`` and a parsed ``.npz`` under ``data/processed`` (via the persistence
helpers here). Experiments read only the processed payload through
:func:`read_mtr_dataset` / :func:`drforest.datasets.load_dataset`, so they are
reproducible and network-free by construction.

Data lives under ``$DRFOREST_DATA_DIR`` when set, else ``<repo>/data``.
"""

import os
from pathlib import Path

import numpy as np

# External datasets that have a fetch recipe in benchmarks/data/fetch_mtr.py.
# That script asserts its source registry matches this set, so drift fails loudly.
EXTERNAL_MTR_NAMES = frozenset({"enb"})


def is_fetchable(name: str) -> bool:
    """Whether an external dataset has a wired fetch recipe."""
    return name in EXTERNAL_MTR_NAMES


def data_root() -> Path:
    """Project data directory: ``$DRFOREST_DATA_DIR`` or ``<repo>/data``."""
    env = os.environ.get("DRFOREST_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


def raw_dir() -> Path:
    return data_root() / "raw"


def processed_dir() -> Path:
    return data_root() / "processed"


def raw_path(name: str) -> Path:
    return raw_dir() / f"{name}.arff"


def processed_path(name: str) -> Path:
    return processed_dir() / f"{name}.npz"


def write_processed(
    name: str, X: np.ndarray, Y: np.ndarray, feature_names: tuple[str, ...], response_names: tuple[str, ...]
) -> Path:
    """Persist a parsed dataset as ``data/processed/{name}.npz`` and return the path.

    The write half of the npz schema; :func:`read_mtr_dataset` is its inverse, so
    keeping both here pins one schema for writer and reader.
    """
    path = processed_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=X,
        Y=Y,
        feature_names=np.asarray(feature_names),
        response_names=np.asarray(response_names),
    )
    return path


def read_mtr_dataset(name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Read a processed external dataset as ``(X, Y, feature_names, response_names)``.

    Reads only from ``data/processed`` — no network, no on-the-fly parsing. Raises
    :class:`FileNotFoundError` with the materialisation command if it is absent.
    """
    if name not in EXTERNAL_MTR_NAMES:
        raise NotImplementedError(f"no fetch recipe is wired for external dataset {name!r}")
    path = processed_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"processed dataset {name!r} not found at {path}. "
            f"Materialise it with:  pixi run python benchmarks/data/fetch_mtr.py {name}"
        )
    with np.load(path, allow_pickle=False) as npz:
        X = np.ascontiguousarray(npz["X"], dtype=np.float64)
        Y = np.ascontiguousarray(npz["Y"], dtype=np.float64)
        feature_names = tuple(str(s) for s in npz["feature_names"])
        response_names = tuple(str(s) for s in npz["response_names"])
    return X, Y, feature_names, response_names
