"""Thin optional wrappers around the Rust split-search extension."""

from collections.abc import Sequence

import numpy as np

try:
    from drforest import _drforest_core
except ModuleNotFoundError:  # pragma: no cover - exercised only in unbuilt source trees
    _drforest_core = None


def rust_available() -> bool:
    return _drforest_core is not None


def best_complex_embedding_split(
    X: np.ndarray,
    psi: np.ndarray,
    features: Sequence[int],
    *,
    scale: float,
    min_leaf: int,
    threshold_bounds: np.ndarray | None,
    max_cutpoints: int | None,
) -> tuple[int, float, float] | None:
    if _drforest_core is None:
        raise RuntimeError("Rust extension _drforest_core is not available")
    X = np.ascontiguousarray(X, dtype=np.float64)
    psi = np.ascontiguousarray(psi)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if psi.ndim != 2:
        raise ValueError(f"psi must be 2-D; got shape {psi.shape}")
    if X.shape[0] != psi.shape[0]:
        raise ValueError(f"X and psi disagree on n: {X.shape[0]} vs {psi.shape[0]}")

    psi_re = np.ascontiguousarray(np.real(psi), dtype=np.float64)
    psi_im = np.ascontiguousarray(np.imag(psi), dtype=np.float64)
    feature_ids = [int(feature) for feature in features]
    bounds = None
    if threshold_bounds is not None:
        bounds_arr = np.ascontiguousarray(threshold_bounds, dtype=np.float64)
        if bounds_arr.shape != (len(feature_ids), 2):
            raise ValueError(f"threshold_bounds must have shape ({len(feature_ids)}, 2); got {bounds_arr.shape}")
        bounds = bounds_arr

    return _drforest_core.best_complex_embedding_split(
        X,
        psi_re,
        psi_im,
        feature_ids,
        float(scale),
        int(min_leaf),
        bounds,
        max_cutpoints,
    )


def best_cart_split(
    X: np.ndarray,
    Y: np.ndarray,
    features: Sequence[int],
    *,
    min_leaf: int,
    threshold_bounds: np.ndarray | None,
    max_cutpoints: int | None,
) -> tuple[int, float, float] | None:
    if _drforest_core is None:
        raise RuntimeError("Rust extension _drforest_core is not available")
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y disagree on n: {X.shape[0]} vs {Y.shape[0]}")

    feature_ids = [int(feature) for feature in features]
    bounds = None
    if threshold_bounds is not None:
        bounds_arr = np.ascontiguousarray(threshold_bounds, dtype=np.float64)
        if bounds_arr.shape != (len(feature_ids), 2):
            raise ValueError(f"threshold_bounds must have shape ({len(feature_ids)}, 2); got {bounds_arr.shape}")
        bounds = bounds_arr

    return _drforest_core.best_cart_split(
        X,
        Y,
        feature_ids,
        int(min_leaf),
        bounds,
        max_cutpoints,
    )
