"""Small offline datasets for DRF experiments.

The synthetic loaders mirror the simulation mechanisms used in the DRF paper
where practical. External benchmark names are registered here as placeholders so
experiment code can fail loudly until fetch/preprocessing support is added.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

DatasetKind = Literal["synthetic", "external"]


@dataclass(frozen=True)
class Dataset:
    """In-memory dataset with predictor matrix ``X`` and response matrix ``Y``."""

    name: str
    X: np.ndarray
    Y: np.ndarray
    feature_names: tuple[str, ...]
    response_names: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DatasetInfo:
    """Registry metadata for available and planned datasets."""

    name: str
    kind: DatasetKind
    n_features: int | None
    n_responses: int | None
    description: str
    source: str


PAPER_SOURCE = "Cevid et al. 2022, Distributional Random Forests, Appendix C"
MULAN_SOURCE = "Tsoumakas et al. 2011 / Mulan multiple-target regression collection"

_SYNTHETIC_INFOS = {
    "paper_quantile_1": DatasetInfo(
        name="paper_quantile_1",
        kind="synthetic",
        n_features=None,
        n_responses=1,
        description="Paper scenario 1: univariate conditional mean shift based on X1.",
        source=PAPER_SOURCE,
    ),
    "paper_quantile_2": DatasetInfo(
        name="paper_quantile_2",
        kind="synthetic",
        n_features=None,
        n_responses=1,
        description="Paper scenario 2: univariate conditional variance shift based on X1.",
        source=PAPER_SOURCE,
    ),
    "paper_quantile_3": DatasetInfo(
        name="paper_quantile_3",
        kind="synthetic",
        n_features=None,
        n_responses=1,
        description="Paper scenario 3: same first two moments, different conditional shape.",
        source=PAPER_SOURCE,
    ),
    "paper_copula": DatasetInfo(
        name="paper_copula",
        kind="synthetic",
        n_features=None,
        n_responses=None,
        description="Gaussian-copula example with marginal N(0, 1) responses and correlation depending on X1.",
        source=PAPER_SOURCE,
    ),
    "paper_heterogeneous_regression": DatasetInfo(
        name="paper_heterogeneous_regression",
        kind="synthetic",
        n_features=None,
        n_responses=2,
        description="Causal/heterogeneous regression toy with response columns (treatment, outcome).",
        source=PAPER_SOURCE,
    ),
    "shrinkage_toy": DatasetInfo(
        name="shrinkage_toy",
        kind="synthetic",
        n_features=2,
        n_responses=2,
        description="Compact multivariate distribution-shift toy for raw-vs-shrunk metric smoke tests.",
        source="drforest local synthetic benchmark",
    ),
}

_EXTERNAL_INFOS = {
    name: DatasetInfo(
        name=name,
        kind="external",
        n_features=None,
        n_responses=None,
        description="Registered benchmark dataset; fetch/preprocessing is not implemented yet.",
        source=MULAN_SOURCE,
    )
    for name in ("jura", "slump", "wq", "enb", "atp1d", "atp7d", "scpf", "sf1", "sf2")
}
_EXTERNAL_INFOS.update(
    {
        "birth1": DatasetInfo(
            name="birth1",
            kind="external",
            n_features=None,
            n_responses=2,
            description="CDC natality-derived dataset with pregnancy length and birthweight responses.",
            source=PAPER_SOURCE,
        ),
        "birth2": DatasetInfo(
            name="birth2",
            kind="external",
            n_features=None,
            n_responses=4,
            description="CDC natality-derived health-response dataset.",
            source=PAPER_SOURCE,
        ),
        "wage": DatasetInfo(
            name="wage",
            kind="external",
            n_features=None,
            n_responses=2,
            description="ACS-derived wage/fairness dataset with log hourly wage and gender responses.",
            source=PAPER_SOURCE,
        ),
        "air": DatasetInfo(
            name="air",
            kind="external",
            n_features=None,
            n_responses=6,
            description="EPA air-quality dataset with six pollutant responses.",
            source=PAPER_SOURCE,
        ),
    }
)

DATASET_REGISTRY = MappingProxyType({**_SYNTHETIC_INFOS, **_EXTERNAL_INFOS})


def _validate_size(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}: {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _dataset(
    *,
    name: str,
    X: np.ndarray,
    Y: np.ndarray,
    feature_prefix: str,
    response_names: tuple[str, ...],
    description: str,
) -> Dataset:
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y disagree on n: {X.shape[0]} vs {Y.shape[0]}")
    if len(response_names) != Y.shape[1]:
        raise ValueError(f"expected {Y.shape[1]} response names; got {len(response_names)}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("datasets must contain only finite values")
    return Dataset(
        name=name,
        X=X,
        Y=Y,
        feature_names=tuple(f"{feature_prefix}{j + 1}" for j in range(X.shape[1])),
        response_names=response_names,
        description=description,
    )


def list_datasets(*, kind: DatasetKind | None = None) -> tuple[DatasetInfo, ...]:
    """Return registered dataset metadata, sorted by dataset name."""
    if kind is not None and kind not in ("synthetic", "external"):
        raise ValueError(f"kind must be 'synthetic', 'external', or None; got {kind!r}")
    infos = DATASET_REGISTRY.values()
    if kind is not None:
        infos = [info for info in infos if info.kind == kind]
    return tuple(sorted(infos, key=lambda info: info.name))


def make_quantile_scenario(
    scenario: Literal[1, 2, 3],
    *,
    n: int = 2000,
    p: int = 40,
    seed: int = 0,
) -> Dataset:
    """Generate one of the three univariate quantile scenarios from the paper."""
    n = _validate_size("n", n)
    p = _validate_size("p", p)
    if scenario not in (1, 2, 3):
        raise ValueError(f"scenario must be 1, 2, or 3; got {scenario!r}")

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    right = X[:, 0] > 0.0

    if scenario == 1:
        Y = rng.normal(loc=0.8 * right, scale=1.0, size=n)
    elif scenario == 2:
        Y = rng.normal(loc=0.0, scale=1.0 + right, size=n)
    else:
        Y = np.empty(n, dtype=np.float64)
        Y[~right] = rng.normal(loc=1.0, scale=1.0, size=int(np.sum(~right)))
        Y[right] = rng.exponential(scale=1.0, size=int(np.sum(right)))

    name = f"paper_quantile_{scenario}"
    return _dataset(
        name=name,
        X=X,
        Y=Y[:, None],
        feature_prefix="x",
        response_names=("y",),
        description=DATASET_REGISTRY[name].description,
    )


def make_gaussian_copula(
    *,
    n: int = 5000,
    p: int = 30,
    d: int = 5,
    seed: int = 0,
) -> Dataset:
    """Generate the Gaussian-copula example with ``Cor(Y_i, Y_j | X) = X1``."""
    n = _validate_size("n", n)
    p = _validate_size("p", p)
    d = _validate_size("d", d, minimum=2)

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    rho = X[:, [0]]
    shared = rng.normal(size=(n, 1))
    independent = rng.normal(size=(n, d))
    Y = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * independent
    return _dataset(
        name="paper_copula",
        X=X,
        Y=Y,
        feature_prefix="x",
        response_names=tuple(f"y{j + 1}" for j in range(d)),
        description=DATASET_REGISTRY["paper_copula"].description,
    )


def make_heterogeneous_regression(
    *,
    n: int = 5000,
    p: int = 20,
    seed: int = 0,
) -> Dataset:
    """Generate the heterogeneous regression/causal toy from the paper."""
    n = _validate_size("n", n)
    p = _validate_size("p", p, minimum=2)

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 5.0, size=(n, p))
    treatment = rng.normal(loc=X[:, 1], scale=1.0, size=n)
    outcome_mean = X[:, 1] + X[:, 0] * np.sin(treatment)
    outcome = rng.normal(loc=outcome_mean, scale=1.0, size=n)
    Y = np.column_stack([treatment, outcome])
    return _dataset(
        name="paper_heterogeneous_regression",
        X=X,
        Y=Y,
        feature_prefix="x",
        response_names=("treatment", "outcome"),
        description=DATASET_REGISTRY["paper_heterogeneous_regression"].description,
    )


def make_shrinkage_toy(*, n: int = 260, seed: int = 29) -> Dataset:
    """Generate a compact multivariate toy for shrinkage metric comparisons."""
    n = _validate_size("n", n)

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    location = np.column_stack(
        [
            np.where(X[:, 0] < 0.0, -1.0, 1.0),
            np.where(X[:, 1] < 0.0, 0.75, -0.75),
        ]
    )
    scale = 0.25 + 0.8 * (X[:, 0] >= 0.0)[:, None]
    Y = location + rng.normal(scale=scale, size=(n, 2))
    return _dataset(
        name="shrinkage_toy",
        X=X,
        Y=Y,
        feature_prefix="x",
        response_names=("y1", "y2"),
        description=DATASET_REGISTRY["shrinkage_toy"].description,
    )


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Load a registered dataset.

    Synthetic datasets are generated locally. External benchmark entries are
    registered for planning but intentionally fail until a fetcher is added.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(sorted(DATASET_REGISTRY))}")
    info = DATASET_REGISTRY[name]
    if info.kind == "external":
        raise NotImplementedError(
            f"{name!r} is registered but external dataset fetching/preprocessing is not implemented"
        )

    if name == "paper_quantile_1":
        return make_quantile_scenario(1, **kwargs)
    if name == "paper_quantile_2":
        return make_quantile_scenario(2, **kwargs)
    if name == "paper_quantile_3":
        return make_quantile_scenario(3, **kwargs)
    if name == "paper_copula":
        return make_gaussian_copula(**kwargs)
    if name == "paper_heterogeneous_regression":
        return make_heterogeneous_regression(**kwargs)
    if name == "shrinkage_toy":
        return make_shrinkage_toy(**kwargs)
    raise NotImplementedError(f"no loader is wired for synthetic dataset {name!r}")
