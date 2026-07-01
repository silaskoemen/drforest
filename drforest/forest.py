"""Distributional random forests with estimator-style prediction methods.

The fitted forest represents a conditional distribution as sparse weights over
the training responses. ``predict`` and the distributional prediction methods
are convenient plug-ins on those weights; ``predict_weights`` exposes the
underlying representation for research and custom targets.
"""

import math
import secrets
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix

from drforest.criteria import (
    AdaptiveMmdCriterion,
    AnisotropicMmdCriterion,
    CartCriterion,
    MmdRffCriterion,
    SlicedWassersteinCriterion,
)
from drforest.criteria.base import Criterion
from drforest.features.rff import coordinatewise_median_heuristic, median_heuristic
from drforest.rng import RngStreams
from drforest.targets import weighted_cdf, weighted_mean, weighted_quantile
from drforest.tree import DecisionTree, TreeParams, build_tree, fold_sizes
from drforest.weights import assemble_weights

CriterionFactory = Callable[[np.ndarray], Criterion]

_DEFAULT_CRITERION = "mmd"
_N_RANDOM_FEATURES = 128
_ADAPTIVE_SELECTED_FEATURES = 32
_CRITERION_NAMES = frozenset(
    {
        "adaptive_mmd",
        "anisotropic_mmd",
        "cart",
        "mmd",
        "mmd_rff",
        "sliced_wasserstein",
    }
)


def _criterion_from_name(name: str, Y: np.ndarray) -> Criterion:
    if name == "cart":
        return CartCriterion()
    if name in {"mmd", "mmd_rff"}:
        return MmdRffCriterion.from_data(
            Y,
            n_features=_N_RANDOM_FEATURES,
            bandwidth_rule=median_heuristic,
        )
    if name == "anisotropic_mmd":
        return AnisotropicMmdCriterion.from_data(
            Y,
            n_features=_N_RANDOM_FEATURES,
            bandwidth_rule=coordinatewise_median_heuristic,
        )
    if name == "sliced_wasserstein":
        return SlicedWassersteinCriterion.from_data(Y, n_projections=_N_RANDOM_FEATURES)
    if name == "adaptive_mmd":
        return AdaptiveMmdCriterion.from_data(
            Y,
            pool_features=_N_RANDOM_FEATURES,
            selected_features=_ADAPTIVE_SELECTED_FEATURES,
            bandwidth_rule=median_heuristic,
        )
    choices = ", ".join(sorted(_CRITERION_NAMES))
    raise ValueError(f"unknown criterion {name!r}; expected one of: {choices}")


class DistributionalRandomForest:
    """Estimate conditional distributions with an honest random forest.

    ``criterion`` selects a built-in criterion and takes precedence over
    ``criterion_factory`` when both are provided. If neither is supplied, the
    Gaussian MMD criterion is used. The factory interface is intended for
    research criteria and data-dependent criterion configuration.
    """

    def __init__(
        self,
        *,
        criterion: str | None = None,
        criterion_factory: CriterionFactory | None = None,
        n_estimators: int = 100,
        subsample: float = 0.5,
        bootstrap: bool = False,
        min_samples_leaf: int = 5,
        alpha: float = 0.05,
        honesty_fraction: float = 0.5,
        colsample: float = 0.7,
        max_cutpoints: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1; got {n_estimators}")
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1]; got {subsample}")
        self.criterion = criterion
        self.criterion_factory = criterion_factory
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.bootstrap = bool(bootstrap)
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.honesty_fraction = honesty_fraction
        self.colsample = colsample
        self.max_cutpoints = max_cutpoints
        self.random_state = random_state
        self._trees: list[DecisionTree] | None = None
        self._n_train: int | None = None
        self._y_was_1d = False

    @property
    def is_fitted(self) -> bool:
        return self._trees is not None

    @property
    def tree_params(self) -> TreeParams:
        """Validated low-level tree parameters for this forest."""
        return TreeParams(
            min_samples_leaf=self.min_samples_leaf,
            alpha=self.alpha,
            honesty_fraction=self.honesty_fraction,
            colsample=self.colsample,
            max_cutpoints=self.max_cutpoints,
        )

    @property
    def _subsample_is_sublinear(self) -> bool:
        return False

    @property
    def inference_valid(self) -> bool:
        """Whether this configuration is in the theorem's inference-valid regime."""
        return self.honesty_fraction > 0.0 and not self.bootstrap and self.alpha > 0.0 and self._subsample_is_sublinear

    @property
    def trees(self) -> list[DecisionTree]:
        """Fitted trees; prefer ``estimators_`` in estimator-style code."""
        if self._trees is None:
            raise RuntimeError("forest is not fitted; call fit() first")
        return self._trees

    def subsample_size(self, n: int) -> int:
        """Return the per-tree sample size for ``n`` training rows."""
        return min(max(math.ceil(self.subsample * n), 1), n)

    def _make_criterion(self, Y: np.ndarray) -> Criterion:
        if self.criterion is not None:
            if not isinstance(self.criterion, str):
                raise TypeError(f"criterion must be a string or None; got {type(self.criterion).__name__}")
            return _criterion_from_name(self.criterion, Y)
        if self.criterion_factory is not None:
            configured = self.criterion_factory(Y)
            if not isinstance(configured, Criterion):
                raise TypeError("criterion_factory must return a Criterion; " f"got {type(configured).__name__}")
            return configured
        return _criterion_from_name(_DEFAULT_CRITERION, Y)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DistributionalRandomForest":
        """Fit the forest on predictors ``X`` and scalar or multivariate ``y``."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        y_was_1d = y_array.ndim == 1
        if y_was_1d:
            y_array = y_array[:, None]
        Y = np.ascontiguousarray(y_array, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}")
        if Y.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D; got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and y disagree on n: {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[1] == 0:
            raise ValueError("X has zero features (p == 0)")
        if Y.shape[1] == 0:
            raise ValueError("y has zero response dimensions")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values; missing-value handling is not implemented")
        if not np.isfinite(Y).all():
            raise ValueError("y contains non-finite values; missing-value handling is not implemented")

        params = self.tree_params
        n = X.shape[0]
        subsample_size = self.subsample_size(n)
        n_s, n_l = fold_sizes(subsample_size, params.honesty_fraction)
        if min(n_s, n_l) < params.min_samples_leaf:
            raise ValueError(
                f"subsample too small: n={n}, subsample={self.subsample} -> "
                f"s_n={subsample_size}; honest folds (S={n_s}, L={n_l}) fall below "
                f"min_samples_leaf={params.min_samples_leaf}. Increase subsample or "
                "n, or lower min_samples_leaf/honesty_fraction."
            )

        criterion = self._make_criterion(Y)
        seed = secrets.randbits(128) if self.random_state is None else int(self.random_state)
        streams = RngStreams(seed)
        trees = [
            build_tree(
                X,
                Y,
                criterion,
                params,
                subsample_size=subsample_size,
                bootstrap=self.bootstrap,
                tree_rng=streams.tree(k),
                node_rng=self._node_rng(streams, k),
            )
            for k in range(self.n_estimators)
        ]

        self._trees = trees
        self._n_train = n
        self._y_was_1d = y_was_1d
        self.estimators_ = trees
        self.criterion_ = criterion
        self.y_train_ = Y.copy()
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = Y.shape[1]
        return self

    def predict_weights(self, X: ArrayLike) -> csr_matrix:
        """Return sparse conditional weights with shape ``(n_test, n_train)``."""
        if self._trees is None or self._n_train is None:
            raise RuntimeError("forest is not fitted; call fit() first")
        X = np.ascontiguousarray(X, dtype=np.float64)
        return assemble_weights(self._trees, X, self._n_train)

    def weights(self, X: ArrayLike) -> csr_matrix:
        """Return conditional weights; alias for :meth:`predict_weights`."""
        return self.predict_weights(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the conditional mean."""
        self._require_targets()
        prediction = weighted_mean(self.predict_weights(X), self.y_train_)
        return prediction[:, 0] if self._y_was_1d else prediction

    def predict_quantiles(self, X: ArrayLike, quantiles: ArrayLike) -> np.ndarray:
        """Predict conditional quantiles.

        Univariate results have shape ``(n_test, n_quantiles)``; multivariate
        results have shape ``(n_test, n_outputs, n_quantiles)``.
        """
        self._require_targets()
        prediction = weighted_quantile(
            self.predict_weights(X),
            self.y_train_,
            np.asarray(quantiles, dtype=np.float64),
        )
        return prediction[:, 0, :] if self._y_was_1d else prediction

    def predict_cdf(self, X: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
        """Evaluate conditional marginal CDFs at ``thresholds``."""
        self._require_targets()
        prediction = weighted_cdf(
            self.predict_weights(X),
            self.y_train_,
            np.asarray(thresholds, dtype=np.float64),
        )
        return prediction[:, 0, :] if self._y_was_1d else prediction

    def _require_targets(self) -> None:
        if self._trees is None or not hasattr(self, "y_train_"):
            raise RuntimeError("forest is not fitted; call fit() first")

    @staticmethod
    def _node_rng(streams: RngStreams, tree_id: int) -> Callable[[int], np.random.Generator]:
        return lambda node_id: streams.node(tree_id, node_id)
