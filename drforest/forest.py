"""Distributional random forest: N honest trees -> sparse weight matrix.

The forest owns what is shared across trees: the per-tree row subsample
(``s_n = ceil(subsample * n)`` rows, without replacement by default;
``bootstrap=True`` draws with replacement), the per-tree / per-node RNG
streams, and criterion configuration. A *criterion factory* maps the full
training ``Y`` to a configured ``Criterion`` once, so data-dependent choices
(e.g. the global median-heuristic bandwidth for ``mmd_rff``) are fixed on all
of ``Y`` rather than re-derived per node.

``subsample`` (row fraction, forest-level) and ``colsample`` (column fraction,
per node, on ``TreeParams``) are the two sampling axes. ``bootstrap`` is the
opt-in fast path: it is convenient for prediction but, like fast (non-honest)
mode, is **not** inference-valid (it breaks the honest-fold disjointness the
consistency results assume).

The forest's product is the row-stochastic weight matrix ``W`` from
:meth:`weights`; targets, shrinkage, and metrics are plug-ins on it.
"""

import math
from collections.abc import Callable

import numpy as np
from scipy.sparse import csr_matrix

from drforest.criteria.base import Criterion
from drforest.rng import RngStreams
from drforest.tree import DecisionTree, TreeParams, build_tree, fold_sizes
from drforest.weights import assemble_weights

CriterionFactory = Callable[[np.ndarray], Criterion]


class DistributionalRandomForest:
    """A target-free forest whose estimate is a plug-in on the weight matrix."""

    def __init__(
        self,
        *,
        criterion_factory: CriterionFactory,
        seed: int,
        n_trees: int = 100,
        subsample: float = 0.5,
        bootstrap: bool = False,
        tree_params: TreeParams = TreeParams(),
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1; got {n_trees}")
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1]; got {subsample}")
        self.criterion_factory = criterion_factory
        self.seed = int(seed)
        self.n_trees = int(n_trees)
        self.subsample = float(subsample)
        self.bootstrap = bool(bootstrap)
        self.tree_params = tree_params
        self._trees: list[DecisionTree] | None = None
        self._n_train: int | None = None
        self._n_features: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._trees is not None

    @property
    def _subsample_is_sublinear(self) -> bool:
        # Fixed-fraction subsampling is not sublinear (s_n / n does not -> 0), so
        # it fails the consistency theorem's subsample condition. This flips to
        # True once a sublinear schedule (s_n ~ n^beta) is offered (deferred).
        return False

    @property
    def inference_valid(self) -> bool:
        """Whether this configuration is in the inference-valid (theorem) regime.

        Requires honest folds, no bootstrap, active alpha-regularity, and a
        sublinear subsample schedule (spec A.4). With fixed-fraction subsampling
        the last condition is unmet, so this is currently always ``False`` —
        i.e. fast / prediction mode. Record it alongside results so honest and
        fast runs are never silently mixed.
        """
        return (
            self.tree_params.honesty_fraction > 0.0
            and not self.bootstrap
            and self.tree_params.alpha > 0.0
            and self._subsample_is_sublinear
        )

    @property
    def trees(self) -> list[DecisionTree]:
        if self._trees is None:
            raise RuntimeError("forest is not fitted; call fit() first")
        return self._trees

    def subsample_size(self, n: int) -> int:
        """s_n = ceil(subsample * n), the per-tree without-replacement size."""
        return min(max(math.ceil(self.subsample * n), 1), n)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DistributionalRandomForest":
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        # Full data validation up front, before any criterion configuration.
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, d); got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y disagree on n: {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[1] == 0:
            raise ValueError("X has zero features (p == 0)")
        if Y.shape[1] == 0:
            raise ValueError("Y has zero response dimensions (d == 0)")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values; missing-value handling is not implemented")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values; missing-value handling is not implemented")

        n = X.shape[0]
        subsample_size = self.subsample_size(n)
        n_s, n_l = fold_sizes(subsample_size, self.tree_params.honesty_fraction)
        if min(n_s, n_l) < self.tree_params.min_samples_leaf:
            raise ValueError(
                f"subsample too small: n={n}, subsample={self.subsample} -> "
                f"s_n={subsample_size}; honest folds (S={n_s}, L={n_l}) fall below "
                f"min_samples_leaf={self.tree_params.min_samples_leaf}. Increase "
                "subsample or n, or lower min_samples_leaf/honesty_fraction."
            )

        # One configured criterion is shared across all trees. This is deliberate:
        # the factory fixes data-dependent config (e.g. median-heuristic sigma, an
        # O(n^2) pdist) once on the full Y, and the Criterion contract requires
        # implementations to be stateless/reentrant (randomness only from the
        # passed rng). Per-tree instances would repeat that config for no gain.
        criterion = self.criterion_factory(Y)
        streams = RngStreams(self.seed)

        trees = [
            build_tree(
                X,
                Y,
                criterion,
                self.tree_params,
                subsample_size=subsample_size,
                bootstrap=self.bootstrap,
                tree_rng=streams.tree(k),
                node_rng=self._node_rng(streams, k),
            )
            for k in range(self.n_trees)
        ]

        self._trees = trees
        self._n_train = n
        self._n_features = X.shape[1]
        return self

    def weights(self, X_test: np.ndarray) -> csr_matrix:
        """Row-stochastic ``W`` (n_test × n_train); rows sum to exactly 1."""
        if self._trees is None or self._n_train is None:
            raise RuntimeError("forest is not fitted; call fit() first")
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        return assemble_weights(self._trees, X_test, self._n_train)

    @staticmethod
    def _node_rng(streams: RngStreams, tree_id: int) -> Callable[[int], np.random.Generator]:
        return lambda node_id: streams.node(tree_id, node_id)
