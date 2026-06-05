"""Symbolic row-mixture of weight matrices on shared training atoms.

``shrink`` produces weights of the form ``(1-α)·base + α·target`` per test row.
For the marginal target ``target`` is the uniform ``1/n_train`` distribution, so
the materialized matrix is fully dense even though ``base`` is sparse. This object
keeps the mixture *symbolic*: linear targets (mean, CDF) act through :meth:`apply`
without ever forming the dense matrix, and only the explicit :meth:`to_csr`
escape hatch materializes it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class MixtureWeights:
    """Row-stochastic ``(1-α)·base + α·target`` without materializing it.

    ``base`` and ``target`` are already row-normalized. ``target`` has shape
    ``(1, n_train)`` (broadcast to every test row) or ``(n_test, n_train)``.
    """

    base: csr_matrix  # (n_test, n_train), row-normalized
    alpha: np.ndarray  # (n_test,) float64 in [0, 1]
    target: csr_matrix  # (1, n_train) or (n_test, n_train), row-normalized

    def __post_init__(self) -> None:
        # Coerce to canonical CSR (sorted, deduplicated, no explicit zeros) so the
        # fast paths can rely on base.indptr/indices and so is_uniform_marginal
        # cannot be fooled by duplicate indices summing to a uniform value.
        for name in ("base", "target"):
            W = csr_matrix(getattr(self, name), dtype=np.float64, copy=True)
            W.sum_duplicates()
            W.eliminate_zeros()
            object.__setattr__(self, name, W)
        # Own a private copy so a caller mutating alpha cannot bypass the invariant.
        object.__setattr__(self, "alpha", np.asarray(self.alpha, dtype=np.float64).copy())

        if self.base.ndim != 2:
            raise ValueError(f"base must be 2-D; got shape {self.base.shape}")
        n_test, n_train = self.base.shape
        if n_test < 1 or n_train < 1:
            raise ValueError(f"base must be non-empty; got shape {self.base.shape}")

        if self.alpha.shape != (n_test,):
            raise ValueError(f"alpha must have shape {(n_test,)}; got {self.alpha.shape}")
        if not np.isfinite(self.alpha).all():
            raise ValueError("alpha contains non-finite values")
        if ((self.alpha < 0.0) | (self.alpha > 1.0)).any():
            raise ValueError("alpha must lie in [0, 1]")

        if self.target.shape[1] != n_train:
            raise ValueError(f"target has {self.target.shape[1]} columns but base has {n_train}")
        if self.target.shape[0] not in (1, n_test):
            raise ValueError(f"target must have 1 or {n_test} rows; got {self.target.shape[0]}")

        for name, W in (("base", self.base), ("target", self.target)):
            if W.data.size and not np.isfinite(W.data).all():
                raise ValueError(f"{name} contains non-finite weights")
            if W.data.size and (W.data < 0.0).any():
                raise ValueError(f"{name} contains negative weights")
            row_sums = np.asarray(W.sum(axis=1)).ravel()
            if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(f"{name} rows must sum to 1")

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    @property
    def n_test(self) -> int:
        return self.base.shape[0]

    @property
    def n_train(self) -> int:
        return self.base.shape[1]

    @property
    def is_uniform_marginal(self) -> bool:
        """True iff ``target`` is the single-row uniform ``1/n_train`` marginal.

        Relies on the canonical CSR enforced in ``__post_init__``: full column
        coverage means ``indices == arange(n_train)`` in one deduplicated row.
        """
        t = self.target
        n = self.n_train
        return (
            t.shape[0] == 1
            and t.nnz == n
            and np.array_equal(t.indptr, np.array([0, n]))
            and np.array_equal(t.indices, np.arange(n))
            and bool(np.allclose(t.data, 1.0 / n, rtol=0.0, atol=1e-12))
        )

    def apply(self, M: np.ndarray) -> np.ndarray:
        """Linear action ``((1-α)·base + α·target) @ M``, without densifying."""
        if M.ndim != 2:
            raise ValueError(f"M must be 2-D; got shape {M.shape}")
        if M.shape[0] != self.n_train:
            raise ValueError(f"M has {M.shape[0]} rows but n_train is {self.n_train}")
        base_part = self.base @ M
        target_part = self.target @ M  # (1, c) broadcasts, or (n_test, c)
        return (1.0 - self.alpha)[:, None] * base_part + self.alpha[:, None] * target_part

    def to_csr(self) -> csr_matrix:
        """Materialize the full row-stochastic matrix (the dense escape hatch)."""
        out = self.base.multiply((1.0 - self.alpha)[:, None])
        out = out + self.target.multiply(self.alpha[:, None])
        return out.tocsr()
