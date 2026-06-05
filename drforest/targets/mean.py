"""Conditional means as plug-ins on the DRF weight matrix."""

import numpy as np

from drforest.targets._validation import (
    as_csr_weights,
    as_response_matrix,
    normalize_rows,
)


def weighted_mean(W: object, Y: np.ndarray) -> np.ndarray:
    """Return ``E[Y | X=x]`` for each row of ``W``.

    ``W`` may be sparse or dense, but must be nonnegative with positive row sums.
    Rows are normalized before multiplication so small numerical row-sum drift
    does not leak into downstream targets.
    """
    W_csr = as_csr_weights(W)
    Y = as_response_matrix(Y, W_csr.shape[1])
    return normalize_rows(W_csr) @ Y
