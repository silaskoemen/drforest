"""Conditional means as plug-ins on the DRF weight matrix."""

import numpy as np

from drforest.targets._validation import (
    as_response_matrix,
    as_weight_operator,
    weights_apply,
)


def weighted_mean(W: object, Y: np.ndarray) -> np.ndarray:
    """Return ``E[Y | X=x]`` for each row of ``W``.

    ``W`` may be a sparse/dense weight matrix or a :class:`MixtureWeights`.
    Matrices must be nonnegative with positive row sums; rows are normalized
    before multiplication so small numerical row-sum drift does not leak into
    downstream targets.
    """
    op = as_weight_operator(W)
    Y = as_response_matrix(Y, op.shape[1])
    return weights_apply(op, Y)
