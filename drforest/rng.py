"""Explicit, hierarchical RNG streams.

Reproducibility is first-class in drforest. Per-tree and per-node random
streams are *addressable* (indexed by id) and *independent* (no sequential
coupling), because per-node resampling of RFF frequencies / random
projections is a core, intentional decorrelation source — a node must be able
to draw its own deterministic ``Generator`` from the forest seed alone.

Addressing is done with ``SeedSequence`` spawn keys, so ``tree(k)`` and
``node(k, j)`` are pure functions of the root seed: reproducible across runs,
machines, and call order.
"""

import numpy as np
from numpy.random import Generator, SeedSequence


class RngStreams:
    """Deterministic, index-addressable RNG streams derived from one seed."""

    def __init__(self, seed: int) -> None:
        # No default: a missing seed is a wiring bug, not something to paper over.
        self._entropy = int(seed)

    @property
    def seed(self) -> int:
        return self._entropy

    def tree(self, tree_id: int) -> Generator:
        """Independent stream for tree ``tree_id`` (e.g. subsampling, honesty)."""
        return self._generator((tree_id,))

    def node(self, tree_id: int, node_id: int) -> Generator:
        """Independent stream for one node (per-node RFF / projection resample)."""
        return self._generator((tree_id, node_id))

    def _generator(self, spawn_key: tuple[int, ...]) -> Generator:
        return np.random.default_rng(SeedSequence(self._entropy, spawn_key=spawn_key))
