import numpy as np

from drforest.rng import RngStreams


def test_streams_are_deterministic_and_addressable():
    a = RngStreams(123)
    b = RngStreams(123)
    # Same seed + same address -> identical draws, independent of call order.
    assert np.array_equal(a.tree(3).normal(size=8), b.tree(3).normal(size=8))
    assert np.array_equal(a.node(3, 5).normal(size=8), b.node(3, 5).normal(size=8))


def test_distinct_seeds_differ():
    a = RngStreams(1).tree(0).normal(size=8)
    b = RngStreams(2).tree(0).normal(size=8)
    assert not np.array_equal(a, b)


def test_streams_are_independent():
    s = RngStreams(7)
    # Different trees, and different nodes within a tree, are decorrelated.
    t0, t1 = s.tree(0).normal(size=5000), s.tree(1).normal(size=5000)
    n00, n01 = s.node(0, 0).normal(size=5000), s.node(0, 1).normal(size=5000)
    assert abs(np.corrcoef(t0, t1)[0, 1]) < 0.05
    assert abs(np.corrcoef(n00, n01)[0, 1]) < 0.05
    # A node stream is not accidentally aliased to a tree stream.
    assert not np.array_equal(s.tree(0).normal(size=5), s.node(0, 0).normal(size=5))
