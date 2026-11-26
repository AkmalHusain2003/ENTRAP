"""Regression tests for the exact shared-edge multi-K H0 engine."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from entrap import topology


pytestmark = pytest.mark.skipif(
    not topology.USING_CYTHON,
    reason="The exact multi-K MSF engine requires the compiled Cython extension.",
)


def _knn_indices(X, kmax):
    """Return exact self-excluding kNN indices using the repository's KD-tree convention."""
    n = len(X)
    distances, indices = cKDTree(X).query(X, k=min(n, kmax + 1))
    distances = np.asarray(distances).reshape(n, -1)
    indices = np.asarray(indices).reshape(n, -1)
    output = np.empty((n, min(kmax, n - 1)), dtype=np.int64)
    for row in range(n):
        keep = indices[row] != row
        selected = indices[row, keep]
        if len(selected) < output.shape[1]:
            extra_dist, extra_indices = cKDTree(X).query(
                X[row], k=min(n, output.shape[1] + 2)
            )
            extra_indices = np.atleast_1d(extra_indices)
            selected = extra_indices[extra_indices != row]
        output[row] = selected[: output.shape[1]]
    return output


def _assert_parity(X, knn_idx, ell, scales):
    for mreach in (False, True):
        expected = topology.multiscale_graph_statistics_reference(
            X, knn_idx, ell, scales, mreach=mreach
        )
        actual = topology.multiscale_graph_statistics(
            X, knn_idx, ell, scales, mreach=mreach
        )
        np.testing.assert_allclose(
            actual[0], expected[0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_array_equal(actual[1], expected[1])


def test_random_graphs_match_reference():
    """Preserve persistence maxima and component counts on random point clouds."""
    rng = np.random.default_rng(20261126)
    for case in range(100):
        n = int(rng.integers(12, 100))
        dimension = int(rng.integers(2, 7))
        kmax = min(int(rng.integers(5, 16)), n - 1)
        X = rng.normal(size=(n, dimension))
        knn_idx = _knn_indices(X, kmax)
        ell = rng.normal(size=n)
        scales = np.unique(
            np.array([2, max(2, kmax // 2), kmax], dtype=np.int64)
        )
        _assert_parity(X, knn_idx, ell, scales)


def test_duplicate_points_and_zero_edge_distances_match_reference():
    """Preserve edge filtration when multiple samples share coordinates."""
    rng = np.random.default_rng(321)
    X = rng.integers(-3, 4, size=(60, 3)).astype(np.float64)
    knn_idx = _knn_indices(X, 12)
    ell = rng.normal(size=len(X))
    _assert_parity(X, knn_idx, ell, np.array([2, 4, 8, 12], dtype=np.int64))


def test_density_ties_match_reference():
    """Preserve H0 summaries when many vertices have identical filtration values."""
    rng = np.random.default_rng(99)
    X = rng.normal(size=(75, 4))
    knn_idx = _knn_indices(X, 15)
    ell = rng.integers(-3, 4, size=len(X)).astype(np.float64)
    _assert_parity(X, knn_idx, ell, np.array([2, 5, 9, 15], dtype=np.int64))


def test_equal_coordinates_and_equal_density_match_reference():
    """Preserve deterministic summaries for a fully tied geometric configuration."""
    X = np.zeros((24, 2), dtype=np.float64)
    knn_idx = np.tile(np.arange(1, 13, dtype=np.int64), (24, 1))
    knn_idx = (knn_idx + np.arange(24, dtype=np.int64)[:, None]) % 24
    ell = np.ones(24, dtype=np.float64)
    _assert_parity(X, knn_idx, ell, np.array([2, 4, 8, 12], dtype=np.int64))


def test_component_counts_are_monotone_with_graph_scale():
    """Ensure component counts cannot increase when a larger kNN prefix is used."""
    rng = np.random.default_rng(8128)
    X = rng.normal(size=(100, 5))
    knn_idx = _knn_indices(X, 20)
    ell = rng.normal(size=len(X))
    scales = np.array([2, 4, 8, 12, 20], dtype=np.int64)
    for mreach in (False, True):
        _, components = topology.multiscale_graph_statistics(
            X, knn_idx, ell, scales, mreach=mreach
        )
        assert np.all(np.diff(components) <= 0)


def test_unsorted_scales_match_reference():
    """Preserve outputs when candidate graph scales are supplied out of order."""
    rng = np.random.default_rng(778)
    X = rng.normal(size=(50, 2))
    knn_idx = _knn_indices(X, 10)
    ell = rng.normal(size=len(X))
    _assert_parity(X, knn_idx, ell, np.array([10, 2, 7, 4], dtype=np.int64))


def test_invalid_scales_raise_value_error():
    """Reject graph scales outside the cached maximum neighbourhood."""
    X = np.arange(30, dtype=np.float64).reshape(10, 3)
    knn_idx = _knn_indices(X, 5)
    ell = np.linspace(0.0, 1.0, len(X))
    with pytest.raises(ValueError):
        topology.multiscale_graph_statistics(
            X, knn_idx, ell, np.array([6], dtype=np.int64)
        )


def test_dispatch_uses_reference_when_new_native_kernel_is_unavailable(monkeypatch):
    """Preserve the original implementation as the safe fallback path."""
    rng = np.random.default_rng(101)
    X = rng.normal(size=(40, 3))
    knn_idx = _knn_indices(X, 10)
    ell = rng.normal(size=len(X))
    scales = np.array([2, 5, 10], dtype=np.int64)
    expected = topology.multiscale_graph_statistics_reference(
        X, knn_idx, ell, scales, mreach=True
    )
    monkeypatch.setattr(topology, "_multiscale_msf_statistics", None)
    actual = topology.multiscale_graph_statistics(
        X, knn_idx, ell, scales, mreach=True
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected[1])
