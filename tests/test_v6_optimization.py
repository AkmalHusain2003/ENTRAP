import numpy as np
import pytest

from entrap import topology
from entrap.config import EntrapConfig
from entrap.graph import build_knn_graph, edges_to_csr
from entrap.significance import (
    poisson_null_tau, mode_count_stability, prominence_null,
    graph_distance_blocks,
)
from entrap.graph import log_density
from entrap.filtration import build_filtration
from entrap.stability import compute_pi
from entrap.parallel import resolve_workers


def _reference_csr(n, edges):
    """Convert the test graph to the reference CSR representation."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.empty(0, np.int64), np.zeros(n + 1, np.int64)
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    degree = np.bincount(src, minlength=n)
    ao = np.empty(n + 1, np.int64)
    ao[0] = 0
    np.cumsum(degree, out=ao[1:])
    return dst, ao


def test_v6_csr_builder_matches_reference_and_rows_are_sorted():
    """Verify v6 csr builder matches reference and rows are sorted."""
    rng = np.random.default_rng(44)
    for n in (3, 19, 100):
        edges = np.column_stack((rng.integers(0, n, 8 * n),
                                 rng.integers(0, n, 8 * n)))
        edges = edges[edges[:, 0] != edges[:, 1]]
        ai, ao = edges_to_csr(n, edges)
        ai_ref, ao_ref = _reference_csr(n, edges)
        np.testing.assert_array_equal(ai, ai_ref)
        np.testing.assert_array_equal(ao, ao_ref)
        for i in range(n):
            row = ai[ao[i]:ao[i + 1]]
            assert np.all(row[:-1] <= row[1:])


def test_v6_csr_graph_preserves_public_adjacency_api_and_can_drop_knn_cache():
    """Verify v6 csr graph preserves public adjacency api and can drop knn cache."""
    X = np.random.default_rng(3).normal(size=(120, 4))
    g = build_knn_graph(X, 12, store_knn=False)
    assert g.knn_idx is None and g.knn_dist is None
    assert len(g.adj) == len(X)
    for a, b in g.edges:
        assert int(b) in g.adj[int(a)]
        assert int(a) in g.adj[int(b)]
    assert g.weights.shape == (len(g.edges),)
    assert not g.weights.flags.writeable


def test_v6_cython_edge_fields_match_vectorized_reference():
    """Verify v6 cython edge fields match vectorized reference."""
    rng = np.random.default_rng(8)
    X = rng.normal(size=(90, 5))
    g = build_knn_graph(X, 10)
    ell = np.log(1.0 / np.maximum(g.knn_dist.mean(axis=1), 1e-12))
    resid, elev = topology.edge_fields(X, g.edges, ell)
    dist = np.linalg.norm(X[g.edges[:, 0]] - X[g.edges[:, 1]], axis=1)
    a, b = g.edges[:, 0], g.edges[:, 1]
    ref_resid = -np.log(np.maximum(dist, 1e-300)) - 0.5 * (ell[a] + ell[b])
    lo, hi = np.minimum(ell[a], ell[b]), np.maximum(ell[a], ell[b])
    ref_elev = lo + np.minimum(0.0, 0.5 * (hi - lo) + ref_resid)
    np.testing.assert_allclose(resid, ref_resid, rtol=0, atol=1e-12)
    np.testing.assert_allclose(elev, ref_elev, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mreach", [False, True])
def test_poisson_null_parallel_matches_serial_exactly(mreach):
    """Verify poisson null parallel matches serial exactly."""
    X = np.random.default_rng(12).normal(size=(100, 3))
    serial = poisson_null_tau(X, 9, 11, 0.1, np.random.default_rng(6),
                              mreach=mreach, n_jobs=1)[1]
    parallel = poisson_null_tau(X, 9, 11, 0.1, np.random.default_rng(6),
                                mreach=mreach, n_jobs=2)[1]
    np.testing.assert_array_equal(serial, parallel)




@pytest.mark.parametrize("mreach", [False, True])
def test_block_permutation_parallel_matches_serial_exactly(mreach):
    """Verify block permutation parallel matches serial exactly."""
    X = np.random.default_rng(22).normal(size=(95, 3))
    graph = build_knn_graph(X, 8)
    density = log_density(graph.density)
    blocks = graph_distance_blocks(graph.adj_csr, radius=2)
    edges = resid = None
    if mreach:
        filtration, _ = build_filtration(X, graph.edges, 8, mreach=True)
        edges, resid = filtration.edges, filtration.resid
    serial = prominence_null(
        density, graph.adj_csr, blocks, 11, 0.1, np.random.default_rng(42),
        edges=edges, resid=resid, n_jobs=1)
    parallel = prominence_null(
        density, graph.adj_csr, blocks, 11, 0.1, np.random.default_rng(42),
        edges=edges, resid=resid, n_jobs=2)
    assert serial == parallel


@pytest.mark.parametrize("mreach", [False, True])
def test_stability_parallel_matches_serial_exactly(mreach):
    """Verify stability parallel matches serial exactly."""
    X = np.random.default_rng(16).normal(size=(110, 3))
    g = build_knn_graph(X, 9)
    serial = mode_count_stability(
        X, g.edges, 0.2, 9, 0.8, 0.1, 9, np.random.default_rng(27),
        mreach=mreach, n_jobs=1)
    parallel = mode_count_stability(
        X, g.edges, 0.2, 9, 0.8, 0.1, 9, np.random.default_rng(27),
        mreach=mreach, n_jobs=2)
    assert serial[0] == parallel[0]
    np.testing.assert_array_equal(serial[1], parallel[1])


def test_pi_run_parallelism_is_worker_count_invariant():
    """Verify pi run parallelism is worker count invariant."""
    X = np.vstack([
        np.random.default_rng(1).normal((-3, 0, 0), 0.35, size=(24, 3)),
        np.random.default_rng(2).normal((3, 0, 0), 0.35, size=(24, 3)),
    ])
    reference_labels = np.repeat([0, 1], 24)
    reference_basins = {0: np.arange(24), 1: np.arange(24, 48)}

    def run(n_jobs):
        """Run the patched operation and record its call arguments."""
        cfg = EntrapConfig.from_params(5, 0.2, {
            "B": 2, "P": 5, "pi_runs": 3, "stability_B": 2,
            "pi_full_stability": False, "use_leiden": False,
            "n_jobs": n_jobs, "parallel_max_workers": 2,
        }).resolve(len(X))
        graph = build_knn_graph(X, cfg.k_resolved, store_knn=False)
        return compute_pi(graph, X, reference_labels, reference_basins,
                          cfg, None, np.random.default_rng(31))

    serial = run(1)
    parallel = run(2)
    np.testing.assert_array_equal(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])


def test_parallel_config_validation():
    """Verify parallel config validation."""
    with pytest.raises(ValueError):
        EntrapConfig.from_params(10, 0.05, {"n_jobs": 0})
    with pytest.raises(ValueError):
        EntrapConfig.from_params(10, 0.05, {"parallel_memory_mb": 0})


def test_worker_pool_obeys_task_cpu_and_memory_caps():
    """Verify worker pool obeys task cpu and memory caps."""
    assert resolve_workers(-1, 20, parallel_max_workers=3,
                           parallel_memory_mb=8, bytes_per_worker=1) == 3
    assert resolve_workers(12, 20, parallel_max_workers=8,
                           parallel_memory_mb=1, bytes_per_worker=600_000) == 1
    assert resolve_workers(12, 2, parallel_max_workers=8,
                           parallel_memory_mb=8, bytes_per_worker=1) == 2
