import numpy as np

from entrap import topology
from entrap.graph import _query_knn_euclidean, adaptive_knn_density_from_distances


def _reference_multiscale(X, knn_idx, ell, scales, mreach=True):
    """Evaluate multi-scale graph statistics with the reference kernel path."""
    maxima = []
    components = []
    for k in scales:
        edges, ai, ao = topology.knn_edges_csr(knn_idx[:, :int(k)])
        if mreach:
            _, elev = topology.edge_fields(X, edges, ell)
            maxima.append(topology.max_prominence_edges(
                ell, edges[:, 0], edges[:, 1], elev
            ))
        else:
            maxima.append(topology.max_prominence_arrays(ell, ai, ao))
        components.append(np.unique(
            topology.distance_blocks(ai, ao, len(X))
        ).size)
    return np.asarray(maxima), np.asarray(components, dtype=np.int64)


def test_fused_multiscale_statistics_match_reference_exactly():
    """Verify fused multiscale statistics match reference exactly."""
    for seed, n, d in ((0, 300, 4), (17, 240, 2), (9, 180, 3)):
        rng = np.random.default_rng(seed)
        X = np.ascontiguousarray(rng.normal(size=(n, d)))
        if seed == 0:
            X[1] = X[0]  # include duplicate points / zero-length edge handling
        kd, ki = _query_knn_euclidean(X, 15, workers=1)
        density, _ = adaptive_knn_density_from_distances(
            kd, 4, 2.5, 4, 2, d_hat=float(d)
        )
        ell = np.log(np.maximum(density, 1e-300))
        scales = np.asarray([4, 6, 9, 12, 15], dtype=np.int64)

        for mreach in (False, True):
            got = topology.multiscale_graph_statistics(X, ki, ell, scales, mreach=mreach)
            expected = _reference_multiscale(X, ki, ell, scales, mreach=mreach)
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])


def test_knn_edge_native_dedup_matches_numpy_unique_reference():
    """Verify knn edge native dedup matches numpy unique reference."""
    rng = np.random.default_rng(121)
    n, k = 97, 13
    ki = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        candidates = np.delete(np.arange(n), i)
        ki[i] = rng.choice(candidates, size=k, replace=False)

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = ki.reshape(-1)
    keys, first = np.unique(
        np.minimum(src, dst) * n + np.maximum(src, dst),
        return_index=True,
    )
    expected_edges = np.column_stack((keys // n, keys % n))

    edges, ai, ao = topology.knn_edges_csr(ki)
    np.testing.assert_array_equal(edges, expected_edges)
    expected_ai, expected_ao = topology.edges_to_csr(n, expected_edges)
    np.testing.assert_array_equal(ai, expected_ai)
    np.testing.assert_array_equal(ao, expected_ao)

    edges2, ai2, ao2, first2 = topology.knn_edges_csr(ki, return_first=True)
    np.testing.assert_array_equal(edges2, expected_edges)
    np.testing.assert_array_equal(first2, first)
    np.testing.assert_array_equal(ai2, expected_ai)
    np.testing.assert_array_equal(ao2, expected_ao)


def test_fused_selected_tree_respects_configured_mutual_reachability(monkeypatch):
    """Verify fused selected tree respects configured mutual reachability."""
    import entrap.tree as tree_mod
    from entrap.config import EntrapConfig
    from entrap.graph import build_knn_graph
    from entrap.tree import build_tree
    from helpers import make_blobs

    X, _ = make_blobs(n_per=24, std=0.35, seed=41)
    real_build_filtration = tree_mod.build_filtration_from_density

    for mreach in (False, True):
        cfg = EntrapConfig.from_params(
            8, 0.1,
            {
                "P": 19,
                "B": 1,
                "stability_B": 0,
                "pi_runs": 1,
                "use_leiden": False,
                "mutual_reachability": mreach,
                "seed": 7,
            },
        ).resolve(len(X))
        graph = build_knn_graph(X, cfg.k_resolved, store_knn=True)
        observed_flags = []

        def spy_build_filtration(X_node, edges, base_density, mreach=True):
            """Record filtration calls while delegating to the original builder."""
            observed_flags.append(bool(mreach))
            return real_build_filtration(
                X_node, edges, base_density, mreach=mreach
            )

        monkeypatch.setattr(
            tree_mod, "build_filtration_from_density", spy_build_filtration
        )
        build_tree(
            graph, X, np.arange(len(X), dtype=np.int64),
            cfg, None, np.random.default_rng(9), internal_stability=False,
        )
        assert observed_flags, "selected merge tree should be materialized"
        assert all(flag is bool(mreach) for flag in observed_flags)
