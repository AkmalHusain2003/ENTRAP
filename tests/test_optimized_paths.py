"""Parity checks for the fast Euclidean kNN and Poisson-null paths."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from entrap import topology
from entrap.graph import (
    build_knn_graph, _query_knn_euclidean, _edges_and_distances_from_knn,
    edges_to_csr, log_density, edge_levels,
)
from entrap.significance import poisson_null_tau


def _legacy_knn(X, k):
    """Build reference kNN indices and undirected edges with scikit-learn."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(X)
    dist, idx = nn.kneighbors(X)
    pairs = set()
    for i in range(len(X)):
        for j in idx[i, 1:]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return dist[:, 1:], idx[:, 1:], edges


def _legacy_poisson_maxima(X, k, P, rng, d_hat=None, normalized=False, mreach=False, dim=None):
    """Compute Poisson null maximum prominences with the reference implementation."""
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    lo, hi = X.min(axis=0), X.max(axis=0)
    m = D if dim is None else int(dim)
    out = np.empty(P)
    for p in range(P):
        U = rng.random((n, m))
        Xs = lo + U * (hi - lo) if dim is None else U
        kd, ki, edges = _legacy_knn(Xs, min(k, n - 1))
        if not normalized:
            base = 1.0 / np.maximum(kd.mean(axis=1), 1e-12)
        else:
            from entrap.graph import knn_density
            base = knn_density(Xs, k, d_hat=d_hat, normalized=True)
        ell = log_density(base)
        if mreach:
            ea, eb = edges[:, 0], edges[:, 1]
            dist = np.linalg.norm(Xs[ea] - Xs[eb], axis=1)
            resid = -np.log(np.maximum(dist, 1e-300)) - 0.5 * (ell[ea] + ell[eb])
            elev = edge_levels(ell, edges, resid)
            mpr = topology.merge_tree_edges(
                np.ascontiguousarray(ell), np.ascontiguousarray(ea),
                np.ascontiguousarray(eb), np.ascontiguousarray(elev))[3]
        else:
            ai, ao = edges_to_csr(n, edges)
            mpr = topology.merge_tree_arrays(np.ascontiguousarray(ell), ai, ao)[3]
        finite = mpr[np.isfinite(mpr)]
        out[p] = finite.max() if len(finite) else 0.0
    return out


def test_euclidean_knn_matches_sklearn_on_continuous_data():
    """Verify euclidean knn matches sklearn on continuous data."""
    rng = np.random.default_rng(41)
    X = rng.normal(size=(180, 4))
    k = 13
    dist_ref, idx_ref, edges_ref = _legacy_knn(X, k)
    model = build_knn_graph(X, k, metric="euclidean")
    np.testing.assert_allclose(model.knn_dist, dist_ref, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(model.knn_idx, idx_ref)
    np.testing.assert_array_equal(model.edges, edges_ref)
    np.testing.assert_allclose(model.density, 1.0 / dist_ref.mean(axis=1), rtol=0, atol=1e-14)


def test_euclidean_knn_excludes_self_with_duplicate_points():
    """Verify euclidean knn excludes self with duplicate points."""
    X = np.array([[0., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    d, ix = _query_knn_euclidean(X, 3)
    assert d.shape == ix.shape == (len(X), 3)
    np.testing.assert_array_equal(ix == np.arange(len(X))[:, None], np.zeros_like(ix, dtype=bool))
    assert np.all(d >= 0)


def test_native_max_prominence_matches_full_merge_tree():
    """Verify native max prominence matches full merge tree."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 3))
    graph = build_knn_graph(X, 11)
    ell = log_density(1.0 / np.maximum(graph.knn_dist.mean(axis=1), 1e-12))
    ai, ao = edges_to_csr(len(X), graph.edges)
    full = topology.merge_tree_arrays(np.ascontiguousarray(ell), ai, ao)[3]
    fast = topology.max_prominence_arrays(np.ascontiguousarray(ell), ai, ao)
    expected = full[np.isfinite(full)].max(initial=0.0)
    assert fast == expected

    ea = np.ascontiguousarray(graph.edges[:, 0], dtype=np.int64)
    eb = np.ascontiguousarray(graph.edges[:, 1], dtype=np.int64)
    dist = np.linalg.norm(X[ea] - X[eb], axis=1)
    resid = -np.log(np.maximum(dist, 1e-300)) - 0.5 * (ell[ea] + ell[eb])
    elev = edge_levels(ell, graph.edges, resid)
    full_e = topology.merge_tree_edges(np.ascontiguousarray(ell), ea, eb,
                                       np.ascontiguousarray(elev))[3]
    fast_e = topology.max_prominence_edges(np.ascontiguousarray(ell), ea, eb,
                                           np.ascontiguousarray(elev))
    assert fast_e == full_e[np.isfinite(full_e)].max(initial=0.0)


def test_poisson_fast_path_has_exact_maxima_parity():
    """Verify poisson fast path has exact maxima parity."""
    rng = np.random.default_rng(17)
    X = rng.random((90, 3))
    for mreach in (False, True):
        for normalized in (False, True):
            kwargs = dict(d_hat=2.25, normalized=normalized, mreach=mreach)
            new = poisson_null_tau(X, 9, 9, 0.1, np.random.default_rng(123), **kwargs)[1]
            old = _legacy_poisson_maxima(X, 9, 9, np.random.default_rng(123), **kwargs)
            np.testing.assert_allclose(new, old, rtol=0, atol=1e-12)
