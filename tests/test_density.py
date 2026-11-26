"""Full per-node density re-estimation (the fidelity fix)."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from entrap.graph import knn_density, build_knn_graph, induced_subgraph
from helpers import make_blobs


def test_knn_density_matches_bruteforce():
    """Verify knn density matches bruteforce."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    k = 5
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.sort(D, axis=1)[:, :k]
    expected = 1.0 / nn.mean(axis=1)
    np.testing.assert_allclose(knn_density(X, k), expected, rtol=1e-10)


def test_k_capped_at_n_minus_one():
    """Verify k capped at n minus one."""
    X = np.random.default_rng(0).normal(size=(5, 2))
    d = knn_density(X, 100)
    assert d.shape == (5,) and np.all(np.isfinite(d))


def test_subset_density_differs_from_restricted_parent():
    """Verify subset density differs from restricted parent."""
    X, _ = make_blobs(n_per=40)
    parent = knn_density(X, 10)
    idx = np.arange(0, 120, 3)
    re_est = knn_density(X[idx], 10)
    assert not np.allclose(re_est, parent[idx])


def test_induced_subgraph_edges_and_adj():
    """Verify induced subgraph edges and adj."""
    X, _ = make_blobs(n_per=20)
    g = build_knn_graph(X, 8)
    S = np.arange(0, 60, 2)
    sub = induced_subgraph(g, S)
    assert len(sub.adj) == len(S)
    loc = {int(gidx): i for i, gidx in enumerate(S)}
    for a, b in sub.edges:
        assert 0 <= a < len(S) and 0 <= b < len(S)
        ga, gb = int(S[a]), int(S[b])
        assert (ga in g.adj[gb]) and (gb in g.adj[ga])
    for i, nbrs in enumerate(sub.adj):
        for j in nbrs:
            assert i in sub.adj[j]
