import numpy as np
import pytest
from entrap.leiden import has_leiden, leiden_partition
from entrap.config import EntrapConfig


def _cfg(use_leiden):
    """Build the Leiden configuration used by the test."""
    return EntrapConfig.from_params(5, 0.1, {"use_leiden": use_leiden}).resolve(60)


def test_fallback_returns_seed_partition():
    """Verify fallback returns seed partition."""
    seed = np.array([0, 0, 1, 1, 2, 2])
    edges = np.array([[0, 1], [2, 3], [4, 5]])
    w = np.ones(len(edges))
    out = leiden_partition(6, edges, w, 3, seed, (0.0, 1.0), _cfg(False), np.random.default_rng(0))
    np.testing.assert_array_equal(out, seed)


@pytest.mark.skipif(not has_leiden(), reason="leidenalg/igraph not installed")
def test_leiden_produces_valid_partition():
    """Verify leiden produces valid partition."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (25, 3)) for c in ([0, 0, 0], [8, 0, 0])])
    from entrap.graph import build_knn_graph
    g = build_knn_graph(X, 8)
    seed = np.repeat([0, 1], 25)
    out, info = leiden_partition(len(X), g.edges, g.weights, 2, seed, (0.0, 1.0),
                                 _cfg(True), np.random.default_rng(0), return_info=True)
    assert out.shape == (len(X),)
    assert len(np.unique(out)) >= 1
    assert set(np.unique(out)) <= set(range(len(X)))
    assert info["backend"] == "igraph+leidenalg"
    assert info["status"] in {
        "optimized_exact_target", "seed_fallback_no_exact_target_partition"
    }
    if info["used_leiden"]:
        import igraph as ig
        graph = ig.Graph(n=len(X), edges=[tuple(map(int, e)) for e in g.edges], directed=False)
        for community in np.unique(out):
            vertices = np.flatnonzero(out == community).tolist()
            if len(vertices) > 1:
                assert graph.induced_subgraph(vertices).is_connected()
