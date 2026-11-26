import numpy as np
import pytest
import entrap.tree as tree_mod
from entrap.graph import build_knn_graph
from entrap.tree import build_tree, extract_clusters
from helpers import make_blobs


def test_tree_finds_separated_blobs(blobs, fast_cfg, rng):
    """Verify tree finds separated blobs."""
    X, truth = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    labels, basins, proms = extract_clusters(root, len(X))
    assert len(basins) >= 3
    # purity: no cluster may span two planted blobs (graph is disconnected)
    for c, idx in basins.items():
        assert len(np.unique(truth[idx])) == 1
    assert set(proms) == set(basins)


def test_leaf_invariants(blobs, fast_cfg, rng):
    """Verify leaf invariants."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.is_leaf:
            assert not nd.children
            if nd.is_cluster:
                assert len(nd.indices) >= fast_cfg.min_cluster_size
                assert nd.m_star == 1
        else:
            assert nd.children
            assert nd.m_star >= 2
            stack.extend(nd.children)


def test_each_recursive_node_rebuilds_its_own_knn_graph(monkeypatch, blobs, fast_cfg, rng):
    """Verify each recursive node rebuilds its own knn graph."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    seen = []
    real = tree_mod.build_knn_graph

    def spy(X_node, *a, **kw):
        """Record calls to the operation being tested."""
        seen.append(len(X_node))
        return real(X_node, *a, **kw)
    monkeypatch.setattr(tree_mod, "build_knn_graph", spy)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    # The exact root cache is reused, while each child gets a local neighbour
    # query. This is the V7 fix for parent-induced-subgraph scale mismatch.
    assert len(seen) >= 2
    assert all(10 <= n < len(X) for n in seen)
    assert root.selected_k in root.k_scales
    assert root.k_scores.shape == (len(root.k_scales),)
    assert root.local_k_summary["min"] <= root.local_k_summary["median"] <= root.local_k_summary["max"]


def test_min_cluster_size_suppresses_clusters(blobs, rng):
    """Verify min cluster size suppresses clusters."""
    from entrap.config import EntrapConfig
    X, _ = blobs
    cfg = EntrapConfig.from_params(10_000, 0.1, {"B": 3, "P": 9, "use_leiden": False}).resolve(len(X))
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, rng)
    labels, basins, _ = extract_clusters(root, len(X))
    assert len(basins) == 0
    assert np.all(labels == -1)


def test_recursion_terminates_with_bounded_depth(blobs, fast_cfg, rng):
    """Verify recursion terminates with bounded depth."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)

    def depth(nd):
        """Return the maximum depth of the generated hierarchy."""
        return 0 if nd.is_leaf else 1 + max(depth(c) for c in nd.children)
    assert depth(root) <= int(np.log2(len(X))) + 2
