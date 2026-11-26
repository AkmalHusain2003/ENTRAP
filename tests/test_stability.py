import numpy as np
import pytest
import entrap.stability as stab
from entrap.graph import build_knn_graph
from entrap.tree import build_tree, extract_clusters
from entrap.stability import compute_pi
from helpers import make_blobs


def test_pi_shapes_and_range(blobs, fast_cfg, rng):
    """Verify pi shapes and range."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    labels, basins, _ = extract_clusters(root, len(X))
    pi, pi_pc, _ = compute_pi(g, X, labels, basins, fast_cfg, None, rng)
    assert pi.shape == (len(X),)
    assert pi_pc.shape == (len(X), len(basins))
    assert np.all((pi >= 0) & (pi <= 1))
    assert np.all((pi_pc >= 0) & (pi_pc <= 1))
    assert np.allclose(pi, pi_pc.sum(axis=1), atol=1e-9)


def test_pi_high_for_blob_cores(blobs, fast_cfg, rng):
    """Verify pi high for blob cores."""
    X, truth = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    labels, basins, _ = extract_clusters(root, len(X))
    pi, _, _ = compute_pi(g, X, labels, basins, fast_cfg, None, rng)
    assert np.median(pi) > 0.5


def test_pi_full_stability_flag_is_forwarded(monkeypatch, blobs, fast_cfg, rng):
    """Verify pi full stability flag is forwarded."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), fast_cfg, None, rng)
    labels, basins, _ = extract_clusters(root, len(X))
    recorded = []
    real = stab.build_tree

    def spy(*a, **kw):
        """Record calls to the operation being tested."""
        recorded.append(kw.get("internal_stability"))
        return real(*a, **kw)
    monkeypatch.setattr(stab, "build_tree", spy)
    fast_cfg.pi_full_stability = False
    fast_cfg.pi_runs = 2
    compute_pi(g, X, labels, basins, fast_cfg, None, rng)
    assert recorded and all(v is False for v in recorded)


def test_no_clusters_gives_zero_pi(blobs, fast_cfg, rng):
    """Verify no clusters gives zero pi."""
    X, _ = blobs
    g = build_knn_graph(X, fast_cfg.k_resolved)
    labels = np.full(len(X), -1)
    pi, pi_pc, _ = compute_pi(g, X, labels, {}, fast_cfg, None, rng)
    assert np.all(pi == 0)
