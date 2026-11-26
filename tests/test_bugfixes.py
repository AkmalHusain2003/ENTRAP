"""Regression tests for the seven audited defects."""
import numpy as np
import pytest
from entrap import topology
from entrap.graph import adj_to_csr, build_knn_graph
from entrap.leiden import has_leiden, leiden_partition, _run
from entrap.config import EntrapConfig
from helpers import make_blobs, random_graph


def test_radius_zero_singletons():
    """Verify radius zero singletons."""
    rng = np.random.default_rng(0)
    dens, adj = random_graph(rng, n=10, p=0.4)
    ai, ao = adj_to_csr(adj)
    assert len(np.unique(topology.distance_blocks(ai, ao, 0))) == 10


def test_mstar_clamped_no_indexerror(monkeypatch):
    """Verify mstar clamped no indexerror."""
    import entrap.tree as T
    X, _ = make_blobs(n_per=20)
    cfg = EntrapConfig.from_params(10, 0.1, {"B": 3, "P": 9, "use_leiden": False}).resolve(len(X))
    g = build_knn_graph(X, cfg.k_resolved)
    monkeypatch.setattr(T, "mode_count_stability", lambda *a, **k: (10**6, np.array([10**6])))
    root = T.build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    assert root.m_star <= len(root.merge_tree.mode_prom)


def test_significance_small_n_counts_consistent():
    """Verify significance small n counts consistent."""
    import entrap.significance as S
    X = np.random.default_rng(0).normal(size=(2, 3))
    edges = np.array([[0, 1]], dtype=np.int64)
    mstar, counts = S.mode_count_stability(X, edges, 0.0, B=3, f=0.8, alpha=0.1,
                                           k=1, rng=np.random.default_rng(0))
    assert counts[0] == mstar


@pytest.mark.skipif(not has_leiden(), reason="needs leidenalg + igraph")
def test_cpm_returns_valid_partition_at_distinct_resolutions():
    """Verify cpm returns valid partition at distinct resolutions."""
    X, _ = make_blobs(n_per=25)
    g = build_knn_graph(X, 8)
    lo = _run(len(X), g.edges, g.weights, 0.01, None, np.random.default_rng(0))
    hi = _run(len(X), g.edges, g.weights, 0.90, None, np.random.default_rng(0))
    assert lo.shape == hi.shape == (len(X),)
    assert 1 <= len(np.unique(lo)) <= len(X)
    assert 1 <= len(np.unique(hi)) <= len(X)


@pytest.mark.skipif(not has_leiden(), reason="needs leidenalg + igraph")
def test_gamma_search_contract():
    """Verify gamma search contract."""
    X, _ = make_blobs(n_per=25, centers=((0., 0., 0.), (10., 0., 0.)))
    g = build_knn_graph(X, 8)
    seed = np.repeat([0, 1], 25)
    cfg = EntrapConfig.from_params(5, 0.1, {"use_leiden": True}).resolve(len(X))
    out = leiden_partition(len(X), g.edges, g.weights, 2, seed, (0., 1.),
                           cfg, np.random.default_rng(7))
    assert len(np.unique(out)) == 2 or np.array_equal(out, seed)


@pytest.mark.skipif(not has_leiden(), reason="needs leidenalg + igraph")
def test_leiden_reproducible():
    """Verify leiden reproducible."""
    X, _ = make_blobs(n_per=25, centers=((0., 0., 0.), (10., 0., 0.)))
    g = build_knn_graph(X, 8)
    seed = np.repeat([0, 1], 25)
    cfg = EntrapConfig.from_params(5, 0.1, {"use_leiden": True}).resolve(len(X))
    a = leiden_partition(len(X), g.edges, g.weights, 2, seed, (0., 1.), cfg, np.random.default_rng(7))
    b = leiden_partition(len(X), g.edges, g.weights, 2, seed, (0., 1.), cfg, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not has_leiden(), reason="needs leidenalg + igraph")
def test_estimator_reproducible_real_backend():
    """Verify estimator reproducible real backend."""
    from entrap import ENTRAP
    X, _ = make_blobs(n_per=25)
    adv = {"B": 3, "P": 9, "pi_runs": 2, "stability_B": 3,
           "pi_full_stability": False, "seed": 0, "use_leiden": True}
    a = ENTRAP(10, 0.2, advanced=adv).fit(X).labels_
    b = ENTRAP(10, 0.2, advanced=adv).fit(X).labels_
    np.testing.assert_array_equal(a, b)


def test_estimator_fails_fast_when_leiden_is_requested_but_missing(monkeypatch):
    """Verify estimator fails fast when leiden is requested but missing."""
    import entrap.leiden as leiden_module
    from entrap import ENTRAP
    monkeypatch.setattr(leiden_module, "_HAS", False)
    monkeypatch.setattr(leiden_module, "_IMPORT_ERROR", ModuleNotFoundError("missing backend"))
    X, _ = make_blobs(n_per=10)
    with pytest.raises(ImportError, match="igraph.*leidenalg"):
        ENTRAP(5, 0.1, advanced={"use_leiden": True, "allow_fallback": False}).fit(X)
