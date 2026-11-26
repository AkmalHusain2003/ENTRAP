"""Log scale + mutual-reachability filtration: formulas, invariances, null/stability consistency."""
import numpy as np
import pytest
import entrap.significance as sig
from entrap import topology
from entrap.config import EntrapConfig
from entrap.diagnostics import persistence_entropy
from entrap.filtration import build_filtration, filtration_merge_tree, filtration_count
from entrap.graph import (build_knn_graph, knn_density, log_density, edge_residual, edge_levels,
                          adj_to_csr)
from entrap.tree import build_tree, extract_clusters
from helpers import make_blobs, uniform_field


def _cfg(n, **adv):
    """Build the Leiden configuration used by the test."""
    base = {"B": 4, "P": 19, "pi_runs": 3, "stability_B": 4, "pi_full_stability": False,
            "seed": 0, "use_leiden": False, "multi_k": False}
    base.update(adv)
    return EntrapConfig.from_params(10, 0.1, base).resolve(n)


def test_log_density_is_log_and_guards_zero():
    """Verify log density is log and guards zero."""
    d = np.array([1.0, np.e, 0.0])
    out = log_density(d)
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(1.0) and np.isfinite(out[2])


@pytest.mark.parametrize("s", [0.05, 3.0, 1e4])
def test_log_prominences_invariant_to_multiplicative_scale(s):
    """Verify log prominences invariant to multiplicative scale."""
    X, _ = make_blobs(n_per=60, std=0.3)
    g = build_knn_graph(X, 10)
    rho = knn_density(X, 10)
    ai, ao = adj_to_csr(g.adj)
    a = topology.merge_tree_arrays(np.ascontiguousarray(log_density(rho)), ai, ao)[3]
    b = topology.merge_tree_arrays(np.ascontiguousarray(log_density(s*rho)), ai, ao)[3]
    np.testing.assert_allclose(np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)]), atol=1e-10)


def test_edge_levels_equal_min_of_endpoints_and_edge_term():
    """Verify edge levels equal min of endpoints and edge term."""
    X, _ = make_blobs(n_per=40, std=0.3)
    g = build_knn_graph(X, 10)
    ell = log_density(knn_density(X, 10))
    r = edge_residual(X, g.edges, ell)
    lev = edge_levels(ell, g.edges, r)
    d = np.linalg.norm(X[g.edges[:, 0]] - X[g.edges[:, 1]], axis=1)
    brute = np.minimum.reduce([ell[g.edges[:, 0]], ell[g.edges[:, 1]], -np.log(d)])
    np.testing.assert_allclose(lev, brute, atol=1e-12)
    assert np.all(lev <= np.minimum(ell[g.edges[:, 0]], ell[g.edges[:, 1]]) + 1e-15)


@pytest.mark.parametrize("s", [0.01, 7.0, 1e3])
def test_residual_and_prominences_are_scale_free(s):
    """Verify residual and prominences are scale free."""
    X, _ = make_blobs(n_per=40, std=0.3)
    g = build_knn_graph(X, 10)
    f1, _ = build_filtration(X, g.edges, 10, mreach=True)
    f2, _ = build_filtration(s*X, g.edges, 10, mreach=True)
    np.testing.assert_allclose(f1.resid, f2.resid, atol=1e-9)
    p1 = filtration_merge_tree(f1, g.adj).mode_prom
    p2 = filtration_merge_tree(f2, g.adj).mode_prom
    np.testing.assert_allclose(np.sort(p1[np.isfinite(p1)]), np.sort(p2[np.isfinite(p2)]), atol=1e-9)


def test_filtration_without_mreach_matches_vertex_tree():
    """Verify filtration without mreach matches vertex tree."""
    X, _ = make_blobs(n_per=40, std=0.3)
    g = build_knn_graph(X, 10)
    fv, _ = build_filtration(X, g.edges, 10, mreach=False)
    assert not fv.mreach and fv.resid is None
    mt = filtration_merge_tree(fv, g.adj)
    assert mt.count_significant(0.5) == filtration_count(fv, 0.5)


def test_mreach_prominence_never_below_vertex_prominence_total():
    """Verify mreach prominence never below vertex prominence total."""
    X, _ = make_blobs(n_per=30, std=0.5, centers=((0., 0., 0.), (1.5, 0., 0.)))
    g = build_knn_graph(X, 10)
    fv, _ = build_filtration(X, g.edges, 10, mreach=False)
    fm, _ = build_filtration(X, g.edges, 10, mreach=True)
    pv = filtration_merge_tree(fv, g.adj).mode_prom
    pm = filtration_merge_tree(fm, g.adj).mode_prom
    assert pm[np.isfinite(pm)].sum() >= pv[np.isfinite(pv)].sum() - 1e-9


def test_null_with_no_edge_lowering_equals_vertex_null():
    """Verify null with no edge lowering equals vertex null."""
    X = uniform_field(n=120, dim=3, seed=2)
    g = build_knn_graph(X, 10)
    ell = log_density(knn_density(X, 10))
    blocks = sig.graph_distance_blocks(g.adj, 1)
    a = sig.prominence_null(ell, g.adj, blocks, 49, 0.1, np.random.default_rng(3))
    b = sig.prominence_null(ell, g.adj, blocks, 49, 0.1, np.random.default_rng(3),
                            edges=g.edges, resid=np.full(len(g.edges), 1e6))
    assert a == pytest.approx(b, abs=1e-12)


def test_compute_tau_edge_null_finite_and_deterministic():
    """Verify compute tau edge null finite and deterministic."""
    X = uniform_field(n=100, dim=3, seed=5)
    g = build_knn_graph(X, 10)
    fil, _ = build_filtration(X, g.edges, 10, mreach=True)
    t1, _ = sig.compute_tau(fil.ell, g.adj, 29, 0.1, np.random.default_rng(0), ensemble=True,
                            edges=fil.edges, resid=fil.resid)
    t2, _ = sig.compute_tau(fil.ell, g.adj, 29, 0.1, np.random.default_rng(0), ensemble=True,
                            edges=fil.edges, resid=fil.resid)
    assert np.isfinite(t1) and t1 >= 0.0 and t1 == t2


def test_stability_uses_same_filtration_and_counts_blobs():
    """Verify stability uses same filtration and counts blobs."""
    X, _ = make_blobs(n_per=50, std=0.3)
    g = build_knn_graph(X, 10)
    for mr in (False, True):
        fil, _ = build_filtration(X, g.edges, 10, mreach=mr)
        tau, _ = sig.compute_tau(fil.ell, g.adj, 29, 0.1, np.random.default_rng(0), ensemble=True,
                                 edges=fil.edges if mr else None, resid=fil.resid)
        ms, counts = sig.mode_count_stability(X, g.edges, tau, 8, 0.8, 0.1, 10,
                                              np.random.default_rng(1), mreach=mr)
        assert ms == 3 and counts.min() >= 3


def test_subsample_filtration_edges_are_local():
    """Verify subsample filtration edges are local."""
    X, _ = make_blobs(n_per=60, std=0.3)
    g = build_knn_graph(X, 10)
    from entrap.graph import induced_edges
    idx = np.sort(np.random.default_rng(0).choice(len(X), int(0.8*len(X)), replace=False))
    le = induced_edges(g.edges, idx, len(X))
    fil, _ = build_filtration(X[idx], le, 10, mreach=True)
    assert le.max() < len(idx) and len(fil.resid) == len(le) and len(fil.ell) == len(idx)


@pytest.mark.parametrize("mr", [False, True])
def test_pipeline_three_blobs_for_each_filtration(mr):
    """Verify pipeline three blobs for each filtration."""
    X, y = make_blobs(n_per=50, std=0.3)
    cfg = _cfg(len(X), mutual_reachability=mr)
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    labels, basins, _ = extract_clusters(root, len(X), halo_noise=True)
    assert len(basins) == 3
    from helpers import partition_equal
    assert partition_equal(labels[labels >= 0], y[labels >= 0])


def test_config_defaults_v3():
    """Verify config defaults v3."""
    c = EntrapConfig()
    assert c.mutual_reachability is False and c.halo_mode == "poisson" and c.halo_alpha == 0.001 and c.normalized_density is False
    assert not hasattr(c, "density_transform")
    for bad in ({"halo_eta": 1.0}, {"halo_mode": "x"}, {"halo_alpha": 0.7}, {"halo_alpha": 1e-6}):
        with pytest.raises(ValueError):
            EntrapConfig.from_params(10, 0.1, bad)


def test_entropy_superlevel_pairs_birth_above_death():
    """Verify entropy superlevel pairs birth above death."""
    pairs = np.column_stack([np.full(5, 3.0), np.full(5, 1.0)])
    assert persistence_entropy(pairs) == pytest.approx(np.log(5), rel=1e-9)


def test_filtration_count_and_tree_use_edge_levels():
    """Verify filtration count and tree use edge levels."""
    from entrap.filtration import Filtration
    ell = np.array([3.0, 1.0, 2.0])
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    adj = [[1], [0, 2], [1]]
    fv = Filtration(ell, edges)
    fm = Filtration(ell, edges, resid=np.zeros(2), elev=np.array([-5.0, -5.0]))
    assert filtration_count(fv, 3.0) == 1 and filtration_count(fm, 3.0) == 3
    assert filtration_count(fv, 0.5) == 2 and filtration_count(fm, 6.5) == 2
    mt = filtration_merge_tree(fm, adj)
    assert np.sort(mt.mode_prom[np.isfinite(mt.mode_prom)])[-1] == pytest.approx(7.0)
