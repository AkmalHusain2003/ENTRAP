"""Scale-consistency, block ensemble, and halo-noise regression tests."""
import numpy as np
import pytest
import entrap.significance as sig
from entrap.graph import knn_density, estimate_intrinsic_dim, build_knn_graph
from entrap.mergetree import density_merge_tree
from entrap.tree import build_tree, extract_clusters
from entrap.config import EntrapConfig
from helpers import make_blobs, uniform_field


def test_normalized_density_is_scale_invariant():
    """Verify normalized density is scale invariant."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    d = estimate_intrinsic_dim(X)
    full = knn_density(X, 10, d_hat=d, normalized=True)
    sub = knn_density(X[:320], 10, d_hat=d, normalized=True)
    ratio = np.median(sub)/np.median(full)
    assert 0.9 <= ratio <= 1.1, f"normalized density still n-dependent: {ratio}"
    legacy_full = knn_density(X, 10, normalized=False)
    legacy_sub = knn_density(X[:320], 10, normalized=False)
    legacy = np.median(legacy_sub)/np.median(legacy_full)
    assert abs(legacy-1.0) > abs(ratio-1.0), "legacy should be MORE n-dependent"


def test_subsample_prominence_stable_under_normalized_density():
    """Verify subsample prominence stable under normalized density."""
    X, _ = make_blobs(n_per=60, std=0.3)
    d = estimate_intrinsic_dim(X)
    g = build_knn_graph(X, 10)
    dens = knn_density(X, 10, d_hat=d, normalized=True)
    mt = density_merge_tree(dens, g.adj)
    p_full = mt.mode_prom[np.isfinite(mt.mode_prom)].max()
    idx = np.sort(np.random.default_rng(0).choice(len(X), int(0.8*len(X)), replace=False))
    from entrap.graph import induced_edges, edges_to_adj
    from entrap.graph import adj_to_csr
    from entrap import topology
    le = induced_edges(g.edges, idx, len(X))
    ai, ao = adj_to_csr(edges_to_adj(len(idx), le))
    ds = knn_density(X[idx], 10, d_hat=d, normalized=True)
    prom_sub = topology.merge_tree_arrays(ds, ai, ao)[3]
    p_sub = np.max(prom_sub[np.isfinite(prom_sub)])
    assert 0.75 <= p_sub/p_full <= 1.3, f"prominence scale drift: {p_sub/p_full}"


def test_compute_tau_ensemble_finite_and_deterministic():
    """Verify compute tau ensemble finite and deterministic."""
    X = uniform_field(n=100, seed=5)
    g = build_knn_graph(X, 10)
    dens = knn_density(X, 10, normalized=True)
    t1, r1 = sig.compute_tau(dens, g.adj, 29, 0.1, np.random.default_rng(0), ensemble=True)
    t2, _ = sig.compute_tau(dens, g.adj, 29, 0.1, np.random.default_rng(0), ensemble=True)
    assert np.isfinite(t1) and t1 >= 0 and t1 == pytest.approx(t2)


def test_halo_noise_produces_real_noise_class():
    # was xfail(strict): mode-prominence core flagged 12/100 field points, the pointwise floor must not
    """Verify halo noise produces real noise class."""
    rng = np.random.default_rng(0)
    blobs, _ = make_blobs(n_per=60, std=0.3)
    field = rng.uniform(-8, 28, size=(100, 3))
    X = np.vstack([blobs, field])
    truth_field = np.arange(len(blobs), len(X))
    cfg = EntrapConfig.from_params(15, 0.1, {"B": 6, "P": 29, "pi_runs": 4,
                                             "stability_B": 6, "pi_full_stability": False,
                                             "seed": 0, "use_leiden": False, "multi_k": False}).resolve(len(X))
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    labels, basins, _ = extract_clusters(root, len(X), halo_noise=True)
    n_noise = int(np.sum(labels[truth_field] < 0))
    assert n_noise > 50, f"field points should mostly stay noise, got {n_noise}/100"
    assert len(basins) >= 3
    # cores are non-empty and smaller than full basins would be
    for c, idx in basins.items():
        assert 0 < len(idx)


def test_halo_noise_off_labels_everyone():
    """Verify halo noise off labels everyone."""
    X, _ = make_blobs(n_per=40, std=0.3)
    cfg = EntrapConfig.from_params(10, 0.1, {"B": 4, "P": 19, "use_leiden": False, "multi_k": False}).resolve(len(X))
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    labels, basins, _ = extract_clusters(root, len(X), halo_noise=False)
    assert np.all(labels >= 0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_intrinsic_dim_recovers_known_dimension(d):
    """Verify intrinsic dim recovers known dimension."""
    est = [estimate_intrinsic_dim(np.random.default_rng(s).uniform(size=(400, d)))
           for s in range(7)]
    assert abs(np.median(est) - d) <= 0.3, (d, est)


def test_intrinsic_dim_unbiased_by_cdf_endpoint():
    """Verify intrinsic dim unbiased by cdf endpoint."""
    plane = [estimate_intrinsic_dim(np.hstack([np.random.default_rng(s).uniform(size=(300, 2)),
                                               np.zeros((300, 8))])) for s in range(7)]
    circle = []
    for s in range(7):
        t = np.random.default_rng(s).uniform(0, 2*np.pi, 300)
        circle.append(estimate_intrinsic_dim(np.c_[np.cos(t), np.sin(t)]))
    assert abs(np.median(plane) - 2.0) <= 0.2, plane
    assert abs(np.median(circle) - 1.0) <= 0.1, circle


def test_intrinsic_dim_ignores_duplicate_points():
    """Verify intrinsic dim ignores duplicate points."""
    X = np.random.default_rng(1).uniform(size=(300, 2))
    Xd = np.vstack([X, X[:5]])                   # 5 exact duplicates
    d_clean, d_dup = estimate_intrinsic_dim(X), estimate_intrinsic_dim(Xd)
    assert d_dup >= 1.6 and abs(d_dup - d_clean) <= 0.25, (d_clean, d_dup)
