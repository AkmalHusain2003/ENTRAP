"""Null threshold, stable mode count, and verification of full re-estimation."""
import numpy as np
import pytest
import entrap.significance as sig
from entrap.graph import build_knn_graph, knn_density, adj_to_csr
from entrap.mergetree import density_merge_tree
from entrap import topology
from helpers import make_blobs, uniform_field


def _setup(X, k=10):
    """Create the graph and density inputs for the significance test."""
    g = build_knn_graph(X, k)
    dens = knn_density(X, k)
    return g, dens


def test_null_threshold_positive_and_deterministic():
    """Verify null threshold positive and deterministic."""
    X = uniform_field(n=90, seed=2)
    g, dens = _setup(X)
    blocks = sig.graph_distance_blocks(g.adj, 2)
    t1 = sig.prominence_null(dens, g.adj, blocks, 39, 0.1, np.random.default_rng(0))
    t2 = sig.prominence_null(dens, g.adj, blocks, 39, 0.1, np.random.default_rng(0))
    assert t1 == pytest.approx(t2)
    assert t1 >= 0.0


def test_null_threshold_monotone_in_alpha():
    """Verify null threshold monotone in alpha."""
    X = uniform_field(n=90, seed=3)
    g, dens = _setup(X)
    blocks = sig.graph_distance_blocks(g.adj, 2)
    lo = sig.prominence_null(dens, g.adj, blocks, 49, 0.30, np.random.default_rng(1))
    hi = sig.prominence_null(dens, g.adj, blocks, 49, 0.02, np.random.default_rng(1))
    assert hi >= lo


def test_signal_exceeds_null_on_separated_blobs():
    """Verify signal exceeds null on separated blobs."""
    X, _ = make_blobs(n_per=35)
    g, dens = _setup(X, k=8)
    blocks = sig.graph_distance_blocks(g.adj, 2)
    tau = sig.prominence_null(dens, g.adj, blocks, 39, 0.05, np.random.default_rng(0))
    mt = density_merge_tree(dens, g.adj)
    # three disconnected components -> three essential (infinite) modes
    assert np.isinf(mt.mode_prom).sum() == 3
    assert mt.count_significant(tau) >= 3


def test_mode_count_stability_recovers_components():
    """Verify mode count stability recovers components."""
    X, _ = make_blobs(n_per=35)
    g, dens = _setup(X, k=8)
    blocks = sig.graph_distance_blocks(g.adj, 2)
    tau = sig.prominence_null(dens, g.adj, blocks, 29, 0.1, np.random.default_rng(0))
    mstar, counts = sig.mode_count_stability(X, g.edges, tau, B=9, f=0.8,
                                             alpha=0.1, k=8, rng=np.random.default_rng(0))
    assert mstar >= 3
    assert len(counts) == 9 and np.all(counts >= 3)


def test_mode_count_stability_field_is_small():
    """Verify mode count stability field is small."""
    X = uniform_field(n=110, seed=7)
    g, dens = _setup(X, k=10)
    blocks = sig.graph_distance_blocks(g.adj, 2)
    tau = sig.prominence_null(dens, g.adj, blocks, 39, 0.1, np.random.default_rng(0))
    mstar, _ = sig.mode_count_stability(X, g.edges, tau, B=9, f=0.8,
                                        alpha=0.1, k=10, rng=np.random.default_rng(0))
    assert mstar <= 2


def test_full_density_reestimation_is_called_per_subsample(monkeypatch):
    """Verify full density reestimation is called per subsample."""
    X = uniform_field(n=50, seed=4)
    g, _ = _setup(X, k=8)
    calls = []
    import entrap.filtration as fil_mod        # density is estimated inside build_filtration
    real = fil_mod.knn_density

    def spy(*a, **kw):
        """Record calls to the operation being tested."""
        calls.append(a[0].shape)
        return real(*a, **kw)
    monkeypatch.setattr(fil_mod, "knn_density", spy)
    sig.mode_count_stability(X, g.edges, tau=1e9, B=5, f=0.8, alpha=0.1,
                             k=8, rng=np.random.default_rng(0))
    assert len(calls) == 5
    assert all(shape == (40, 3) for shape in calls)   # floor(0.8*50) = 40


@pytest.mark.slow
def test_null_false_positive_rate_on_pure_field():
    """Verify null false positive rate on pure field."""
    alpha, P, R = 0.10, 49, 30
    exceed = 0
    for r in range(R):
        X = uniform_field(n=100, seed=1000 + r)
        g, dens = _setup(X, k=10)
        blocks = sig.graph_distance_blocks(g.adj, 2)
        tau = sig.prominence_null(dens, g.adj, blocks, P, alpha, np.random.default_rng(r))
        mt = density_merge_tree(dens, g.adj)
        fin = mt.mode_prom[np.isfinite(mt.mode_prom)]
        obs = fin.max() if len(fin) else 0.0
        exceed += int(obs > tau)
    rate = exceed / R
    print(f"\nnull exceedance rate = {rate:.3f} (target {alpha})")
    assert rate <= alpha + 0.15, f"null is anti-conservative: {rate}"
