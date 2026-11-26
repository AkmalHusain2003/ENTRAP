"""Poisson-model halo: null distribution, calibration against simulation, detection, safeguards."""
import numpy as np
import pytest
from entrap.config import EntrapConfig
from entrap.graph import build_knn_graph, knn_density, log_density
from entrap.halo import poisson_null_quantiles, poisson_core, fixed_core, halo_core
from entrap.tree import build_tree, extract_clusters
from helpers import make_blobs


def test_null_quantiles_match_closed_form_d1_k1():
    """Verify null quantiles match closed form d1 k1."""
    g50, ghi, sd = poisson_null_quantiles(1, 1, 0.01)
    assert g50 == pytest.approx(-np.log(np.log(2.0)), abs=0.02)
    assert sd == pytest.approx(np.pi/np.sqrt(6.0), abs=0.03)


def test_null_quantiles_cached_and_ordered():
    """Verify null quantiles cached and ordered."""
    a = poisson_null_quantiles(15, 3, 0.01)
    b = poisson_null_quantiles(15, 3, 0.01)
    assert a == b and a[1] > a[0] and a[2] > 0


@pytest.mark.parametrize("d", [2, 3])
def test_poisson_theory_predicts_log_density_of_homogeneous_process(d):
    """Verify poisson theory predicts log density of homogeneous process."""
    from scipy.special import gamma as G
    rng = np.random.default_rng(11)
    n = 6000 if d == 2 else 8000
    X = rng.uniform(0, 1, (n, d))
    k = 15
    ell = log_density(knn_density(X, k))
    inner = np.all((X > 0.2) & (X < 0.8), axis=1)
    lam = n
    Vd = np.pi**(d/2)/G(d/2+1)
    c = np.log(lam*Vd)/d
    s = -np.log(np.mean(np.cumsum(np.random.default_rng(0).exponential(1, (200000, k)), 1)**(1.0/d), 1))
    assert ell[inner].mean() == pytest.approx(c + s.mean(), abs=0.03)
    assert ell[inner].std() == pytest.approx(s.std(), rel=0.2)


def _blob_plus_noise(seed=0, n_cl=150, n_noise=120):
    """Generate a Gaussian cluster embedded in uniform noise."""
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal(0, 0.35, (n_cl, 3)), r.uniform(-3, 3, (n_noise, 3))])
    y = np.r_[np.zeros(n_cl, int), np.full(n_noise, -1)]
    return X, y


def test_poisson_core_separates_cluster_from_uniform_noise():
    """Verify poisson core separates cluster from uniform noise."""
    for seed in range(3):
        X, y = _blob_plus_noise(seed)
        ell = log_density(knn_density(X, 15))
        core, info = poisson_core(ell, 15, 3)
        assert info["ok"], info
        rec = np.mean(~core[y < 0])
        kept = np.mean(core[y >= 0])
        assert rec > 0.7 and kept > 0.99, (rec, kept)


def test_poisson_core_keeps_homogeneous_data():
    """Verify poisson core keeps homogeneous data."""
    X = np.random.default_rng(3).uniform(0, 1, (300, 3))
    core, info = poisson_core(log_density(knn_density(X, 15)), 15, 3)
    assert core.all() and not info["ok"]


def test_poisson_core_degenerate_inputs():
    """Verify poisson core degenerate inputs."""
    core, info = poisson_core(np.zeros(5), 3, 2)
    assert core.all() and info["reason"] == "degenerate"
    core, info = poisson_core(np.ones(50), 5, 2)
    assert core.all()


def test_poisson_threshold_adapts_to_background_intensity():
    """Verify poisson threshold adapts to background intensity."""
    r = np.random.default_rng(5)
    cl = r.normal(0, 0.3, (150, 3))
    res = []
    for n_noise in (100, 400):
        X = np.vstack([cl, np.random.default_rng(6).uniform(-3, 3, (n_noise, 3))])
        core, info = poisson_core(log_density(knn_density(X, 15)), 15, 3)
        assert info["ok"]
        res.append(info["c"])
    assert res[1] - res[0] == pytest.approx(np.log(4.0)/3, abs=0.12)


def test_halo_core_never_deletes_whole_cluster():
    """Verify halo core never deletes whole cluster."""

    class Cfg:  # minimal config surface
        """Provide the halo settings required by this isolated test."""
        halo_mode = "poisson"
        halo_dim = None
        halo_alpha = 0.01
        halo_kappa = 2.5
        halo_contrast = 2.0
        halo_eta = 0.1
        min_cluster_size = 10_000
    X, _ = _blob_plus_noise(0)
    base = knn_density(X, 15)
    core = halo_core(X, log_density(base), base, Cfg, 15)
    assert core.all()


def test_fixed_mode_matches_range_fraction_and_eta_monotone():
    """Verify fixed mode matches range fraction and eta monotone."""
    d = np.linspace(1.0, 11.0, 11)
    assert fixed_core(d, 0.0).all()
    assert fixed_core(d, 0.35).sum() < fixed_core(d, 0.05).sum()
    assert fixed_core(np.ones(4), 0.5).all()


def test_pipeline_halo_labels_field_as_noise_and_modes():
    """Verify pipeline halo labels field as noise and modes."""
    X, _ = make_blobs(n_per=60, std=0.3)
    rng = np.random.default_rng(0)
    field = rng.uniform(-8, 28, size=(100, 3))
    Xa = np.vstack([X, field])
    fidx = np.arange(len(X), len(Xa))
    cfg = EntrapConfig.from_params(15, 0.1, {"B": 6, "P": 29, "pi_runs": 4, "stability_B": 6,
                                             "pi_full_stability": False, "seed": 0,
                                             "use_leiden": False, "multi_k": False}).resolve(len(Xa))
    g = build_knn_graph(Xa, cfg.k_resolved)
    root = build_tree(g, Xa, np.arange(len(Xa)), cfg, None, np.random.default_rng(0))
    labels, basins, _ = extract_clusters(root, len(Xa), halo_noise=True)
    assert np.sum(labels[fidx] < 0) > 50 and len(basins) >= 3
    labels_off, _, _ = extract_clusters(root, len(Xa), halo_noise=False)
    assert np.all(labels_off >= 0)


def test_pipeline_fixed_eta_controls_core_fraction():
    """Verify pipeline fixed eta controls core fraction."""
    X, _ = make_blobs(n_per=50, std=0.3)
    g = build_knn_graph(X, 12)
    sizes = {}
    for eta in (0.05, 0.4):
        cfg = EntrapConfig.from_params(10, 0.1, {"B": 4, "P": 19, "use_leiden": False, "pi_full_stability": False,
                                                 "halo_mode": "fixed", "halo_eta": eta, "multi_k": False}).resolve(len(X))
        root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
        _, basins, _ = extract_clusters(root, len(X), halo_noise=True)
        sizes[eta] = sum(len(b) for b in basins.values())
    assert sizes[0.4] <= sizes[0.05]


def test_background_false_core_rate_is_controlled_by_alpha():
    """Verify background false core rate is controlled by alpha."""
    r = np.random.default_rng(21)
    X = np.vstack([r.normal(0, 0.3, (200, 3)), r.uniform(-6, 6, (2500, 3))])
    ell = log_density(knn_density(X, 15))
    core, info = poisson_core(ell, 15, 3, alpha=0.001)
    assert info["ok"]
    far = np.r_[np.zeros(200, bool), np.linalg.norm(X[200:], axis=1) > 2.0]
    assert np.mean(core[far]) < 0.02


def test_weak_density_contrast_is_rejected_by_contrast_guard():
    """Verify weak density contrast is rejected by contrast guard."""
    r = np.random.default_rng(0)
    A = r.uniform(0, 1, (400, 3))
    B = np.c_[r.uniform(1, 2, 1200), r.uniform(0, 1, (1200, 2))]
    ell = log_density(knn_density(np.vstack([A, B]), 15))
    core, info = poisson_core(ell, 15, 3)
    assert core.all() and not info["ok"] and info["reason"] == "no density contrast"
