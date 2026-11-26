"""Poisson-simulated null threshold tau: determinism, invariances, dimension dependence, calibration."""
import numpy as np
import pytest
from entrap.config import EntrapConfig
from entrap.graph import build_knn_graph
from entrap.filtration import build_filtration, filtration_merge_tree
from entrap.significance import poisson_null_tau
from entrap.tree import build_tree, extract_clusters
from helpers import make_blobs


def _obs_max_prom(X, k, mreach=False):
    """Calculate observed maximum persistence for the test fixture."""
    g = build_knn_graph(X, k)
    fil, _ = build_filtration(X, g.edges, k, mreach=mreach)
    p = filtration_merge_tree(fil, g.adj).mode_prom
    return float(p[np.isfinite(p)].max())


def test_poisson_tau_deterministic_and_shapes():
    """Verify poisson tau deterministic and shapes."""
    X = np.random.default_rng(0).uniform(0, 1, (120, 2))
    t1, m1 = poisson_null_tau(X, 10, 19, 0.1, np.random.default_rng(5))
    t2, m2 = poisson_null_tau(X, 10, 19, 0.1, np.random.default_rng(5))
    assert t1 == t2 and np.array_equal(m1, m2) and m1.shape == (19,) and np.isfinite(t1) and t1 >= 0
    assert t1 == pytest.approx(np.quantile(m1, 0.9))


@pytest.mark.parametrize("mreach", [False, True])
def test_poisson_tau_invariant_to_affine_rescaling_of_the_node(mreach):
    """Verify poisson tau invariant to affine rescaling of the node."""
    X = np.random.default_rng(1).uniform(0, 1, (100, 3))
    a, _ = poisson_null_tau(X, 10, 15, 0.1, np.random.default_rng(2), mreach=mreach)
    b, _ = poisson_null_tau(37.0 * X + 5.0, 10, 15, 0.1, np.random.default_rng(2), mreach=mreach)
    assert a == pytest.approx(b, abs=1e-9)


def test_poisson_tau_uses_bounding_box_aspect_ratio():
    """Verify poisson tau uses bounding box aspect ratio."""
    r = np.random.default_rng(3)
    sq = r.uniform(0, 1, (200, 2))
    thin = sq * np.array([1.0, 0.01])
    t_sq, _ = poisson_null_tau(sq, 10, 29, 0.05, np.random.default_rng(4))
    t_thin, _ = poisson_null_tau(thin, 10, 29, 0.05, np.random.default_rng(4))
    assert t_thin > 1.5 * t_sq


def test_poisson_tau_decreases_with_dimension():
    """Verify poisson tau decreases with dimension."""
    ts = []
    for d in (1, 2, 3):
        X = np.random.default_rng(0).uniform(0, 1, (200, d))
        ts.append(poisson_null_tau(X, 10, 39, 0.05, np.random.default_rng(1))[0])
    assert ts[0] > ts[1] > ts[2]


def test_dim_override_equals_native_low_dimensional_simulation():
    """Verify dim override equals native low dimensional simulation."""
    X2 = np.random.default_rng(0).uniform(0, 1, (120, 2))
    X1 = np.random.default_rng(9).uniform(0, 1, (120, 1))
    a, _ = poisson_null_tau(X2, 10, 15, 0.1, np.random.default_rng(6), dim=1)
    b, _ = poisson_null_tau(X1, 10, 15, 0.1, np.random.default_rng(6))
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.slow
def test_false_positive_rate_on_homogeneous_data_is_near_nominal():
    """Verify false positive rate on homogeneous data is near nominal."""
    fp = 0
    N = 30
    for s in range(N):
        X = np.random.default_rng(100 + s).uniform(0, 1, (150, 2))
        tau, _ = poisson_null_tau(X, 10, 49, 0.05, np.random.default_rng(s))
        fp += _obs_max_prom(X, 10) > tau
    assert fp / N <= 0.20, fp / N


def test_planted_structure_exceeds_tau_while_noise_does_not():
    # Two CONNECTED blobs (separation 5 sd): a disconnected kNN graph has no finite prominence at all.
    """Verify planted structure exceeds tau while noise does not."""
    r = np.random.default_rng(0)
    Xb = np.vstack([r.normal([0, 0], 0.4, (100, 2)), r.normal([2.0, 0], 0.4, (100, 2))])
    tau_b, _ = poisson_null_tau(Xb, 10, 39, 0.05, np.random.default_rng(0))
    assert _obs_max_prom(Xb, 10) > 1.3 * tau_b   # measured: 1.04-1.44 vs tau ~0.62 over 4 seeds
    Xu = np.random.default_rng(7).uniform(0, 1, (180, 2))
    tau_u, _ = poisson_null_tau(Xu, 10, 39, 0.05, np.random.default_rng(0))
    assert _obs_max_prom(Xu, 10) <= tau_u


def _cfg(n, **adv):
    """Build the Leiden configuration used by the test."""
    base = {"B": 4, "P": 29, "pi_runs": 3, "stability_B": 6, "pi_full_stability": False,
            "seed": 0, "use_leiden": False, "multi_k": False}
    base.update(adv)
    return EntrapConfig.from_params(10, 0.1, base).resolve(n)


@pytest.mark.parametrize("null", ["poisson", "permutation"])
def test_pipeline_three_blobs_for_each_null(null):
    """Verify pipeline three blobs for each null."""
    X, y = make_blobs(n_per=50, std=0.3)
    cfg = _cfg(len(X), tau_null=null)
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    _, basins, _ = extract_clusters(root, len(X), halo_noise=True)
    assert len(basins) == 3


def test_pipeline_homogeneous_square_is_one_cluster_with_poisson_tau():
    """Verify pipeline homogeneous square is one cluster with poisson tau."""
    X = np.random.default_rng(11).uniform(0, 1, (200, 2))
    cfg = _cfg(len(X))
    g = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(g, X, np.arange(len(X)), cfg, None, np.random.default_rng(0))
    assert root.m_star <= 1


def test_config_tau_null_validation():
    """Verify config tau null validation."""
    c = EntrapConfig()
    assert c.tau_null == "poisson" and c.tau_null_dim is None and c.mutual_reachability is False
    for bad in ({"tau_null": "bootstrap"}, {"tau_null_dim": 0}):
        with pytest.raises(ValueError):
            EntrapConfig.from_params(10, 0.1, bad)


def test_poisson_tau_follows_the_filtration_it_is_used_with():
    """Verify poisson tau follows the filtration it is used with."""
    X = np.random.default_rng(0).uniform(0, 1, (120, 2))
    a, ma = poisson_null_tau(X, 10, 19, 0.1, np.random.default_rng(5), mreach=False)
    b, mb = poisson_null_tau(X, 10, 19, 0.1, np.random.default_rng(5), mreach=True)
    assert b > a + 0.05 and np.all(mb >= ma - 1e-12)   # measured: 0.476 vs 0.598
