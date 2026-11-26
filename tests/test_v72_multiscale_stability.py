import numpy as np
from sklearn.datasets import make_moons

from entrap.config import EntrapConfig
from entrap.graph import build_knn_graph
from entrap.tree import build_tree


def test_two_moons_keeps_topological_split_with_multiscale_subsample_stability():
    """Verify two moons keeps topological split with multiscale subsample stability."""
    X, _ = make_moons(n_samples=400, noise=0.07, random_state=0)
    cfg = EntrapConfig.from_params(
        15, 0.05,
        {
            "P": 199,
            "B": 5,
            "stability_B": 5,
            "pi_runs": 1,
            "pi_full_stability": False,
            "compute_pi": False,
            "use_leiden": False,
            "mutual_reachability": True,
            "n_jobs": 1,
        },
    ).resolve(len(X))
    graph = build_knn_graph(X, cfg.k_resolved, n_jobs=1)
    root = build_tree(
        graph, X, np.arange(len(X)), cfg, None,
        np.random.default_rng(0), internal_stability=True,
    )

    assert root.m_star >= 2
    assert root.selected_k in cfg.k_scales_resolved
    assert root.stability_selected_k.shape == (cfg.stability_B,)
    assert root.stability_mode_counts.shape == (cfg.stability_B,)
    assert np.all(root.stability_mode_counts >= 2)
    assert np.unique(root.stability_selected_k).size >= 2
    assert np.all(root.stability_adjusted_pvalues <= cfg.significance + 1e-12)
    assert len(root.children) == 2
    assert sorted(len(child.indices) for child in root.children) == [200, 200]


def test_multiscale_stability_diagnostics_report_each_subsample_scale():
    """Verify multiscale stability diagnostics report each subsample scale."""
    X, _ = make_moons(n_samples=240, noise=0.07, random_state=11)
    cfg = EntrapConfig.from_params(
        10, 0.1,
        {
            "P": 99,
            "B": 3,
            "stability_B": 3,
            "pi_runs": 1,
            "pi_full_stability": False,
            "compute_pi": False,
            "use_leiden": False,
            "mutual_reachability": True,
            "n_jobs": 1,
        },
    ).resolve(len(X))
    graph = build_knn_graph(X, cfg.k_resolved, n_jobs=1)
    root = build_tree(
        graph, X, np.arange(len(X)), cfg, None,
        np.random.default_rng(123), internal_stability=True,
    )
    assert root.stability_selected_k is not None
    assert len(root.stability_selected_k) == 3
    assert np.all(np.isin(root.stability_selected_k, cfg.k_scales_resolved))
    assert root.stability_adjusted_pvalues.shape == (3,)
