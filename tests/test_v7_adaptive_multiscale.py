import numpy as np

from entrap.config import EntrapConfig
from entrap.graph import (adaptive_k_from_distances,
                          adaptive_knn_density_from_distances,
                          build_knn_graph, knn_scale_subgraph)
from entrap.significance import poisson_null_multiscale_tau
from entrap.tree import build_tree
from scipy.special import gamma


def test_adaptive_k_detects_a_persistent_neighbour_distance_knee():
    """Verify adaptive k detects a persistent neighbour distance knee."""
    k = np.arange(1, 25, dtype=np.float64)
    smooth = np.tile(np.sqrt(k), (8, 1))
    broken = smooth.copy()
    broken[:, 10:] *= 3.0
    ks_smooth = adaptive_k_from_distances(smooth, 4, 2.5, 4, 2)
    ks_broken = adaptive_k_from_distances(broken, 4, 2.5, 4, 2)
    assert np.all(ks_smooth == 24)
    assert np.all((ks_broken >= 4) & (ks_broken < 24))


def test_adaptive_density_returns_per_point_k_and_finite_density():
    """Verify adaptive density returns per point k and finite density."""
    rng = np.random.default_rng(71)
    X = np.vstack([rng.normal(0, 0.15, (80, 2)), rng.uniform(-3, 3, (120, 2))])
    g = build_knn_graph(X, 20)
    density, local_k = adaptive_knn_density_from_distances(
        g.knn_dist, 4, 2.5, 4, 2)
    assert density.shape == (len(X),)
    assert local_k.shape == (len(X),)
    assert np.all(np.isfinite(density)) and np.all(density > 0)
    assert np.all((local_k >= 4) & (local_k <= 20))
    # A heterogeneous mixture should not be forced to one universal K.
    assert np.unique(local_k).size > 1


def test_scale_graphs_are_nested_and_use_cached_knn():
    """Verify scale graphs are nested and use cached knn."""
    X = np.random.default_rng(4).normal(size=(100, 3))
    g = build_knn_graph(X, 16)
    small = knn_scale_subgraph(g, 5)
    large = knn_scale_subgraph(g, 12)
    assert set(map(tuple, small.edges)).issubset(set(map(tuple, large.edges)))
    assert small.adj_indptr[-1] == 2 * len(small.edges)
    assert large.adj_indptr[-1] == 2 * len(large.edges)


def test_multiscale_poisson_null_reproducible_and_familywise_thresholds():
    """Verify multiscale poisson null reproducible and familywise thresholds."""
    X = np.random.default_rng(9).uniform(0, 1, (100, 2))
    scales = (4, 7, 11)
    a, na = poisson_null_multiscale_tau(
        X, scales, 29, 0.1, np.random.default_rng(3),
        adaptive_k=True, adaptive_k_min=4, adaptive_k_max=15,
        n_jobs=1)
    b, nb = poisson_null_multiscale_tau(
        X, scales, 29, 0.1, np.random.default_rng(3),
        adaptive_k=True, adaptive_k_min=4, adaptive_k_max=15,
        n_jobs=1)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(na, nb)
    assert a.shape == (3,) and na.shape == (29, 3)
    # The family-wise threshold for each scale is its (1-alpha/M) null quantile.
    np.testing.assert_allclose(a, np.quantile(na, 1 - 0.1 / 3, axis=0))


def test_tree_exposes_selected_scale_and_adaptive_k_diagnostics(blobs):
    """Verify tree exposes selected scale and adaptive k diagnostics."""
    X, _ = blobs
    cfg = EntrapConfig.from_params(
        10, 0.1,
        {"B": 3, "P": 29, "stability_B": 3, "pi_runs": 2,
         "pi_full_stability": False, "use_leiden": False, "multi_k": True},
    ).resolve(len(X))
    graph = build_knn_graph(X, cfg.k_resolved)
    node = build_tree(graph, X, np.arange(len(X)), cfg, None,
                      np.random.default_rng(22), internal_stability=False)
    assert node.selected_k in node.k_scales
    assert node.k_scores.shape == (len(node.k_scales),)
    assert node.local_k_summary["adaptive"] is True
    assert node.k_pvalues is not None
    assert node.k_prominence_pvalues is not None
    assert node.k_component_pvalues is not None
    assert node.component_counts.shape == (len(node.k_scales),)
    assert node.component_thresholds.shape == (len(node.k_scales),)
    assert 0.0 <= node.k_adjusted_p <= 1.0


def test_component_count_null_is_calibrated_with_same_two_statistic_family():
    """Verify component count null is calibrated with same two statistic family."""
    X = np.random.default_rng(17).uniform(0, 1, (90, 2))
    alpha = 0.1
    scales = (4, 8)
    tau, maxima, c_tau, components = poisson_null_multiscale_tau(
        X, scales, 39, alpha, np.random.default_rng(11),
        adaptive_k=True, adaptive_k_min=4, adaptive_k_max=12,
        n_jobs=1, include_components=True,
    )
    level = 1.0 - alpha / (2.0 * len(scales))
    np.testing.assert_allclose(tau, np.quantile(maxima, level, axis=0))
    np.testing.assert_allclose(c_tau, np.quantile(components, level, axis=0))
    assert maxima.shape == components.shape == (39, 2)
    assert np.all(components >= 1)


def test_explicit_graph_scales_can_exceed_automatic_k_heuristic():
    """Verify explicit graph scales can exceed automatic k heuristic."""
    cfg = EntrapConfig.from_params(
        10, 0.05, {"k_scales": (5, 8, 12, 18, 24)}
    ).resolve(1000)
    assert cfg.k_scales_resolved == (5, 8, 12, 18, 24)
    assert cfg.k_resolved == 24
    assert cfg.adaptive_k_max_resolved == 24

    capped = EntrapConfig.from_params(
        10, 0.05, {"k": 12, "k_scales": (5, 8, 12, 18, 24)}
    ).resolve(1000)
    assert capped.k_scales_resolved == (5, 8, 12)
    assert capped.k_resolved == 12


def test_adaptive_mean_distance_density_corrects_k_dependent_poisson_scale():
    # Two synthetic distance profiles force different adaptive K values. The
    # expected homogeneous-Poisson mean-radius factor is A(k,d).
    """Verify adaptive mean distance density corrects k dependent poisson scale."""
    base = np.sqrt(np.arange(1, 25, dtype=np.float64))
    d = np.tile(base, (12, 1))
    d[6:, 10:] *= 3.0
    density, ks = adaptive_knn_density_from_distances(
        d, 4, 2.5, 4, 2, normalized=False, d_hat=2.0,
        k_bias_correction=True,
    )
    raw, ks_raw = adaptive_knn_density_from_distances(
        d, 4, 2.5, 4, 2, normalized=False, d_hat=2.0,
        k_bias_correction=False,
    )
    np.testing.assert_array_equal(ks, ks_raw)
    rows = np.arange(len(d))
    mean_r = np.cumsum(d, axis=1)[rows, ks - 1] / ks
    j = np.arange(1, d.shape[1] + 1, dtype=np.float64)
    moments = gamma(j + 0.5) / gamma(j)
    a_k = np.cumsum(moments)[ks - 1] / ks
    np.testing.assert_allclose(density, a_k / mean_r)
    np.testing.assert_allclose(raw, 1.0 / mean_r)
    assert np.unique(ks).size > 1


def test_adaptive_normalized_density_uses_inverse_gamma_k_minus_one_factor():
    """Verify adaptive normalized density uses inverse gamma k minus one factor."""
    d = np.tile(np.sqrt(np.arange(1, 25, dtype=np.float64)), (12, 1))
    d[6:, 10:] *= 3.0
    density, ks = adaptive_knn_density_from_distances(
        d, 4, 2.5, 4, 2, normalized=True, d_hat=2.0,
        k_bias_correction=True,
    )
    rows = np.arange(len(d))
    rk = d[rows, ks - 1]
    expected = (ks - 1) / (len(d) * np.pi * rk**2)
    np.testing.assert_allclose(density, expected)


def test_homogeneous_poisson_field_is_not_accepted_as_single_cluster():
    """Verify homogeneous poisson field is not accepted as single cluster."""
    from entrap.tree import extract_clusters

    X = np.random.default_rng(20261124).uniform(0.0, 1.0, (250, 2))
    cfg = EntrapConfig.from_params(
        10, 0.05,
        {"P": 199, "B": 2, "stability_B": 0, "pi_runs": 1,
         "compute_pi": False, "multi_k": True, "adaptive_k": True,
         "use_leiden": False},
    ).resolve(len(X))
    graph = build_knn_graph(X, cfg.k_resolved)
    root = build_tree(graph, X, np.arange(len(X)), cfg, None,
                      np.random.default_rng(41), internal_stability=False)
    labels, basins, _ = extract_clusters(root, len(X), halo_noise=True)

    assert root.k_adjusted_p is not None
    assert root.k_adjusted_p > cfg.significance
    assert root.is_leaf and not root.is_cluster
    assert len(basins) == 0
    assert np.all(labels == -1)
