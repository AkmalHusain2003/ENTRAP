"""End-to-end parity tests for the exact multi-K H0 engine."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from entrap import ENTRAP, topology


pytestmark = pytest.mark.skipif(
    not topology.USING_CYTHON,
    reason="The optimized multi-K H0 engine requires the compiled Cython extension.",
)


def _tree_signature(node):
    """Return the hierarchy decisions needed to compare reference and optimized fits."""
    return {
        "indices": np.asarray(node.indices).copy(),
        "selected_k": int(node.selected_k),
        "m_star": int(node.m_star),
        "adjusted_p": node.k_adjusted_p,
        "components": None if node.component_counts is None else np.asarray(node.component_counts).copy(),
        "children": [_tree_signature(child) for child in node.children],
    }


def _assert_tree_equal(left, right):
    """Compare recursive hierarchy decisions without relying on runtime diagnostics."""
    assert left["selected_k"] == right["selected_k"]
    assert left["m_star"] == right["m_star"]
    np.testing.assert_array_equal(left["indices"], right["indices"])
    if left["adjusted_p"] is not None or right["adjusted_p"] is not None:
        assert left["adjusted_p"] == pytest.approx(right["adjusted_p"], abs=1e-14)
    if left["components"] is not None:
        np.testing.assert_array_equal(left["components"], right["components"])
    assert len(left["children"]) == len(right["children"])
    for child_left, child_right in zip(left["children"], right["children"]):
        _assert_tree_equal(child_left, child_right)


@pytest.mark.parametrize(
    "dataset,mutual_reachability,min_cluster_size",
    [
        ("blobs", False, 10),
        ("moons", True, 10),
    ],
)
def test_fit_matches_original_multi_k_kernel(
    monkeypatch, dataset, mutual_reachability, min_cluster_size
):
    """Verify labels and hierarchy are unchanged by the shared-edge H0 engine."""
    if dataset == "blobs":
        X, _ = make_blobs(
            n_samples=180,
            centers=3,
            n_features=3,
            random_state=7,
        )
    else:
        X, _ = make_moons(
            n_samples=240,
            noise=0.07,
            random_state=11,
        )

    advanced = {
        "P": 99,
        "B": 3,
        "stability_B": 3,
        "pi_runs": 1,
        "pi_full_stability": False,
        "compute_pi": False,
        "use_leiden": False,
        "n_jobs": 1,
        "parallel_max_workers": 1,
        "seed": 42,
        "halo_noise": False,
        "mutual_reachability": mutual_reachability,
    }

    optimized_function = topology.multiscale_graph_statistics
    monkeypatch.setattr(
        topology,
        "multiscale_graph_statistics",
        topology.multiscale_graph_statistics_reference,
    )
    reference = ENTRAP(
        min_cluster_size,
        0.1,
        advanced,
    ).fit(X)

    monkeypatch.setattr(
        topology,
        "multiscale_graph_statistics",
        optimized_function,
    )
    optimized = ENTRAP(
        min_cluster_size,
        0.1,
        advanced,
    ).fit(X)

    np.testing.assert_array_equal(optimized.labels_, reference.labels_)
    np.testing.assert_array_equal(optimized.core_labels_, reference.core_labels_)
    np.testing.assert_array_equal(
        optimized.result_.probabilities,
        reference.result_.probabilities,
    )
    np.testing.assert_array_equal(
        optimized.result_.pi_per_cluster,
        reference.result_.pi_per_cluster,
    )
    assert optimized.result_.n_clusters == reference.result_.n_clusters
    _assert_tree_equal(
        _tree_signature(optimized.result_.tree),
        _tree_signature(reference.result_.tree),
    )
