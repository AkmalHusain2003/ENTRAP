"""Build and extract the recursive multi-scale clustering hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .graph import (build_knn_graph, knn_scale_subgraph,
                    adaptive_knn_density_from_distances,
                    estimate_intrinsic_dim, log_density)
from .filtration import build_filtration_from_density, filtration_merge_tree
from .halo import halo_core
from .mergetree import MergeTree
from .significance import (compute_tau, mode_count_stability,
                           multiscale_mode_count_stability,
                           select_multiscale_mode_count,
                           _mode_count_from_selected_scale,
                           poisson_null_multiscale_tau)
from .leiden import leiden_partition
from .diagnostics import persistence_entropy
from . import topology


@dataclass
class TreeNode:
    """Store topology, significance, partition, and membership data for one node."""
    indices: np.ndarray
    depth: int
    children: list = field(default_factory=list)
    is_leaf: bool = False
    is_cluster: bool = False
    m_star: int = 0
    tau: float = 0.0
    entropy: float = 0.0
    merge_tree: MergeTree | None = None
    partition: np.ndarray | None = None
    leiden_info: dict | None = None
    cluster_id: int = -1
    prominence: float = 0.0
    core: np.ndarray | None = None
    selected_k: int = 0
    k_scales: tuple = ()
    k_scores: np.ndarray | None = None
    k_pvalues: np.ndarray | None = None
    k_prominence_pvalues: np.ndarray | None = None
    k_component_pvalues: np.ndarray | None = None
    component_counts: np.ndarray | None = None
    component_thresholds: np.ndarray | None = None
    k_adjusted_p: float | None = None
    local_k_summary: dict = field(default_factory=dict)
    stability_selected_k: np.ndarray | None = None
    stability_adjusted_pvalues: np.ndarray | None = None
    stability_mode_counts: np.ndarray | None = None
    stability_skipped_reason: str | None = None


def _node_graph(graph, X_node, S, cfg):
    """Build a node-local graph and retain a maximum-K neighbour cache."""
    n = len(S)
    needed = max(max(cfg.k_scales_resolved), cfg.adaptive_k_max_resolved)
    is_full = (graph is not None and graph.N == n and
               len(S) == n and np.array_equal(S, np.arange(n, dtype=np.int64)))
    if (is_full and graph.knn_idx is not None and graph.knn_dist is not None
            and graph.knn_idx.shape[1] >= needed):
        return graph
    return build_knn_graph(X_node, min(needed, n - 1), cfg.metric,
                           n_jobs=cfg.n_jobs, store_knn=True)


def _node_density(X_node, local_graph, cfg):
    """Estimate node-local adaptive density and fixed-K halo density."""
    kd = np.ascontiguousarray(local_graph.knn_dist, dtype=np.float64)
    kmax = min(cfg.adaptive_k_max_resolved, kd.shape[1], len(X_node) - 1)
    # A local intrinsic-dimension estimate is also needed to normalize the
    # Poisson expectation of inverse-mean-radius density when K varies by point.
    d_hat = (
        estimate_intrinsic_dim(X_node, workers=cfg.n_jobs)
        if (cfg.normalized_density or cfg.adaptive_k)
        else None
    )
    kmin = min(cfg.adaptive_k_min_resolved, kmax)
    if cfg.adaptive_k:
        base, local_ks = adaptive_knn_density_from_distances(
            kd[:, :kmax], kmin, cfg.adaptive_k_sensitivity,
            cfg.adaptive_k_window, cfg.adaptive_k_patience,
            normalized=cfg.normalized_density, d_hat=d_hat,
        )
    else:
        base, local_ks = adaptive_knn_density_from_distances(
            kd[:, :kmax], kmax, cfg.adaptive_k_sensitivity,
            cfg.adaptive_k_window, cfg.adaptive_k_patience,
            normalized=cfg.normalized_density, d_hat=d_hat,
            k_bias_correction=False,
        )
    # Halo's homogeneous-Poisson reference assumes one fixed neighbour count.
    # Precompute that separate scale now so the max-K cache can be released after
    # constructing the scale family (including for non-leaf recursive nodes).
    kh = min(cfg.k_resolved, kd.shape[1], len(X_node) - 1)
    fixed_mean_radius = kd[:, :kh].mean(axis=1)
    ell_halo = log_density(1.0 / np.maximum(fixed_mean_radius, 1e-12))
    return base, local_ks, d_hat, ell_halo, kh


def _max_finite_prominence(mt):
    """Return the maximum finite persistence prominence in a merge tree."""
    finite = mt.mode_prom[np.isfinite(mt.mode_prom)]
    return float(finite.max()) if len(finite) else 0.0


def _scale_null_thresholds(X_node, scales, density, graphs, fils, cfg, rng, d_hat):
    """Calibrate prominence and component thresholds across candidate graph scales."""
    m = len(scales)
    if cfg.tau_null == "poisson":
        return poisson_null_multiscale_tau(
            X_node, scales, cfg.P, cfg.significance, rng,
            adaptive_k=cfg.adaptive_k,
            adaptive_k_min=cfg.adaptive_k_min_resolved,
            adaptive_k_sensitivity=cfg.adaptive_k_sensitivity,
            adaptive_k_window=cfg.adaptive_k_window,
            adaptive_k_patience=cfg.adaptive_k_patience,
            adaptive_k_max=cfg.adaptive_k_max_resolved,
            d_hat=d_hat, normalized=cfg.normalized_density,
            mreach=cfg.mutual_reachability,
            dim=cfg.tau_null_dim, n_jobs=cfg.n_jobs,
            parallel_max_workers=cfg.parallel_max_workers,
            parallel_memory_mb=cfg.parallel_memory_mb,
            include_components=True,
        )

    # Block-permutation null: each graph scale gets alpha/M. Components are
    # conditioned on and therefore are not an independent statistic here.
    taus = np.empty(m, dtype=np.float64)
    for j, (sub, fil) in enumerate(zip(graphs, fils)):
        tau, _ = compute_tau(
            density=fil.ell, adj=sub.adj_csr, P=cfg.P,
            alpha=cfg.significance / m, rng=rng,
            ensemble=cfg.block_ensemble, r_auto=cfg.block_scale,
            edges=fil.edges if fil.mreach else None, resid=fil.resid,
            n_jobs=cfg.n_jobs,
            parallel_max_workers=cfg.parallel_max_workers,
            parallel_memory_mb=cfg.parallel_memory_mb,
        )
        taus[j] = tau
    return taus, None, None, None


def build_tree(graph, X, S, cfg, tracker, rng, depth=0, internal_stability=True,
               inherited_cluster_support=False):
    """Recursively build the ENTRAP hierarchy from topological and statistical evidence."""
    S = np.ascontiguousarray(S, dtype=np.int64)
    X_node = np.ascontiguousarray(X[S], dtype=np.float64)
    node = TreeNode(indices=S, depth=depth)
    n = len(S)
    if n < cfg.min_cluster_size:
        node.is_leaf = True
        return node

    local_graph = _node_graph(graph, X_node, S, cfg)
    base, local_ks, d_hat, ell_halo, halo_k = _node_density(X_node, local_graph, cfg)
    scales = tuple(sorted({min(int(k), n - 1) for k in cfg.k_scales_resolved if int(k) >= 2}))
    if not scales:
        node.is_leaf = True
        return node

    fused_native = bool(
        cfg.tau_null == "poisson" and topology.USING_CYTHON
    )
    graphs, fils, trees = [], [], []
    if fused_native:
        # The Cython kernel evaluates all graph K scales in one call. It returns
        # the exact per-scale max persistence and connected-component counts,
        # the complete filtration/merge tree is then built only for the chosen K.
        ell_node = log_density(base)
        observed, observed_components = topology.multiscale_graph_statistics(
            X_node, local_graph.knn_idx, ell_node,
            np.asarray(scales, dtype=np.int64),
            mreach=cfg.mutual_reachability,
        )
        null_density = ell_node
    else:
        observed = np.empty(len(scales), dtype=np.float64)
        observed_components = np.empty(len(scales), dtype=np.int64)
        for j, k in enumerate(scales):
            sub_j = knn_scale_subgraph(local_graph, k)
            fil_j, _ = build_filtration_from_density(X_node, sub_j.edges, base,
                                                     cfg.mutual_reachability)
            mt_j = filtration_merge_tree(fil_j, sub_j.adj_csr)
            graphs.append(sub_j)
            fils.append(fil_j)
            trees.append(mt_j)
            observed[j] = _max_finite_prominence(mt_j)
            observed_components[j] = len(np.unique(topology.distance_blocks(
                sub_j.adj_csr[0], sub_j.adj_csr[1], n)))
        null_density = fils[0].ell

    taus, null_maxima, component_thresholds, null_component_counts = _scale_null_thresholds(
        X_node, scales, null_density, graphs, fils, cfg, rng, d_hat
    )
    prominence_scores = observed / np.maximum(taus, 1e-12)
    if null_maxima is not None and null_component_counts is not None:
        # The exact same selection rule is used later inside every stability
        # subsample, only the null sample size/bounds differ appropriately.
        selection = select_multiscale_mode_count(
            observed, observed_components, null_maxima,
            null_component_counts, taus, component_thresholds,
            cfg.significance,
        )
        p_prom = selection["p_prominence"]
        p_comp = selection["p_components"]
        p_scale = selection["p_scale"]
        adjusted_p = selection["adjusted_p"]
        component_scores = selection["component_score"]
        scores = selection["scores"]
        best = int(selection["best"])
    else:
        p_prom = None
        p_comp = None
        p_scale = None
        adjusted_p = None
        component_scores = np.zeros_like(prominence_scores)
        scores = prominence_scores
        best = int(np.argmax(scores))
    k_node = int(scales[best])
    tau = float(taus[best])
    if fused_native:
        sub = knn_scale_subgraph(local_graph, k_node)
        fil, _ = build_filtration_from_density(
            X_node, sub.edges, base, mreach=cfg.mutual_reachability)
        mt = filtration_merge_tree(fil, sub.adj_csr)
    else:
        sub = graphs[best]
        fil = fils[best]
        mt = trees[best]
    # Only the selected graph and merge tree are needed from here onward. Drop
    # the unselected K-scale temporaries and the local max-K cache before any
    # recursive calls, preventing memory from accumulating down the hierarchy.
    del graphs, fils, trees
    local_graph.knn_idx = None
    local_graph.knn_dist = None
    del fil

    node.selected_k = k_node
    node.k_scales = scales
    node.k_scores = scores
    node.k_prominence_pvalues = p_prom
    node.k_component_pvalues = p_comp
    node.component_counts = observed_components
    node.component_thresholds = component_thresholds
    node.k_pvalues = p_scale
    node.k_adjusted_p = adjusted_p
    node.tau = tau
    node.merge_tree = mt
    node.entropy = persistence_entropy(mt.pairs)
    node.local_k_summary = {
        "min": int(np.min(local_ks)),
        "median": float(np.median(local_ks)),
        "p90": float(np.quantile(local_ks, 0.90)),
        "max": int(np.max(local_ks)),
        "adaptive": bool(cfg.adaptive_k),
    }
    # Compute the full-data topological ceiling before running subsample
    # stability. A node with no supported mode can skip stability entirely.
    # A node with one supported mode still needs stability before it can be
    # accepted as a single cluster, but only the selected K is needed for that
    # yes/no decision. Nodes with >=2 supported modes run the multi-K
    # stability procedure so a split visible at another feasible scale is not
    # erased by locking the bootstrap to the full-data winning K.
    if cfg.tau_null == "poisson":
        topological_count = _mode_count_from_selected_scale(
            mt, selection, taus, cfg.significance, len(scales)
        )
        if topological_count <= 0:
            mstar = 0
            node.stability_skipped_reason = "full_data_topology_supports_no_modes"
        elif internal_stability and cfg.stability_B > 0 and topological_count >= 2:
            mstar, stability_counts, stability_info = multiscale_mode_count_stability(
                X_node, scales, cfg.stability_B, cfg.f, cfg.significance, rng,
                null_maxima=null_maxima, tau_thresholds=taus,
                p_component_reference=p_comp,
                component_thresholds=component_thresholds,
                adaptive_k=cfg.adaptive_k,
                adaptive_k_min=cfg.adaptive_k_min_resolved,
                adaptive_k_max=cfg.adaptive_k_max_resolved,
                adaptive_k_sensitivity=cfg.adaptive_k_sensitivity,
                adaptive_k_window=cfg.adaptive_k_window,
                adaptive_k_patience=cfg.adaptive_k_patience,
                normalized=cfg.normalized_density, d_hat=d_hat,
                mreach=cfg.mutual_reachability,
                metric=cfg.metric,
                n_jobs=cfg.n_jobs,
                parallel_max_workers=cfg.parallel_max_workers,
                parallel_memory_mb=cfg.parallel_memory_mb,
                return_diagnostics=True,
            )
            node.stability_selected_k = stability_info["selected_k"]
            node.stability_adjusted_pvalues = stability_info["adjusted_p"]
            node.stability_mode_counts = stability_counts
            mstar = min(int(mstar), int(topological_count))
        elif internal_stability and cfg.stability_B > 0:
            # A one-mode node can only be accepted/rejected as one cluster, use
            # the selected scale for this scalar stability decision. Multi-K
            # reselection is reserved for genuine hierarchical split proposals.
            mstar, stability_counts = mode_count_stability(
                X_node, sub.edges, tau, cfg.stability_B, cfg.f,
                cfg.significance, k_node, rng,
                d_hat=d_hat, normalized=cfg.normalized_density,
                mreach=cfg.mutual_reachability,
                n_jobs=cfg.n_jobs,
                parallel_max_workers=cfg.parallel_max_workers,
                parallel_memory_mb=cfg.parallel_memory_mb,
                adaptive_k=cfg.adaptive_k,
                adaptive_k_min=cfg.adaptive_k_min_resolved,
                adaptive_k_max=cfg.adaptive_k_max_resolved,
                adaptive_k_sensitivity=cfg.adaptive_k_sensitivity,
                adaptive_k_window=cfg.adaptive_k_window,
                adaptive_k_patience=cfg.adaptive_k_patience,
                metric=cfg.metric,
            )
            node.stability_mode_counts = stability_counts
            node.stability_selected_k = np.full(
                len(stability_counts), k_node, dtype=np.int64
            )
            mstar = min(int(mstar), int(topological_count))
        else:
            mstar = min(int(mt.count_significant(tau)), int(topological_count))
            node.stability_skipped_reason = "stability_disabled"
    else:
        if internal_stability and cfg.stability_B > 0:
            mstar, stability_counts = mode_count_stability(
                X_node, sub.edges, tau, cfg.stability_B, cfg.f,
                cfg.significance, k_node, rng,
                d_hat=d_hat, normalized=cfg.normalized_density,
                mreach=cfg.mutual_reachability,
                n_jobs=cfg.n_jobs,
                parallel_max_workers=cfg.parallel_max_workers,
                parallel_memory_mb=cfg.parallel_memory_mb,
                adaptive_k=cfg.adaptive_k,
                adaptive_k_min=cfg.adaptive_k_min_resolved,
                adaptive_k_max=cfg.adaptive_k_max_resolved,
                adaptive_k_sensitivity=cfg.adaptive_k_sensitivity,
                adaptive_k_window=cfg.adaptive_k_window,
                adaptive_k_patience=cfg.adaptive_k_patience,
                metric=cfg.metric,
            )
            node.stability_mode_counts = stability_counts
        else:
            mstar = mt.count_significant(tau)
        mstar = int(min(max(mstar, 0), len(mt.mode_prom)))
    node.m_star = mstar
    if tracker is not None:
        tracker.record(node)

    if mstar <= 1:
        node.is_leaf = True
        # A one-mode hierarchy is not, by itself, evidence for a cluster: a
        # homogeneous field also has one essential H0 component. Under the
        # Poisson null require the family-wise adjusted multiscale test to pass
        # before accepting the node as a cluster. Non-Poisson/permutation mode
        # retains its established decision path because its component-aware
        # adjusted p-value is not currently defined.
        node_significant = (
            cfg.tau_null != "poisson"
            or inherited_cluster_support
            or (node.k_adjusted_p is not None
                and node.k_adjusted_p <= cfg.significance + 1e-12)
        )
        node.is_cluster = bool(
            mstar == 1 and n >= cfg.min_cluster_size and node_significant
        )
        if node.is_cluster:
            node.prominence = _max_finite_prominence(mt)
            # Halo/background discrimination intentionally uses a fixed-K
            # Poisson reference, separate from adaptive-K topology density.
            # This keeps the analytic null assumptions of poisson_core intact.
            node.core = halo_core(X_node, ell_halo, base, cfg, halo_k)
        return node

    prom_sorted = np.sort(mt.mode_prom)[::-1]
    tau_split = prom_sorted[mstar - 1]
    seed = mt.assign(tau_split)
    partition, node.leiden_info = leiden_partition(
        n, sub.edges, sub.weights, mstar, seed, cfg.gamma_range, cfg, rng,
        return_info=True,
    )
    node.partition = partition
    # The selected graph is no longer needed once the partition is available.
    del sub, local_graph
    for comm in np.unique(partition):
        local_idx = np.flatnonzero(partition == comm)
        cs = S[local_idx]
        if len(cs):
            # Child graph is rebuilt on the child's own samples. This removes
            # the parent-induced-subgraph neighbour mismatch at recursive nodes.
            node.children.append(build_tree(
                None, X, cs, cfg, tracker, rng, depth + 1,
                internal_stability,
                inherited_cluster_support=(cfg.tau_null != "poisson" or
                    (node.k_adjusted_p is not None and
                     node.k_adjusted_p <= cfg.significance + 1e-12)),
            ))
    return node


def extract_clusters(root, N, halo_noise=True):
    """Extract accepted core memberships from hierarchical leaf nodes."""
    labels = np.full(N, -1, np.int64)
    basins, proms, cid = {}, {}, 0
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.is_leaf:
            if nd.is_cluster:
                core = (
                    nd.core
                    if (halo_noise and nd.core is not None)
                    else np.ones(len(nd.indices), bool)
                )
                idx_core = nd.indices[core]
                labels[idx_core] = cid
                basins[cid] = idx_core
                proms[cid] = nd.prominence
                nd.cluster_id = cid
                cid += 1
        else:
            stack.extend(nd.children)
    return labels, basins, proms
