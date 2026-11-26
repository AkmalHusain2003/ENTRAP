"""Calibrate persistence and component statistics with null simulations."""

from __future__ import annotations
import math
import numpy as np
from scipy.special import gamma as _gamma
from . import topology
from .graph import (adj_to_csr, induced_edges, _query_knn_euclidean,
                    _edges_and_distances_from_knn, log_density,
                    estimate_intrinsic_dim, adaptive_knn_density_from_distances,
                    build_knn_graph, knn_scale_subgraph)
from .filtration import (build_filtration, build_filtration_from_density,
                         filtration_count, filtration_merge_tree)
from .parallel import resolve_workers, ordered_thread_map
try:
    from . import _null as _null_native
    _HAS_NULL_NATIVE = True
except Exception:
    _null_native = None
    _HAS_NULL_NATIVE = False


def _csr_arrays(adj):
    """Normalize adjacency input into contiguous CSR arrays."""
    return adj_to_csr(adj)


def _estimate_block_scale_python(density, adj, max_hop=6, n_samples=100, rng=None):
    """Estimate graph correlation length with a Python CSR traversal."""
    rng = rng or np.random.default_rng(0)
    density = np.asarray(density, dtype=np.float64)
    n = len(density)
    if n < 4:
        return 1
    ai, ao = _csr_arrays(adj)
    sources = rng.choice(n, size=min(n_samples, n), replace=False)
    hop = {h: [] for h in range(1, max_hop + 1)}
    for s in sources:
        seen = {int(s)}
        frontier = [int(s)]
        for h in range(1, max_hop + 1):
            nxt = []
            for v in frontier:
                for pos in range(ao[v], ao[v + 1]):
                    u = int(ai[pos])
                    if u not in seen:
                        seen.add(u)
                        nxt.append(u)
                        hop[h].append((density[s], density[u]))
            frontier = nxt
            if not frontier:
                break
    for h in range(1, max_hop + 1):
        pr = hop[h]
        if len(pr) < 5:
            continue
        a = np.asarray(pr, dtype=np.float64)
        if a[:, 0].std() < 1e-12 or a[:, 1].std() < 1e-12:
            continue
        if np.corrcoef(a[:, 0], a[:, 1])[0, 1] < 1.0 / math.e:
            return max(1, h)
    return max(1, max_hop)


def estimate_block_scale(density, adj, max_hop=6, n_samples=100, rng=None):
    """Estimate graph correlation length using the native CSR traversal when available."""
    density = np.ascontiguousarray(density, dtype=np.float64)
    n = len(density)
    if n < 4:
        return 1
    if int(max_hop) <= 0:
        return max(1, int(max_hop))
    rng = rng or np.random.default_rng(0)
    if _HAS_NULL_NATIVE:
        ai, ao = _csr_arrays(adj)
        ai = np.ascontiguousarray(ai, dtype=np.int64)
        ao = np.ascontiguousarray(ao, dtype=np.int64)
        sources = np.ascontiguousarray(
            rng.choice(n, size=min(int(n_samples), n), replace=False),
            dtype=np.int64,
        )
        return int(_null_native.estimate_block_scale(
            density, ai, ao, sources, int(max_hop)
        ))
    return _estimate_block_scale_python(
        density, adj, max_hop=max_hop, n_samples=n_samples, rng=rng
    )


def graph_distance_blocks(adj, radius):
    """Partition a CSR graph into bounded-radius graph-distance blocks."""
    ai, ao = _csr_arrays(adj)
    return topology.distance_blocks(ai, ao, radius)


def prominence_null(density, adj, blocks, P, alpha, rng, edges=None, resid=None,
                    n_jobs=1, parallel_max_workers=8, parallel_memory_mb=1024):
    """Generate block-permutation null fields for the configured filtration."""
    density = np.ascontiguousarray(density, dtype=np.float64)
    blocks = np.ascontiguousarray(blocks, dtype=np.int64)
    P = int(P)
    if resid is not None:
        edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        ea = np.ascontiguousarray(edges[:, 0], dtype=np.int64)
        eb = np.ascontiguousarray(edges[:, 1], dtype=np.int64)
        resid = np.ascontiguousarray(resid, dtype=np.float64)
        bytes_per_worker = max(1, density.nbytes * 2 + resid.nbytes + ea.nbytes + eb.nbytes)
    else:
        ai, ao = _csr_arrays(adj)
        bytes_per_worker = max(1, density.nbytes * 2 + ai.nbytes + ao.nbytes)
    workers = resolve_workers(n_jobs, P, parallel_max_workers,
                              parallel_memory_mb, bytes_per_worker)
    if workers == 1:
        if resid is not None:
            return topology.prominence_null_edges(density, ea, eb, resid, blocks,
                                                   P, alpha, rng)
        return topology.prominence_null(density, ai, ao, blocks, P, alpha, rng)

    block_idx = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]

    def permuted_fields():
        # RNG is used only in the caller thread, order and draws do not depend on scheduling.
        """Yield density fields permuted independently within graph blocks."""
        for _ in range(P):
            perm = density.copy()
            for idx in block_idx:
                if len(idx) > 1:
                    perm[idx] = density[rng.permutation(idx)]
            yield perm

    if resid is None:
        def one(perm):
            """Return maximum H0 prominence for one permuted vertex-density field."""
            return topology.max_prominence_arrays(perm, ai, ao)
    else:
        def one(perm):
            """Return maximum H0 prominence for one permuted edge-filtration field."""
            elev = topology.edge_levels_from_resid(perm, ea, eb, resid)
            return topology.max_prominence_edges(perm, ea, eb, elev)

    maxs = np.fromiter(ordered_thread_map(one, permuted_fields(), workers),
                       dtype=np.float64, count=P)
    return float(np.quantile(maxs, 1.0 - alpha))


def compute_tau(density, adj, P, alpha, rng, ensemble=True, r_auto=None,
                edges=None, resid=None, n_jobs=1, parallel_max_workers=8,
                parallel_memory_mb=1024):
    """Estimate a persistence threshold from a block-permutation null ensemble."""
    if r_auto is None:
        r_auto = estimate_block_scale(density, adj, rng=rng)
    radii = [r_auto] if not ensemble else [max(1, r_auto - 1), r_auto, r_auto + 1]
    taus = []
    for r in radii:
        blocks = graph_distance_blocks(adj, r)
        taus.append(prominence_null(
            density, adj, blocks, P, alpha, rng, edges=edges, resid=resid,
            n_jobs=n_jobs, parallel_max_workers=parallel_max_workers,
            parallel_memory_mb=parallel_memory_mb))
    return float(np.median(taus)), int(r_auto)


def _poisson_maximum(U, lo, span, use_box, n, k_eff, m, normalized, d_hat,
                     mreach, tree_workers):
    """Compute maximum finite H0 prominence for one Poisson null realization."""
    Xs = np.ascontiguousarray(lo + U * span if use_box else U, dtype=np.float64)
    kd, ki = _query_knn_euclidean(Xs, k_eff, workers=tree_workers)
    if not normalized:
        base = 1.0 / np.maximum(kd.mean(axis=1), 1e-12)
    else:
        rk = np.maximum(kd[:, -1], 1e-12)
        dh = (float(d_hat) if d_hat is not None else
              estimate_intrinsic_dim(Xs, workers=tree_workers))
        volume = np.pi ** (dh / 2.0) / _gamma(dh / 2.0 + 1.0)
        base = k_eff / np.maximum(n * volume * np.power(rk, dh), 1e-300)
    ell = log_density(base)

    if mreach:
        edges, edge_dist = _edges_and_distances_from_knn(ki, kd)
        ea = np.ascontiguousarray(edges[:, 0], dtype=np.int64)
        eb = np.ascontiguousarray(edges[:, 1], dtype=np.int64)
        _, elev = topology.edge_fields_from_dist(ell, edges, edge_dist)
        return topology.max_prominence_edges(ell, ea, eb, elev)

    edges, ai, ao = topology.knn_edges_csr(ki)
    return topology.max_prominence_arrays(ell, ai, ao)


def poisson_null_tau(X, k, P, alpha, rng, d_hat=None, normalized=False,
                     mreach=False, dim=None, n_jobs=1,
                     parallel_max_workers=8, parallel_memory_mb=1024):
    """Estimate a Poisson null threshold for one graph scale."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, D = X.shape
    k_eff = min(int(k), n - 1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    m = D if dim is None else int(dim)
    P = int(P)
    # A conservative estimate includes the sample, point cloud, kNN index/dist,
    # directed keys, graph edges and scratch for the native sweep.
    bytes_per_worker = max(1, 24 * n * m + 40 * n * k_eff + 64 * n)
    workers = resolve_workers(n_jobs, P, parallel_max_workers,
                              parallel_memory_mb, bytes_per_worker)
    tree_workers = 1 if (workers > 1 or int(n_jobs) == 1) else -1
    maxs = np.empty(P, dtype=np.float64)

    def samples():
        """Yield Poisson point clouds for the single-scale null model."""
        for _ in range(P):
            yield rng.random((n, m))

    def one(U):
        """Evaluate maximum persistence for one Poisson null point cloud."""
        return _poisson_maximum(U, lo, span, dim is None, n, k_eff, m,
                                normalized, d_hat, mreach, tree_workers)

    for p, value in enumerate(ordered_thread_map(one, samples(), workers)):
        maxs[p] = value
    return float(np.quantile(maxs, 1.0 - alpha)), maxs


def mode_count_stability(X_sub, edges, tau, B, f, alpha, k, rng,
                         d_hat=None, normalized=False, mreach=True,
                         n_jobs=1, parallel_max_workers=8,
                         parallel_memory_mb=1024, adaptive_k=False,
                         adaptive_k_min=4, adaptive_k_max=None,
                         adaptive_k_sensitivity=2.5, adaptive_k_window=4,
                         adaptive_k_patience=2, metric="euclidean"):
    """Estimate significant-mode stability at one selected graph scale."""
    X_sub = np.ascontiguousarray(X_sub, dtype=np.float64)
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    n = len(X_sub)
    if n < 3:
        fil, _ = build_filtration(X_sub, edges, k, d_hat, normalized, mreach)
        c = filtration_count(fil, tau)
        return c, np.array([c])

    m = max(2, int(math.floor(f * n)))
    B = int(B)
    bytes_per_worker = max(1, 32 * m * X_sub.shape[1] + 48 * m * min(k, m - 1)
                            + 24 * len(edges))
    workers = resolve_workers(n_jobs, B, parallel_max_workers,
                              parallel_memory_mb, bytes_per_worker)
    tree_workers = 1 if (workers > 1 or int(n_jobs) == 1) else -1

    def samples():
        """Yield reproducible subsamples for the selected graph scale."""
        for _ in range(B):
            yield np.sort(rng.choice(n, size=m, replace=False))

    def one(idx):
        """Evaluate significant-mode count for one stability subsample."""
        X_small = np.ascontiguousarray(X_sub[idx], dtype=np.float64)
        if adaptive_k:
            qmax = min(int(adaptive_k_max or k), len(X_small) - 1)
            qmax = max(1, qmax)
            local_graph = build_knn_graph(X_small, qmax, metric=metric,
                                          n_jobs=tree_workers, store_knn=True)
            local_density, _ = adaptive_knn_density_from_distances(
                local_graph.knn_dist, min(int(adaptive_k_min), qmax),
                adaptive_k_sensitivity, adaptive_k_window,
                adaptive_k_patience, normalized=normalized, d_hat=d_hat,
                k_bias_correction=adaptive_k)
            graph_scale = knn_scale_subgraph(local_graph, min(int(k), qmax))
            fil, _ = build_filtration_from_density(
                X_small, graph_scale.edges, local_density, mreach=mreach)
        else:
            local_edges = induced_edges(edges, idx, n)
            fil, _ = build_filtration(X_small, local_edges, k, d_hat,
                                      normalized, mreach, workers=tree_workers)
        return filtration_count(fil, tau)

    counts = np.fromiter(ordered_thread_map(one, samples(), workers),
                         dtype=np.int64, count=B)
    mstar = 0
    for mm in range(0, int(counts.max()) + 1):
        if np.mean(counts >= mm) >= (1.0 - alpha):
            mstar = mm
    return int(mstar), counts



def select_multiscale_from_pvalues(observed_prominence, observed_components,
                                   p_prominence, p_components,
                                   tau_thresholds, component_thresholds,
                                   alpha, family_size=None):
    """Select a graph scale using family-wise corrected p-values."""
    observed_prominence = np.asarray(observed_prominence, dtype=np.float64)
    observed_components = np.asarray(observed_components, dtype=np.int64)
    p_prominence = np.asarray(p_prominence, dtype=np.float64)
    p_components = np.asarray(p_components, dtype=np.float64)
    tau_thresholds = np.asarray(tau_thresholds, dtype=np.float64)
    component_thresholds = np.asarray(component_thresholds, dtype=np.float64)
    m = len(observed_prominence)
    shapes = (observed_components.shape, p_prominence.shape,
              p_components.shape, tau_thresholds.shape,
              component_thresholds.shape)
    if not m or any(shape != (m,) for shape in shapes):
        raise ValueError("all multiscale selection arrays must have equal non-zero length")
    family = m if family_size is None else int(family_size)
    if family < m:
        raise ValueError("family_size cannot be smaller than the evaluated scale count")
    p_scale = np.minimum(1.0, 2.0 * np.minimum(p_prominence, p_components))
    adjusted_p = float(min(1.0, family * float(np.min(p_scale))))
    prominence_score = observed_prominence / np.maximum(tau_thresholds, 1e-12)
    component_score = observed_components / np.maximum(component_thresholds, 1.0)
    scores = np.maximum(prominence_score, component_score)
    best = int(np.lexsort((-scores, p_scale))[0])
    return {
        "best": best,
        "p_prominence": p_prominence,
        "p_components": p_components,
        "p_scale": p_scale,
        "adjusted_p": adjusted_p,
        "scores": scores,
        "component_score": component_score,
        "family_size": family,
    }


def select_multiscale_mode_count(observed_prominence, observed_components,
                                null_maxima, null_component_counts,
                                tau_thresholds, component_thresholds, alpha):
    """Select the supported mode count from multi-scale evidence."""
    null_maxima = np.asarray(null_maxima, dtype=np.float64)
    null_component_counts = np.asarray(null_component_counts, dtype=np.int64)
    if (null_maxima.ndim != 2 or null_component_counts.shape != null_maxima.shape
            or null_maxima.shape[1] != len(observed_prominence)):
        raise ValueError("null arrays must have matching shape (n_null, n_scales)")
    if len(null_maxima) < 1:
        raise ValueError("at least one null realization is required")
    obs_p = np.asarray(observed_prominence, dtype=np.float64)
    obs_c = np.asarray(observed_components, dtype=np.int64)
    p_prom = (
        1.0 + np.sum(null_maxima >= obs_p[None, :], axis=0)
    ) / (len(null_maxima) + 1.0)
    p_comp = (
        1.0 + np.sum(null_component_counts >= obs_c[None, :], axis=0)
    ) / (len(null_component_counts) + 1.0)
    return select_multiscale_from_pvalues(
        obs_p, obs_c, p_prom, p_comp, tau_thresholds,
        component_thresholds, alpha, family_size=len(obs_p),
    )


def _mode_count_from_selected_scale(merge_tree, selection, tau_thresholds,
                                    alpha, n_scales):
    """Combine selected-scale persistence and component evidence into a mode count."""
    best = int(selection["best"])
    tau = float(tau_thresholds[best])
    p_comp = float(selection["p_components"][best])
    n_essential = int(np.sum(np.isinf(merge_tree.mode_prom)))
    n_finite_sig = int(np.sum(
        np.isfinite(merge_tree.mode_prom) & (merge_tree.mode_prom >= tau)
    ))
    component_supported = p_comp <= float(alpha) / (2.0 * int(n_scales)) + 1e-12
    topological_count = (n_essential if component_supported else min(n_essential, 1)) + n_finite_sig
    if selection["adjusted_p"] > float(alpha) + 1e-12:
        topological_count = min(topological_count, 1)
    return int(max(0, topological_count))


def multiscale_mode_count_stability(
    X, k_scales, B, f, alpha, rng, *, null_maxima, tau_thresholds,
    p_component_reference, component_thresholds,
    adaptive_k=True, adaptive_k_min=4, adaptive_k_max=None,
    adaptive_k_sensitivity=2.5, adaptive_k_window=4,
    adaptive_k_patience=2, normalized=False, d_hat=None,
    mreach=True, metric="euclidean", n_jobs=1,
    parallel_max_workers=8, parallel_memory_mb=1024,
    return_diagnostics=False,
):
    """Re-select graph K independently in each multi-scale stability subsample."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, D = X.shape
    B = int(B)
    if B < 1:
        raise ValueError("B must be >= 1")
    if not (0.0 < float(f) < 1.0):
        raise ValueError("f must be in (0, 1)")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    scales = tuple(sorted({int(k) for k in k_scales if int(k) >= 2}))
    if not scales:
        raise ValueError("k_scales must contain at least one K >= 2")
    null_maxima = np.asarray(null_maxima, dtype=np.float64)
    tau_thresholds = np.asarray(tau_thresholds, dtype=np.float64)
    p_component_reference = np.asarray(p_component_reference, dtype=np.float64)
    component_thresholds = np.asarray(component_thresholds, dtype=np.float64)
    M = len(scales)
    if (null_maxima.ndim != 2 or null_maxima.shape[1] != M or
            tau_thresholds.shape != (M,) or p_component_reference.shape != (M,) or
            component_thresholds.shape != (M,)):
        raise ValueError("parent null and threshold arrays must align with k_scales")

    sample_size = max(2, int(math.floor(float(f) * n)))
    if sample_size < 3:
        counts = np.ones(B, dtype=np.int64)
        out = (1, counts)
        return (*out, {"selected_k": np.zeros(B, dtype=np.int64),
                       "adjusted_p": np.ones(B),
                       "candidate_scales": np.empty(0, dtype=np.int64)}) if return_diagnostics else out

    valid_indices = np.asarray([j for j, k in enumerate(scales)
                                if k <= sample_size - 1], dtype=np.int64)
    if len(valid_indices) == 0:
        counts = np.ones(B, dtype=np.int64)
        out = (1, counts)
        return (*out, {"selected_k": np.zeros(B, dtype=np.int64),
                       "adjusted_p": np.ones(B),
                       "candidate_scales": np.empty(0, dtype=np.int64)}) if return_diagnostics else out
    valid_scales = tuple(scales[j] for j in valid_indices)
    valid_null = null_maxima[:, valid_indices]
    valid_tau = tau_thresholds[valid_indices]
    valid_pcomp = p_component_reference[valid_indices]
    valid_cth = component_thresholds[valid_indices]
    family_size = M  # keep the original predeclared family size after clipping

    density_kmax = (min(int(adaptive_k_max), sample_size - 1)
                    if adaptive_k_max is not None else max(valid_scales))
    query_k = min(max(max(valid_scales), density_kmax), sample_size - 1)
    density_kmax = min(density_kmax, query_k)
    density_kmin = min(max(2, int(adaptive_k_min)), density_kmax)
    if not adaptive_k:
        density_kmin = density_kmax

    bytes_per_worker = max(1, 32 * sample_size * D +
                           48 * sample_size * query_k +
                           32 * sample_size * query_k * max(1, len(valid_scales)))
    workers = resolve_workers(n_jobs, B, parallel_max_workers,
                              parallel_memory_mb, bytes_per_worker)
    tree_workers = 1 if workers > 1 or int(n_jobs) == 1 else -1

    def tasks():
        """Yield bounded tasks for multi-scale stability evaluation."""
        for _ in range(B):
            yield np.sort(rng.choice(n, size=sample_size, replace=False))

    def one(idx):
        """Re-select graph K and evaluate modes for one stability subsample."""
        X_small = np.ascontiguousarray(X[idx], dtype=np.float64)
        dmax = min(density_kmax, len(X_small) - 1)
        qmax = min(max(max(valid_scales), dmax), len(X_small) - 1)
        local_graph = build_knn_graph(X_small, qmax, metric=metric,
                                      n_jobs=tree_workers, store_knn=True)
        density_dist = np.ascontiguousarray(local_graph.knn_dist[:, :dmax], dtype=np.float64)
        if adaptive_k:
            base, _ = adaptive_knn_density_from_distances(
                density_dist, min(density_kmin, dmax),
                adaptive_k_sensitivity, adaptive_k_window,
                adaptive_k_patience, normalized=normalized, d_hat=d_hat,
                k_bias_correction=True,
            )
        else:
            base, _ = adaptive_knn_density_from_distances(
                density_dist, dmax, adaptive_k_sensitivity,
                adaptive_k_window, adaptive_k_patience,
                normalized=normalized, d_hat=d_hat,
                k_bias_correction=False,
            )

        fused_native = bool(topology.USING_CYTHON)
        if fused_native:
            # multi-scale semantics, fused in Cython: evaluate all K scales from the
            # same kNN cache, reuse the vertex event order, and retain only the
            # per-scale maximum persistence/component statistics. The selected
            # scale's full merge tree is materialised after scale selection.
            ell_small = log_density(base)
            observed_prominence, observed_components = topology.multiscale_graph_statistics(
                X_small, local_graph.knn_idx, ell_small,
                np.asarray(valid_scales, dtype=np.int64), mreach=mreach,
            )
            merge_trees = None
        else:
            observed_prominence = np.zeros(len(valid_scales), dtype=np.float64)
            observed_components = np.ones(len(valid_scales), dtype=np.int64)
            merge_trees = []
            for j, k in enumerate(valid_scales):
                sub = knn_scale_subgraph(local_graph, k)
                fil, _ = build_filtration_from_density(
                    X_small, sub.edges, base, mreach=mreach)
                mt = filtration_merge_tree(fil, sub.adj_csr)
                merge_trees.append(mt)
                finite = mt.mode_prom[np.isfinite(mt.mode_prom)]
                observed_prominence[j] = float(finite.max()) if len(finite) else 0.0
                ai, ao = sub.adj_csr
                components = topology.distance_blocks(ai, ao, len(X_small))
                observed_components[j] = int(np.unique(components).size)

        # Prominence p-values use the parent's null family. Component p-values
        # are also parent-calibrated, but the observed component count is freshly
        # computed above for every subsample.
        p_prom = (
            1.0 + np.sum(valid_null >= observed_prominence[None, :], axis=0)
        ) / (len(valid_null) + 1.0)
        selection = select_multiscale_from_pvalues(
            observed_prominence, observed_components, p_prom,
            valid_pcomp, valid_tau, valid_cth, alpha,
            family_size=family_size,
        )
        best = int(selection["best"])
        if fused_native:
            # Only the selected scale needs a full merge tree for m*, the other
            # scales were evaluated with the exact max-persistence kernel above.
            selected_sub = knn_scale_subgraph(local_graph, valid_scales[best])
            selected_fil, _ = build_filtration_from_density(
                X_small, selected_sub.edges, base, mreach=mreach)
            selected_tree = filtration_merge_tree(
                selected_fil, selected_sub.adj_csr)
        else:
            selected_tree = merge_trees[best]
        count = _mode_count_from_selected_scale(
            selected_tree, selection, valid_tau, alpha, family_size)
        return count, int(valid_scales[best]), float(selection["adjusted_p"])

    rows = list(ordered_thread_map(one, tasks(), workers))
    counts = np.asarray([r[0] for r in rows], dtype=np.int64)
    selected_ks = np.asarray([r[1] for r in rows], dtype=np.int64)
    adjusted_ps = np.asarray([r[2] for r in rows], dtype=np.float64)
    mstar = 0
    for mm in range(0, int(counts.max()) + 1):
        if np.mean(counts >= mm) >= (1.0 - float(alpha)):
            mstar = mm
    out = (int(mstar), counts)
    if return_diagnostics:
        return (*out, {
            "selected_k": selected_ks,
            "adjusted_p": adjusted_ps,
            "candidate_scales": np.asarray(valid_scales, dtype=np.int64),
            "family_size": family_size,
            "null_P": int(len(null_maxima)),
        })
    return out


def poisson_null_multiscale_tau(
    X, k_scales, P, alpha, rng, *, adaptive_k=True, adaptive_k_min=4,
    adaptive_k_sensitivity=2.5, adaptive_k_window=4,
    adaptive_k_patience=2, adaptive_k_max=None, d_hat=None,
    normalized=False, mreach=False, dim=None, n_jobs=1,
    parallel_max_workers=8, parallel_memory_mb=1024,
    include_components=False,
):
    """Calibrate a family of graph scales against shared Poisson null realizations."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, D = X.shape
    scales = tuple(sorted({min(int(k), n - 1) for k in k_scales if int(k) >= 2}))
    if not scales:
        raise ValueError("k_scales must contain at least one K >= 2")
    P = int(P)
    if P < 1:
        raise ValueError("P must be >= 1")
    graph_kmax = max(scales)
    # Density K is an independent scale in independently of graph scale, whether it is adaptive per
    # point or fixed. The null must use the same maximum density neighbourhood
    # as the observed estimator, otherwise the Poisson calibration is mismatched.
    density_kmax = (min(int(adaptive_k_max), n - 1)
                    if adaptive_k_max is not None else graph_kmax)
    query_k = min(max(graph_kmax, density_kmax), n - 1)
    density_kmin = min(max(2, int(adaptive_k_min)), query_k)
    m = D if dim is None else int(dim)
    density_dimension = float(d_hat) if d_hat is not None else float(m)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    bytes_per_worker = max(1, 24 * n * m + 40 * n * query_k + 64 * n
                           + 24 * n * query_k * max(1, len(scales)))
    workers = resolve_workers(n_jobs, P, parallel_max_workers,
                              parallel_memory_mb, bytes_per_worker)
    tree_workers = 1 if workers > 1 or int(n_jobs) == 1 else -1

    def samples():
        """Yield Poisson null clouds for the full graph-scale family."""
        for _ in range(P):
            yield rng.random((n, m))

    def one(U):
        """Evaluate all graph scales for one Poisson null realization."""
        Xs = np.ascontiguousarray(lo + U * span if dim is None else U,
                                  dtype=np.float64)
        kd, ki = _query_knn_euclidean(Xs, query_k, workers=tree_workers)
        if adaptive_k:
            base, _ = adaptive_knn_density_from_distances(
                kd[:, :density_kmax], density_kmin, adaptive_k_sensitivity,
                adaptive_k_window, adaptive_k_patience,
                normalized=normalized, d_hat=density_dimension,
            )
        else:
            dk = min(density_kmax, query_k)
            if normalized:
                rk = np.maximum(kd[:, dk - 1], 1e-12)
                dh = float(d_hat) if d_hat is not None else 1.0
                volume = np.pi ** (dh / 2.0) / _gamma(dh / 2.0 + 1.0)
                base = dk / np.maximum(n * volume * np.power(rk, dh), 1e-300)
            else:
                base = 1.0 / np.maximum(kd[:, :dk].mean(axis=1), 1e-12)
        ell = log_density(base)
        if topology.USING_CYTHON:
            values, components = topology.multiscale_graph_statistics(
                Xs, ki, ell, np.asarray(scales, dtype=np.int64), mreach=mreach)
            return (values, components) if include_components else values

        values = np.empty(len(scales), dtype=np.float64)
        components = np.empty(len(scales), dtype=np.int64) if include_components else None
        for j, k in enumerate(scales):
            edges, ai, ao = topology.knn_edges_csr(np.ascontiguousarray(ki[:, :k]))
            if include_components:
                block_ids = topology.distance_blocks(ai, ao, n)
                components[j] = len(np.unique(block_ids))
            if mreach:
                _, elev = topology.edge_fields(Xs, edges, ell)
                ea = np.ascontiguousarray(edges[:, 0], dtype=np.int64)
                eb = np.ascontiguousarray(edges[:, 1], dtype=np.int64)
                values[j] = topology.max_prominence_edges(ell, ea, eb, elev)
            else:
                values[j] = topology.max_prominence_arrays(ell, ai, ao)
        return (values, components) if include_components else values

    rows = list(ordered_thread_map(one, samples(), workers))
    if include_components:
        maxima = np.vstack([r[0] for r in rows])
        component_counts = np.vstack([r[1] for r in rows]).astype(np.int64, copy=False)
        family_alpha = float(alpha) / (2.0 * len(scales))
        thresholds = np.quantile(maxima, 1.0 - family_alpha, axis=0)
        component_thresholds = np.quantile(component_counts, 1.0 - family_alpha, axis=0)
        return (np.asarray(thresholds, dtype=np.float64), maxima,
                np.asarray(component_thresholds, dtype=np.float64), component_counts)

    maxima = np.vstack(rows)
    family_alpha = float(alpha) / len(scales)
    thresholds = np.quantile(maxima, 1.0 - family_alpha, axis=0)
    return np.asarray(thresholds, dtype=np.float64), maxima

