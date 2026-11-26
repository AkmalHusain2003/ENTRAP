"""Conservative geometric extension for candidate tidal-tail members.

This module does not label a tail as astrophysically confirmed. It only extends
high-confidence cores along a connected, anisotropic continuation in user-selected
feature coordinates. A physical interpretation needs meaningful spatial units
(and ideally independent kinematic validation).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
try:
    from . import _membership as _membership_native
    _HAS_MEMBERSHIP_NATIVE = True
except Exception:
    _membership_native = None
    _HAS_MEMBERSHIP_NATIVE = False


def extend_tidal_tail_candidates(X, labels, basins, feature_indices, *,
                                 k_neighbors=5, max_hops=3,
                                 radius_factor=2.0, width_factor=2.5,
                                 min_anisotropy=1.5):
    """Extend accepted cluster cores with optional geometric tail candidates."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).copy()
    tail_mask = np.zeros(len(X), dtype=bool)
    if not basins or len(X) < 3:
        return labels, tail_mask
    dims = tuple(int(i) for i in feature_indices)
    if not dims or min(dims) < 0 or max(dims) >= X.shape[1]:
        raise ValueError("tail_feature_indices must refer to valid X columns")
    Z = np.ascontiguousarray(X[:, dims], dtype=np.float64)
    n, d = Z.shape
    if d < 1:
        return labels, tail_mask
    k = min(max(1, int(k_neighbors)), n - 1)
    tree = cKDTree(Z)
    dist, nbr = tree.query(Z, k=k + 1, workers=1)
    dist = np.asarray(dist, dtype=np.float64).reshape(n, k + 1)
    nbr = np.asarray(nbr, dtype=np.int64).reshape(n, k + 1)
    # Remove self even in the presence of duplicate coordinates.
    keep_dist = np.empty((n, k), dtype=np.float64)
    keep_nbr = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        m = nbr[i] != i
        di, ni = dist[i, m], nbr[i, m]
        if len(ni) < k:
            di2, ni2 = tree.query(Z[i], k=min(n, k + 2), workers=1)
            ni2 = np.atleast_1d(ni2)
            di2 = np.atleast_1d(di2)
            m2 = ni2 != i
            di, ni = di2[m2], ni2[m2]
        keep_dist[i] = di[:k]
        keep_nbr[i] = ni[:k]

    for cid, core_idx in basins.items():
        core_idx = np.asarray(core_idx, dtype=np.int64)
        if len(core_idx) < max(5, d + 2):
            continue
        core = Z[core_idx]
        center = np.median(core, axis=0)
        centered = core - center
        cov = np.cov(centered, rowvar=False)
        cov = np.atleast_2d(cov)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
        if d == 1:
            anisotropy = np.inf
            minor_width = max(float(np.sqrt(vals[0])) * 0.25, 1e-8)
        else:
            anisotropy = float(vals[0] / max(vals[1], 1e-12))
            minor_width = max(float(np.sqrt(vals[1])), 1e-8)
        if anisotropy < float(min_anisotropy):
            continue
        axis = vecs[:, 0]
        core_proj = centered @ axis
        qlo, qhi = np.quantile(core_proj, [0.05, 0.95])
        all_centered = Z - center
        proj = all_centered @ axis
        transverse = all_centered - proj[:, None] * axis[None, :]
        transverse_norm = np.linalg.norm(transverse, axis=1)
        geometry_ok = ((proj < qlo) | (proj > qhi)) & (
            transverse_norm <= float(width_factor) * minor_width)

        # Link threshold is based on the core's own local neighbour scale.
        core_radius = float(np.median(keep_dist[core_idx, -1]))
        link_limit = max(float(radius_factor) * core_radius, 1e-12)
        accepted = np.zeros(n, dtype=bool)
        accepted[core_idx] = True
        for _ in range(max(1, int(max_hops))):
            if _HAS_MEMBERSHIP_NATIVE:
                labels_c = np.ascontiguousarray(labels, dtype=np.int64)
                tail_c = np.ascontiguousarray(tail_mask, dtype=np.uint8)
                labels_c, tail_c, _ = _membership_native.propagate_tail_candidates(
                    labels_c,
                    tail_c,
                    np.ascontiguousarray(geometry_ok, dtype=np.uint8),
                    np.ascontiguousarray(keep_dist, dtype=np.float64),
                    np.ascontiguousarray(keep_nbr, dtype=np.int64),
                    np.ascontiguousarray(accepted, dtype=np.uint8),
                    float(link_limit), int(max_hops), int(cid),
                )
                labels = labels_c
                tail_mask = tail_c.astype(bool, copy=False)
                break  # the native kernel executes every synchronous hop
            candidates = np.flatnonzero((labels < 0) & ~tail_mask & geometry_ok)
            if len(candidates) == 0:
                break
            newly = []
            for i in candidates:
                near = keep_nbr[i]
                linked = accepted[near]
                if not np.any(linked):
                    continue
                if float(np.min(keep_dist[i, linked])) <= link_limit:
                    newly.append(int(i))
            if not newly:
                break
            newly = np.asarray(newly, dtype=np.int64)
            labels[newly] = int(cid)
            tail_mask[newly] = True
            accepted[newly] = True
    return labels, tail_mask
