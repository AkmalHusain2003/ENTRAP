"""Unified topology API: prefers compiled Cython, falls back to pure Python."""
import numpy as np
try:
    from ._topology import (merge_tree_arrays, distance_blocks, prominence_null,
                            count_significant, assign_basins,
                            merge_tree_edges, count_significant_edges, prominence_null_edges,
                            max_prominence_arrays, max_prominence_edges,
                            knn_edges_csr as _knn_edges_csr,
                            edges_to_csr as _edges_to_csr,
                            edge_fields as _edge_fields,
                            edge_fields_from_dist as _edge_fields_from_dist,
                            edge_levels_from_resid as _edge_levels_from_resid,
                            induced_edges_csr as _induced_edges_csr,
                            multiscale_graph_statistics as _multiscale_graph_statistics,
                            multiscale_msf_statistics as _multiscale_msf_statistics)
    USING_CYTHON = True
except Exception:
    USING_CYTHON = False

    def _find(p, x):
        """Find a disjoint-set representative using path compression."""
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def _sweep_py(dens, adj, full):
        """Compute H0 persistence on a vertex filtration using union-find."""
        n = len(dens)
        order = np.argsort(-dens, kind="stable")
        parent = np.arange(n)
        active = np.zeros(n, bool)
        birth = np.full(n, -np.inf)
        root_mode = np.full(n, -1, np.int64)
        vertex_mode = np.full(n, -1, np.int64)
        mp = []
        mb = []
        mpr = []
        pairs = []
        for v in order:
            v = int(v)
            active[v] = True
            birth[v] = dens[v]
            roots = set()
            for u in adj[v]:
                if active[u]:
                    roots.add(_find(parent, u))
            roots.discard(v)
            if not roots:
                mid = len(mp)
                mp.append(-1)
                mb.append(dens[v])
                mpr.append(np.inf)
                root_mode[v] = mid
                if full:
                    vertex_mode[v] = mid
            else:
                roots = list(roots)
                elder = max(roots, key=lambda r: birth[r])
                em = root_mode[elder]
                if full:
                    vertex_mode[v] = em
                for r in roots:
                    if r == elder:
                        continue
                    rm = root_mode[r]
                    mp[rm] = em
                    mpr[rm] = birth[r]-dens[v]
                    if full:
                        pairs.append((birth[r], dens[v]))
                    parent[r] = elder
                parent[v] = elder
        return (
            vertex_mode,
            np.array(mp, np.int64),
            np.array(mb),
            np.array(mpr),
            np.array(pairs).reshape(-1, 2),
        )

    def _csr_adj(ai, ao):
        """Convert CSR adjacency arrays into a list-based representation."""
        return [ai[ao[i]:ao[i+1]].tolist() for i in range(len(ao)-1)]

    def merge_tree_arrays(dens, ai, ao):
        """Construct H0 merge-tree arrays from a vertex filtration (Edelsbrunner et al., 2002)."""
        return _sweep_py(np.asarray(dens), _csr_adj(ai, ao), True)

    def distance_blocks(ai, ao, radius):
        """Partition a graph into bounded graph-distance blocks."""
        adj = _csr_adj(ai, ao)
        n = len(adj)
        block = np.full(n, -1, np.int64)
        bid = 0
        for s in range(n):
            if block[s] >= 0:
                continue
            block[s] = bid
            frontier = [s]
            for _ in range(int(radius)):
                nxt = []
                for v in frontier:
                    for u in adj[v]:
                        if block[u] < 0:
                            block[u] = bid
                            nxt.append(u)
                frontier = nxt
                if not frontier:
                    break
            bid += 1
        return block

    def prominence_null(dens, ai, ao, blocks, P, alpha, rng):
        """Generate block-permutation null fields for the configured filtration."""
        adj = _csr_adj(ai, ao)
        dens = np.asarray(dens)
        block_idx = [np.where(blocks == b)[0] for b in np.unique(blocks)]
        maxs = np.empty(P)
        for p in range(P):
            perm = dens.copy()
            for idx in block_idx:
                if len(idx) > 1:
                    perm[idx] = dens[rng.permutation(idx)]
            _, _, _, mpr, _ = _sweep_py(perm, adj, False)
            fin = mpr[np.isfinite(mpr)]
            maxs[p] = fin.max() if len(fin) else 0.0
        return float(np.quantile(maxs, 1.0-alpha))

    def count_significant(dens, ai, ao, tau):
        """Count significant modes in a vertex filtration."""
        _, _, _, mpr, _ = _sweep_py(np.asarray(dens), _csr_adj(ai, ao), False)
        return int(np.sum(mpr >= tau))

    def _sweep_edges_py(dens, ea, eb, elev, full):
        """Compute H0 persistence on a graph-edge filtration using union-find."""
        dens = np.asarray(dens, np.float64)
        elev = np.asarray(elev, np.float64)
        n = len(dens)
        ne = len(ea)
        order_v = np.argsort(-dens, kind="stable")
        order_e = np.argsort(-elev, kind="stable")
        parent = np.arange(n)
        birth = np.zeros(n)
        root_mode = np.full(n, -1, np.int64)
        vertex_mode = np.full(n, -1, np.int64)
        mp = []
        mb = []
        mpr = []
        pairs = []
        iv = ie = 0
        while iv < n or ie < ne:
            if ie >= ne:
                take_vertex = True
            elif iv >= n:
                take_vertex = False
            else:
                take_vertex = dens[order_v[iv]] >= elev[order_e[ie]]
            if take_vertex:
                v = int(order_v[iv])
                iv += 1
                birth[v] = dens[v]
                mid = len(mp)
                mp.append(-1)
                mb.append(dens[v])
                mpr.append(np.inf)
                root_mode[v] = mid
                if full:
                    vertex_mode[v] = mid
            else:
                e = int(order_e[ie])
                ie += 1
                ra = _find(parent, int(ea[e]))
                rb = _find(parent, int(eb[e]))
                if ra == rb:
                    continue
                if birth[ra] > birth[rb] or (
                    birth[ra] == birth[rb] and root_mode[ra] < root_mode[rb]
                ):
                    elder, young = ra, rb
                else:
                    elder, young = rb, ra
                mp[root_mode[young]] = root_mode[elder]
                mpr[root_mode[young]] = birth[young] - elev[e]
                if full:
                    pairs.append((birth[young], elev[e]))
                parent[young] = elder
        return (vertex_mode, np.array(mp, np.int64), np.array(mb), np.array(mpr),
                np.array(pairs).reshape(-1, 2))

    def merge_tree_edges(dens, ea, eb, elev):
        """Construct H0 merge-tree arrays from edge-filtration events (Edelsbrunner et al., 2002)."""
        return _sweep_edges_py(dens, ea, eb, elev, True)

    def count_significant_edges(dens, ea, eb, elev, tau):
        """Count significant modes under edge filtration."""
        _, _, _, mpr, _ = _sweep_edges_py(dens, ea, eb, elev, False)
        return int(np.sum(mpr >= tau))

    def prominence_null_edges(dens, ea, eb, resid, blocks, P, alpha, rng):
        """Generate a block-permutation null for edge-filtration prominence."""
        dens = np.asarray(dens, np.float64)
        resid = np.asarray(resid, np.float64)
        block_idx = [np.where(blocks == b)[0] for b in np.unique(blocks)]
        maxs = np.empty(P)
        for p in range(P):
            perm = dens.copy()
            for idx in block_idx:
                if len(idx) > 1:
                    perm[idx] = dens[rng.permutation(idx)]
            pa = perm[ea]
            pb_ = perm[eb]
            m = np.minimum(pa, pb_)
            M = np.maximum(pa, pb_)
            t = 0.5*(M - m) + resid
            elev = m + np.where(t < 0.0, t, 0.0)
            _, _, _, mpr, _ = _sweep_edges_py(perm, ea, eb, elev, False)
            fin = mpr[np.isfinite(mpr)]
            maxs[p] = fin.max() if len(fin) else 0.0
        return float(np.quantile(maxs, 1.0-alpha))

    def assign_basins(vertex_mode, mode_parent, mode_prom, tau):
        """Assign vertices to their persistent mode basins."""
        out = np.empty(len(vertex_mode), np.int64)
        for i, vm in enumerate(vertex_mode):
            m = int(vm)
            while mode_prom[m] < tau and mode_parent[m] != -1:
                m = mode_parent[m]
            out[i] = m
        return out


def max_prominence_arrays(dens, ai, ao):
    """Compute maximum finite persistence without materializing a full merge tree."""
    if USING_CYTHON:
        from ._topology import max_prominence_arrays as _fn
        return _fn(np.ascontiguousarray(dens, dtype=np.float64),
                   np.ascontiguousarray(ai, dtype=np.int64),
                   np.ascontiguousarray(ao, dtype=np.int64))
    _, _, _, mpr, _ = _sweep_py(np.asarray(dens), _csr_adj(ai, ao), False)
    fin = mpr[np.isfinite(mpr)]
    return float(fin.max()) if len(fin) else 0.0


def max_prominence_edges(dens, ea, eb, elev):
    """Compute maximum finite edge-filtration persistence without materializing a full merge tree."""
    if USING_CYTHON:
        from ._topology import max_prominence_edges as _fn
        return _fn(np.ascontiguousarray(dens, dtype=np.float64),
                   np.ascontiguousarray(ea, dtype=np.int64),
                   np.ascontiguousarray(eb, dtype=np.int64),
                   np.ascontiguousarray(elev, dtype=np.float64))
    _, _, _, mpr, _ = _sweep_edges_py(dens, ea, eb, elev, False)
    fin = mpr[np.isfinite(mpr)]
    return float(fin.max()) if len(fin) else 0.0


def knn_edges_csr(knn_idx, return_first=False):
    """Convert kNN neighbours into unique undirected edges and CSR arrays."""
    ki = np.ascontiguousarray(knn_idx, dtype=np.int64)
    if ki.ndim != 2:
        raise ValueError("knn_idx must be a 2D integer array")
    n, k = ki.shape
    if n and k and (np.any(ki < 0) or np.any(ki >= n)):
        raise ValueError("kNN indices must be in [0, n)")
    if USING_CYTHON:
        return _knn_edges_csr(ki, bool(return_first))
    if n == 0 or k == 0:
        base = (np.empty((0, 2), np.int64), np.empty(0, np.int64),
                np.zeros(n + 1, np.int64))
        if return_first:
            return base + (np.empty(0, np.int64),)
        return base
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = ki.reshape(-1)
    keys, first = np.unique(np.minimum(src, dst) * n + np.maximum(src, dst),
                            return_index=True)
    edges = np.column_stack((keys // n, keys % n)).astype(np.int64, copy=False)
    ai, ao = edges_to_csr(n, edges)
    if return_first:
        return edges, ai, ao, first.astype(np.int64, copy=False)
    return edges, ai, ao


def edges_to_csr(n, edges):
    """Convert undirected edge pairs into sorted symmetric CSR arrays."""
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge endpoint outside [0, n)")
    if USING_CYTHON:
        return _edges_to_csr(n, edges)
    if n == 0:
        return np.empty(0, np.int64), np.zeros(1, np.int64)
    if len(edges) == 0:
        return np.empty(0, np.int64), np.zeros(n + 1, np.int64)
    src = np.concatenate((edges[:, 0], edges[:, 1]))
    dst = np.concatenate((edges[:, 1], edges[:, 0]))
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    degree = np.bincount(src, minlength=n)
    ao = np.empty(n + 1, dtype=np.int64)
    ao[0] = 0
    np.cumsum(degree, out=ao[1:])
    return np.ascontiguousarray(dst), ao


def edge_fields(X, edges, ell):
    """Compute edge residuals and mutual-reachability filtration levels."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    ell = np.ascontiguousarray(ell, dtype=np.float64)
    if USING_CYTHON:
        return _edge_fields(X, edges, ell)
    if len(edges) == 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    delta = X[edges[:, 0]] - X[edges[:, 1]]
    dist = np.linalg.norm(delta, axis=1)
    return edge_fields_from_dist(ell, edges, dist)


def edge_fields_from_dist(ell, edges, distances):
    """Compute edge filtration fields from precomputed neighbour distances."""
    ell = np.ascontiguousarray(ell, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    if USING_CYTHON:
        return _edge_fields_from_dist(ell, edges, distances)
    if len(edges) == 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    a, b = edges[:, 0], edges[:, 1]
    da, db = ell[a], ell[b]
    lo, hi = np.minimum(da, db), np.maximum(da, db)
    resid = -np.log(np.maximum(distances, 1e-300)) - 0.5 * (da + db)
    elev = lo + np.minimum(0.0, 0.5 * (hi - lo) + resid)
    return resid, elev


def edge_levels_from_resid(ell, ea, eb, resid):
    """Compute edge filtration levels from residual distances."""
    ell = np.ascontiguousarray(ell, dtype=np.float64)
    ea = np.ascontiguousarray(ea, dtype=np.int64)
    eb = np.ascontiguousarray(eb, dtype=np.int64)
    resid = np.ascontiguousarray(resid, dtype=np.float64)
    if USING_CYTHON:
        return _edge_levels_from_resid(ell, ea, eb, resid)
    da, db = ell[ea], ell[eb]
    lo, hi = np.minimum(da, db), np.maximum(da, db)
    return lo + np.minimum(0.0, 0.5 * (hi - lo) + resid)


def induced_edges_csr(n, edges, indices):
    """Extract subset edges and remap endpoints to local vertex indices."""
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    n = int(n)
    if USING_CYTHON:
        return _induced_edges_csr(n, edges, indices)
    loc = np.full(n, -1, dtype=np.int64)
    loc[indices] = np.arange(len(indices), dtype=np.int64)
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    mask = (loc[edges[:, 0]] >= 0) & (loc[edges[:, 1]] >= 0)
    return np.ascontiguousarray(loc[edges[mask]], dtype=np.int64)


def multiscale_graph_statistics_reference(X, knn_idx, ell, scales, mreach=True):
    """Run the original multi-K statistics kernel for exact parity checks."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    knn_idx = np.ascontiguousarray(knn_idx, dtype=np.int64)
    ell = np.ascontiguousarray(ell, dtype=np.float64)
    scales = np.ascontiguousarray(scales, dtype=np.int64)
    if X.ndim != 2 or knn_idx.ndim != 2 or ell.ndim != 1 or scales.ndim != 1:
        raise ValueError("X, knn_idx, ell, and scales have invalid dimensions")
    if len(X) != len(knn_idx) or len(X) != len(ell):
        raise ValueError("X, knn_idx, and ell must have matching row counts")
    if USING_CYTHON:
        return _multiscale_graph_statistics(X, knn_idx, ell, scales, bool(mreach))
    return _multiscale_graph_statistics_python(X, knn_idx, ell, scales, mreach)


def multiscale_graph_statistics(X, knn_idx, ell, scales, mreach=True):
    """Compute exact multi-K H0 summaries with shared edge ranking and filtration sorting."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    knn_idx = np.ascontiguousarray(knn_idx, dtype=np.int64)
    ell = np.ascontiguousarray(ell, dtype=np.float64)
    scales = np.ascontiguousarray(scales, dtype=np.int64)
    if X.ndim != 2 or knn_idx.ndim != 2 or ell.ndim != 1 or scales.ndim != 1:
        raise ValueError("X, knn_idx, ell, and scales have invalid dimensions")
    if len(X) != len(knn_idx) or len(X) != len(ell):
        raise ValueError("X, knn_idx, and ell must have matching row counts")
    if USING_CYTHON and _multiscale_msf_statistics is not None:
        return _multiscale_msf_statistics(X, knn_idx, ell, scales, bool(mreach))
    return multiscale_graph_statistics_reference(X, knn_idx, ell, scales, mreach)


def _multiscale_graph_statistics_python(X, knn_idx, ell, scales, mreach):
    """Run the original Python reference implementation of multi-K statistics."""
    maxima = np.empty(len(scales), dtype=np.float64)
    components = np.empty(len(scales), dtype=np.int64)
    for j, k in enumerate(scales):
        edges, ai, ao = knn_edges_csr(knn_idx[:, :int(k)])
        if mreach:
            _, elev = edge_fields(X, edges, ell)
            maxima[j] = max_prominence_edges(
                ell, edges[:, 0], edges[:, 1], elev
            )
        else:
            maxima[j] = max_prominence_arrays(ell, ai, ao)
        block_ids = distance_blocks(ai, ao, len(X))
        components[j] = np.unique(block_ids).size
    return maxima, components


def multiscale_mreach_statistics(X, knn_idx, ell, scales):
    """Compute multi-scale statistics for mutual-reachability filtration."""
    return multiscale_graph_statistics(X, knn_idx, ell, scales, mreach=True)
