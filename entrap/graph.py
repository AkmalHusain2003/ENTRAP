"""Build kNN graphs, estimate local density, and manage CSR graph data."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma as _gamma, gammaln
from sklearn.neighbors import NearestNeighbors
from . import topology
try:
    from . import _density as _density_native
    _HAS_DENSITY_NATIVE = True
except Exception:
    _density_native = None
    _HAS_DENSITY_NATIVE = False


def _csr_to_adj(ai, ao):
    """Convert CSR adjacency arrays to the legacy list representation."""
    return [np.asarray(ai[ao[i]:ao[i + 1]]).tolist() for i in range(len(ao) - 1)]


@dataclass
class KnnGraph:
    """Store a kNN graph with CSR adjacency and an optional neighbour cache."""
    knn_idx: np.ndarray | None
    knn_dist: np.ndarray | None
    density: np.ndarray
    order: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    adj_indices: np.ndarray
    adj_indptr: np.ndarray
    k: int
    N: int

    @property
    def adj(self):
        """Materialize the compatibility adjacency-list representation."""
        return _csr_to_adj(self.adj_indices, self.adj_indptr)

    @property
    def adj_csr(self):
        """Return the stored CSR index and pointer arrays."""
        return self.adj_indices, self.adj_indptr


@dataclass
class Subgraph:
    """Store a graph induced by a selected subset of samples."""
    indices: np.ndarray
    density: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    adj_indices: np.ndarray
    adj_indptr: np.ndarray

    @property
    def adj(self):
        """Materialize the compatibility adjacency-list representation."""
        return _csr_to_adj(self.adj_indices, self.adj_indptr)

    @property
    def adj_csr(self):
        """Return the stored CSR index and pointer arrays."""
        return self.adj_indices, self.adj_indptr


def estimate_intrinsic_dim(X, discard=0.1, workers=-1):
    """Estimate intrinsic dimension with the TwoNN method of Facco et al. (2017)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:
        return 1.0
    d, _ = cKDTree(X).query(X, k=3, workers=int(workers))
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    m = len(mu)
    if m < 4:
        return 1.0
    keep = max(2, int(np.floor((1.0 - discard) * m)))
    F = np.arange(1, m + 1) / m
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    dhat = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
    return float(min(max(dhat, 1.0), X.shape[1]))


def _query_knn_euclidean(X, k, tree=None, workers=-1):
    """Query exact Euclidean neighbours while excluding the query point itself."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    k = min(int(k), n - 1)
    if k < 1:
        return np.empty((n, 0), dtype=np.float64), np.empty((n, 0), dtype=np.int64)
    tree = cKDTree(X) if tree is None else tree
    dist, idx = tree.query(X, k=k + 1, workers=int(workers))
    dist = np.asarray(dist, dtype=np.float64).reshape(n, k + 1)
    idx = np.asarray(idx, dtype=np.int64).reshape(n, k + 1)
    rows = np.arange(n, dtype=np.int64)
    if np.array_equal(idx[:, 0], rows):
        return np.ascontiguousarray(dist[:, 1:]), np.ascontiguousarray(idx[:, 1:])

    # Equal-distance duplicates can be returned before the query point itself.
    out_d = np.empty((n, k), dtype=np.float64)
    out_i = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        keep = idx[i] != i
        di, ii = dist[i, keep], idx[i, keep]
        if len(ii) < k:
            di2, ii2 = tree.query(X[i], k=min(n, k + 2), workers=int(workers))
            di2 = np.atleast_1d(di2)
            ii2 = np.atleast_1d(ii2)
            keep2 = ii2 != i
            di, ii = di2[keep2], ii2[keep2]
        out_d[i] = di[:k]
        out_i[i] = ii[:k]
    return out_d, out_i


def _edges_from_knn(ki):
    """Build unique undirected graph edges from a kNN index matrix."""
    return topology.knn_edges_csr(ki)[0]


def _edges_and_distances_from_knn(ki, kd):
    """Build unique undirected edges and retain their first directed neighbour distances."""
    n, k = ki.shape
    if n == 0 or k == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    edges, _, _, first = topology.knn_edges_csr(ki, return_first=True)
    return edges, np.asarray(kd, dtype=np.float64).reshape(-1)[first]


def edges_to_csr(n, edges):
    """Convert undirected edge pairs into sorted symmetric CSR arrays."""
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    return topology.edges_to_csr(int(n), edges)


def _adjacency_from_edges(n, edges):
    """Convert edge pairs into a list-based adjacency representation."""
    ai, ao = edges_to_csr(n, edges)
    return _csr_to_adj(ai, ao)


def knn_density(X, k, d_hat=None, normalized=False, workers=-1):
    """Estimate density with normalized kNN volume following Loftsgaarden & Quesenberry (1965)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    k = min(int(k), n - 1)
    if k < 1:
        return np.ones(n)
    d, _ = _query_knn_euclidean(X, k, workers=workers)
    if not normalized:
        return 1.0 / np.maximum(d.mean(axis=1), 1e-12)
    if d_hat is None:
        d_hat = estimate_intrinsic_dim(X, workers=workers)
    rk = np.maximum(d[:, -1], 1e-12)
    V = np.pi ** (d_hat / 2.0) / _gamma(d_hat / 2.0 + 1.0)
    return k / np.maximum(n * V * np.power(rk, d_hat), 1e-300)


def _unit_weights(n_edges):
    # A read-only zero-stride view: no E-element float64 allocation is needed.
    """Return a zero-stride view representing unit edge weights."""
    return np.broadcast_to(np.ones(1, dtype=np.float64), (int(n_edges),))


def build_knn_graph(X, k, metric="euclidean", n_jobs=-1, store_knn=True):
    """Build a kNN graph with CSR adjacency and optional neighbour caches."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    N = X.shape[0]
    k = min(int(k), N - 1)
    if k < 1:
        raise ValueError("k must be >= 1 and X must contain at least two rows")
    if metric in ("euclidean", "l2"):
        kd, ki = _query_knn_euclidean(X, k, workers=n_jobs)
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=int(n_jobs)).fit(X)
        dist, idx = nn.kneighbors(X)
        kd = np.ascontiguousarray(dist[:, 1:], dtype=np.float64)
        ki = np.ascontiguousarray(idx[:, 1:], dtype=np.int64)
    density = 1.0 / np.maximum(kd.mean(axis=1), 1e-12)
    order = np.argsort(-density, kind="stable")
    edges, ai, ao = topology.knn_edges_csr(ki)
    weights = _unit_weights(len(edges))
    # In estimator mode the neighbour matrices are dead after the graph is built.
    if not store_knn:
        ki_out = None
        kd_out = None
    else:
        ki_out = ki
        kd_out = kd
    return KnnGraph(ki_out, kd_out, density, order, edges, weights, ai, ao, k, N)


def induced_subgraph(graph, S):
    """Extract the graph induced by a selected set of vertices."""
    S = np.ascontiguousarray(S, dtype=np.int64)
    if len(S) == graph.N and np.array_equal(S, np.arange(graph.N, dtype=np.int64)):
        return Subgraph(S, graph.density, graph.edges, graph.weights,
                        graph.adj_indices, graph.adj_indptr)
    le = topology.induced_edges_csr(graph.N, graph.edges, S)
    lw = _unit_weights(len(le))
    ai, ao = edges_to_csr(len(S), le)
    return Subgraph(S, graph.density[S], le, lw, ai, ao)


def log_density(base):
    """Convert positive density values into log-density filtration levels."""
    return np.log(np.maximum(np.asarray(base, dtype=np.float64), 1e-300))


def edge_residual(X, edges, ell):
    """Calculate edge-distance residuals for mutual-reachability filtration."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    if len(edges) == 0:
        return np.empty(0, dtype=np.float64)
    resid, _ = topology.edge_fields(X, edges, ell)
    return resid


def edge_levels(ell, edges, resid):
    """Calculate mutual-reachability filtration levels for graph edges."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.empty(0, dtype=np.float64)
    return topology.edge_levels_from_resid(ell, edges[:, 0], edges[:, 1], resid)


def adj_to_csr(adj):
    """Normalize list or CSR adjacency input into contiguous CSR arrays."""
    if isinstance(adj, tuple) and len(adj) == 2:
        ai = np.ascontiguousarray(adj[0], dtype=np.int64)
        ao = np.ascontiguousarray(adj[1], dtype=np.int64)
        return ai, ao
    n = len(adj)
    ao = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        ao[i + 1] = ao[i] + len(adj[i])
    ai = np.empty(ao[n], dtype=np.int64)
    pos = 0
    for i in range(n):
        row = np.asarray(adj[i], dtype=np.int64)
        ai[pos:pos + len(row)] = row
        pos += len(row)
    return ai, ao


def induced_edges(edges, idx, n):
    """Extract graph edges whose endpoints both belong to a selected subset."""
    return topology.induced_edges_csr(n, edges, idx)


def edges_to_adj(n, edges):
    """Convert edge pairs into a list-based adjacency representation."""
    return _adjacency_from_edges(n, edges)


def _adaptive_k_from_distances_python(knn_dist, k_min=4, sensitivity=2.5,
                              window=4, patience=2):
    """Select each point’s density neighbourhood from a persistent knee in its log-distance curve."""
    d = np.ascontiguousarray(knn_dist, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("knn_dist must be a 2D array")
    n, kmax = d.shape
    if kmax == 0:
        return np.zeros(n, dtype=np.int32)
    kmin = min(max(2, int(k_min)), kmax)
    w = min(max(2, int(window)), max(2, kmax - kmin + 1))
    patience = max(1, int(patience))
    if kmax <= kmin + w - 1:
        return np.full(n, kmax, dtype=np.int32)

    eps = np.finfo(np.float64).tiny
    y = np.log(np.maximum(d, eps))
    x = np.log(np.arange(1, kmax + 1, dtype=np.float64))

    # Baseline log-distance growth, fitted over k_min..k_min+window-1.
    b0 = kmin - 1
    b1 = min(kmax, kmin + w - 1)
    xb = x[b0:b1]
    xc = xb - xb.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-15:
        return np.full(n, kmax, dtype=np.int32)
    yb = y[:, b0:b1]
    baseline = (yb - yb.mean(axis=1, keepdims=True)) @ xc / denom
    baseline = np.maximum(baseline, 0.0)

    # Sliding-window slopes. Require a persistent rise to avoid reacting to one
    # unusually large Poisson spacing. A small absolute floor handles duplicate
    # coordinates / nearly-zero baseline slopes.
    threshold = np.maximum(baseline * float(sensitivity), baseline + 0.08)
    high_count = np.zeros(n, dtype=np.int16)
    chosen = np.full(n, kmax, dtype=np.int32)
    unresolved = np.ones(n, dtype=bool)
    for end in range(kmin + w, kmax + 1):
        start = end - w
        xx = x[start:end]
        xcc = xx - xx.mean()
        dd = float(np.dot(xcc, xcc))
        if dd <= 1e-15:
            continue
        yy = y[:, start:end]
        slope = ((yy - yy.mean(axis=1, keepdims=True)) @ xcc) / dd
        high = (slope > threshold) & unresolved
        high_count = np.where(high, high_count + 1, 0)
        hit = unresolved & (high_count >= patience)
        if np.any(hit):
            # Stop just before the detected transition. The subtraction is
            # conservative because the sliding windows overlap.
            knee = max(kmin, end - patience - 1)
            chosen[hit] = knee
            unresolved[hit] = False
    return np.clip(chosen, kmin, kmax).astype(np.int32, copy=False)


def adaptive_k_from_distances(knn_dist, k_min=4, sensitivity=2.5,
                              window=4, patience=2):
    """Select local neighbourhood sizes using the compiled kernel when available."""
    d = np.ascontiguousarray(knn_dist, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("knn_dist must be a 2D array")
    if _HAS_DENSITY_NATIVE:
        return _density_native.adaptive_k_from_distances(
            d, int(k_min), float(sensitivity), int(window), int(patience)
        )
    return _adaptive_k_from_distances_python(
        d, k_min, sensitivity, window, patience
    )


def _adaptive_knn_density_from_distances_python(knn_dist, k_min=4, sensitivity=2.5,
                                        window=4, patience=2,
                                        normalized=False, d_hat=None,
                                        k_bias_correction=True):
    """Calculate local density and adaptive K from one cached maximum-K distance query."""
    d = np.ascontiguousarray(knn_dist, dtype=np.float64)
    n, kmax = d.shape
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32)
    if kmax == 0:
        return np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.int32)
    ks = _adaptive_k_from_distances_python(d, k_min, sensitivity, window, patience)
    ks = np.maximum(ks, 1)
    rows = np.arange(n)
    if not normalized:
        cs = np.cumsum(d, axis=1)
        means = cs[rows, ks - 1] / ks
        density = 1.0 / np.maximum(means, 1e-12)
        # If K varies between points, inverse mean radius alone is not
        # comparable: its homogeneous-Poisson expectation depends on K.
        # A(k,d) removes this leading order-statistic scale dependence.
        if k_bias_correction and np.unique(ks).size > 1 and d_hat is not None:
            dh = max(float(d_hat), 1.0)
            jj = np.arange(1, kmax + 1, dtype=np.float64)
            moments = np.exp(gammaln(jj + 1.0 / dh) - gammaln(jj))
            a_k = np.cumsum(moments)[ks - 1] / ks
            density *= a_k
    else:
        dh = float(d_hat) if d_hat is not None else 1.0
        rk = np.maximum(d[rows, ks - 1], 1e-12)
        volume = np.pi ** (dh / 2.0) / _gamma(dh / 2.0 + 1.0)
        count_factor = np.maximum(ks - 1, 1) if k_bias_correction else ks
        density = count_factor / np.maximum(n * volume * np.power(rk, dh), 1e-300)
    return np.asarray(density, dtype=np.float64), ks


def adaptive_knn_density_from_distances(knn_dist, k_min=4, sensitivity=2.5,
                                        window=4, patience=2,
                                        normalized=False, d_hat=None,
                                        k_bias_correction=True):
    """Calculate local density and adaptive K using the native kernel when available."""
    d = np.ascontiguousarray(knn_dist, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("knn_dist must be a 2D array")
    if _HAS_DENSITY_NATIVE:
        return _density_native.adaptive_density(
            d, int(k_min), float(sensitivity), int(window), int(patience),
            bool(normalized), d_hat, bool(k_bias_correction)
        )
    return _adaptive_knn_density_from_distances_python(
        d, k_min, sensitivity, window, patience, normalized, d_hat,
        k_bias_correction
    )


def knn_scale_subgraph(graph, k):
    """Build one graph scale from a cached maximum-K neighbour query."""
    if graph.knn_idx is None:
        raise ValueError("KNN index cache is required for multi-scale graph construction")
    k = min(int(k), graph.knn_idx.shape[1], graph.N - 1)
    if k < 1:
        raise ValueError("scale K must be >= 1")
    edges, ai, ao = topology.knn_edges_csr(np.ascontiguousarray(graph.knn_idx[:, :k]))
    return Subgraph(np.arange(graph.N, dtype=np.int64), graph.density,
                    edges, _unit_weights(len(edges)), ai, ao)
