# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Native H0 persistence, merge-tree, graph-filtration, and multi-scale kernels."""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport INFINITY, sqrt, log
from libc.stdint cimport int64_t, int32_t, uint32_t
cnp.import_array()

# Find a disjoint-set representative with path compression.
cdef inline int64_t _find(int64_t[::1] p, int64_t x) nogil:
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x

# Compute vertex-filtration H0 persistence with the elder-rule union-find sweep.
cdef int64_t _sweep(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao,
                    int64_t[::1] parent, unsigned char[::1] active, double[::1] birth,
                    int64_t[::1] root_mode, int64_t[::1] mode_parent, double[::1] mode_birth,
                    double[::1] mode_prom, int64_t[::1] vertex_mode,
                    double[::1] pb, double[::1] pd, int64_t[::1] order,
                    int64_t* roots, int64_t* npairs, bint full) nogil:
    cdef int64_t n = dens.shape[0], i, j, t, v, u, r, elder, mid, rm, em, nr, nm = 0, np_ = 0
    cdef double dv
    cdef bint found
    for i in range(n):
        parent[i] = i
        active[i] = 0
        root_mode[i] = -1
    for i in range(n):
        v = order[i]
        active[v] = 1
        dv = dens[v]
        birth[v] = dv
        nr = 0
        for j in range(ao[v], ao[v+1]):
            u = ai[j]
            if active[u]:
                r = _find(parent, u)
                if r == v:
                    continue
                found = False
                for t in range(nr):
                    if roots[t] == r:
                        found = True
                        break
                if not found:
                    roots[nr] = r
                    nr += 1
        if nr == 0:
            mid = nm
            nm += 1
            mode_parent[mid] = -1
            mode_birth[mid] = dv
            mode_prom[mid] = INFINITY
            root_mode[v] = mid
            if full:
                vertex_mode[v] = mid
        else:
            elder = roots[0]
            for t in range(1, nr):
                if birth[roots[t]] > birth[elder]:
                    elder = roots[t]
            em = root_mode[elder]
            if full:
                vertex_mode[v] = em
            for t in range(nr):
                r = roots[t]
                if r == elder:
                    continue
                rm = root_mode[r]
                mode_parent[rm] = em
                mode_prom[rm] = birth[r] - dv
                if full:
                    pb[np_] = birth[r]
                    pd[np_] = dv
                np_ += 1
                parent[r] = elder
            parent[v] = elder
    npairs[0] = np_
    return nm


cdef double _sweep_max(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao,
                       int64_t[::1] parent, unsigned char[::1] active,
                       double[::1] birth, int64_t[::1] order,
                       int64_t* roots) nogil:
    """Specialized vertex sweep: compute max finite bar without tree arrays."""
    cdef int64_t n = dens.shape[0], i, j, t, v, u, r, elder, nr
    cdef double dv, mx = 0.0, prom
    cdef bint found
    for i in range(n):
        parent[i] = i
        active[i] = 0
    for i in range(n):
        v = order[i]
        active[v] = 1
        dv = dens[v]
        birth[v] = dv
        nr = 0
        for j in range(ao[v], ao[v + 1]):
            u = ai[j]
            if active[u]:
                r = _find(parent, u)
                if r == v:
                    continue
                found = False
                for t in range(nr):
                    if roots[t] == r:
                        found = True
                        break
                if not found:
                    roots[nr] = r
                    nr += 1
        if nr > 0:
            elder = roots[0]
            for t in range(1, nr):
                if birth[roots[t]] > birth[elder]:
                    elder = roots[t]
            for t in range(nr):
                r = roots[t]
                if r == elder:
                    continue
                prom = birth[r] - dv
                if prom > mx:
                    mx = prom
                parent[r] = elder
            parent[v] = elder
    return mx


cdef double _sweep_edges_max(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb,
                             double[::1] elev, int64_t[::1] order_v,
                             int64_t[::1] order_e, int64_t[::1] parent,
                             double[::1] birth, int64_t[::1] root_mode) nogil:
    """Specialized edge sweep: compute max finite bar without merge-tree arrays."""
    cdef int64_t n = dens.shape[0], ne = ea.shape[0], i, iv = 0, ie = 0
    cdef int64_t v, e, a, b, ra, rb, elder, young, nm = 0
    cdef double mx = 0.0, prom
    cdef bint take_vertex
    for i in range(n):
        parent[i] = i
        root_mode[i] = -1
    while iv < n or ie < ne:
        if ie >= ne:
            take_vertex = True
        elif iv >= n:
            take_vertex = False
        else:
            take_vertex = dens[order_v[iv]] >= elev[order_e[ie]]
        if take_vertex:
            v = order_v[iv]
            iv += 1
            birth[v] = dens[v]
            root_mode[v] = nm
            nm += 1
        else:
            e = order_e[ie]
            ie += 1
            a = ea[e]
            b = eb[e]
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                continue
            if birth[ra] > birth[rb] or (birth[ra] == birth[rb] and root_mode[ra] < root_mode[rb]):
                elder = ra
                young = rb
            else:
                elder = rb
                young = ra
            prom = birth[young] - elev[e]
            if prom > mx:
                mx = prom
            parent[young] = elder
    return mx

def merge_tree_arrays(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao):
    """Construct H0 persistence pairs and vertex modes with the elder rule
    (Edelsbrunner et al., 2002)."""
    cdef int64_t n = dens.shape[0]
    if n == 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0),
            np.empty(0),
            np.empty((0, 2)),
        )
    cdef int64_t[::1] order = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef unsigned char[::1] active = np.empty(n, np.uint8)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t* roots = <int64_t*>malloc(n * sizeof(int64_t))
    if roots == NULL:
        raise MemoryError()
    cdef int64_t npairs = 0
    cdef int64_t nm = _sweep(
        dens, ai, ao, parent, active, birth, root_mode, mode_parent,
        mode_birth, mode_prom, vertex_mode, pb, pd, order, roots, &npairs, True
    )
    free(roots)
    pairs = np.column_stack([np.asarray(pb)[:npairs], np.asarray(pd)[:npairs]])
    return (np.asarray(vertex_mode), np.asarray(mode_parent)[:nm].copy(),
            np.asarray(mode_birth)[:nm].copy(), np.asarray(mode_prom)[:nm].copy(), pairs)

def distance_blocks(int64_t[::1] ai, int64_t[::1] ao, int64_t radius):
    """Partition a graph into connected blocks bounded by graph distance."""
    cdef int64_t n = ao.shape[0]-1, s, v, u, j, bid = 0, head, tail, qn
    cdef int64_t[::1] block = np.full(n, -1, dtype=np.int64)
    cdef int64_t[::1] depth = np.empty(n, np.int64)
    cdef int64_t[::1] queue = np.empty(n, np.int64)
    for s in range(n):
        if block[s] >= 0:
            continue
        head = 0
        tail = 0
        queue[tail] = s
        tail += 1
        block[s] = bid
        depth[s] = 0
        while head < tail:
            v = queue[head]
            head += 1
            if depth[v] >= radius:
                continue
            for j in range(ao[v], ao[v+1]):
                u = ai[j]
                if block[u] < 0:
                    block[u] = bid
                    depth[u] = depth[v]+1
                    queue[tail] = u
                    tail += 1
        bid += 1
    return np.asarray(block)

# Return the largest finite persistence value in a merge-tree array.
def _max_fin(double[::1] mp, int64_t nm):
    """Return the largest finite persistence value in a merge-tree array."""
    cdef int64_t i
    cdef double mx = 0.0
    for i in range(nm):
        if mp[i] < INFINITY and mp[i] > mx:
            mx = mp[i]
    return mx

def prominence_null(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao,
                    int64_t[::1] blocks, int64_t P, double alpha, object rng):
    """Estimate a block-permutation null threshold for maximum H0 persistence prominence."""
    cdef int64_t n = dens.shape[0], p, i, nm
    uniq = np.unique(np.asarray(blocks))
    block_idx = [np.where(np.asarray(blocks) == b)[0] for b in uniq]
    cdef double[::1] perm = np.empty(n)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef unsigned char[::1] active = np.empty(n, np.uint8)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t* roots = <int64_t*>malloc(n * sizeof(int64_t))
    if roots == NULL:
        raise MemoryError()
    cdef int64_t npairs = 0
    maxs = np.empty(P)
    dens_np = np.asarray(dens)      # ndarray views (shared memory): memoryviews
    perm_np = np.asarray(perm)      # do not support fancy indexing
    for p in range(P):
        perm_np[:] = dens_np
        for idx in block_idx:
            if len(idx) > 1:
                perm_np[idx] = dens_np[rng.permutation(idx)]
        order = np.argsort(-np.asarray(perm), kind="stable").astype(np.int64)
        nm = _sweep(perm, ai, ao, parent, active, birth, root_mode, mode_parent,
                    mode_birth, mode_prom, vertex_mode, pb, pd, order, roots, &npairs, False)
        maxs[p] = _max_fin(mode_prom, nm)
    free(roots)
    return float(np.quantile(maxs, 1.0-alpha))

# ---------------------------------------------------------------------------
# Edge-filtration sweep (mutual-reachability).  Vertices are born at dens[v], an
# undirected edge e=(ea,eb) enters at level elev[e] <= min(dens[ea], dens[eb]).
# Events are processed in decreasing level, at equal level vertices precede edges.
# Elder rule: the component with the larger birth survives (tie: earlier activation).
# Every vertex is its own mode, mode_prom[m] = birth - level of the edge that merges it.
# ---------------------------------------------------------------------------
# Compute H0 persistence from ordered vertex and edge filtration events.
cdef int64_t _sweep_edges(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb, double[::1] elev,
                          int64_t[::1] order_v, int64_t[::1] order_e,
                          int64_t[::1] parent, double[::1] birth, int64_t[::1] root_mode,
                          int64_t[::1] mode_parent, double[::1] mode_birth, double[::1] mode_prom,
                          int64_t[::1] vertex_mode, double[::1] pb, double[::1] pd,
                          int64_t* npairs, bint full) nogil:
    cdef int64_t n = dens.shape[0], ne = ea.shape[0], i, iv = 0, ie = 0, v, e, a, b, ra, rb
    cdef int64_t elder, young, nm = 0, np_ = 0
    cdef bint take_vertex
    for i in range(n):
        parent[i] = i
        root_mode[i] = -1
    while iv < n or ie < ne:
        if ie >= ne:
            take_vertex = True
        elif iv >= n:
            take_vertex = False
        else:
            take_vertex = dens[order_v[iv]] >= elev[order_e[ie]]
        if take_vertex:
            v = order_v[iv]
            iv += 1
            birth[v] = dens[v]
            mode_parent[nm] = -1
            mode_birth[nm] = dens[v]
            mode_prom[nm] = INFINITY
            root_mode[v] = nm
            if full:
                vertex_mode[v] = nm
            nm += 1
        else:
            e = order_e[ie]
            ie += 1
            a = ea[e]
            b = eb[e]
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                continue
            if birth[ra] > birth[rb] or (birth[ra] == birth[rb] and root_mode[ra] < root_mode[rb]):
                elder = ra
                young = rb
            else:
                elder = rb
                young = ra
            mode_parent[root_mode[young]] = root_mode[elder]
            mode_prom[root_mode[young]] = birth[young] - elev[e]
            if full:
                pb[np_] = birth[young]
                pd[np_] = elev[e]
            np_ += 1
            parent[young] = elder
    npairs[0] = np_
    return nm

def merge_tree_edges(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb, double[::1] elev):
    """Construct H0 persistence pairs from vertex and edge events with the elder
    rule (Edelsbrunner et al., 2002)."""
    cdef int64_t n = dens.shape[0]
    if n == 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0),
            np.empty(0),
            np.empty((0, 2)),
        )
    cdef int64_t[::1] order_v = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] order_e = np.argsort(-np.asarray(elev), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t npairs = 0
    cdef int64_t nm = _sweep_edges(dens, ea, eb, elev, order_v, order_e, parent, birth, root_mode,
                                   mode_parent, mode_birth, mode_prom, vertex_mode, pb, pd,
                                   &npairs, True)
    pairs = np.column_stack([np.asarray(pb)[:npairs], np.asarray(pd)[:npairs]])
    return (np.asarray(vertex_mode), np.asarray(mode_parent)[:nm].copy(),
            np.asarray(mode_birth)[:nm].copy(), np.asarray(mode_prom)[:nm].copy(), pairs)

def count_significant_edges(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb,
                            double[::1] elev, double tau):
    """Count persistent modes whose edge-filtration prominence meets the supplied threshold."""
    cdef int64_t n = dens.shape[0], i, nm, c = 0
    cdef int64_t[::1] order_v = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] order_e = np.argsort(-np.asarray(elev), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t npairs = 0
    nm = _sweep_edges(dens, ea, eb, elev, order_v, order_e, parent, birth, root_mode,
                      mode_parent, mode_birth, mode_prom, vertex_mode, pb, pd, &npairs, False)
    for i in range(nm):
        if mode_prom[i] >= tau:
            c += 1
    return c

def prominence_null_edges(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb, double[::1] resid,
                          int64_t[::1] blocks, int64_t P, double alpha, object rng):
    """Estimate edge-filtration prominence thresholds by permuting vertex
    densities within graph blocks."""
    cdef int64_t n = dens.shape[0], ne = ea.shape[0], p, e, nm
    cdef double m, M, t, pa, pbv
    uniq = np.unique(np.asarray(blocks))
    block_idx = [np.where(np.asarray(blocks) == b)[0] for b in uniq]
    cdef double[::1] perm = np.empty(n)
    cdef double[::1] elev = np.empty(ne)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t npairs = 0
    maxs = np.empty(P)
    dens_np = np.asarray(dens)
    perm_np = np.asarray(perm)
    for p in range(P):
        perm_np[:] = dens_np
        for idx in block_idx:
            if len(idx) > 1:
                perm_np[idx] = dens_np[rng.permutation(idx)]
        for e in range(ne):
            pa = perm[ea[e]]
            pbv = perm[eb[e]]
            if pa < pbv:
                m = pa
                M = pbv
            else:
                m = pbv
                M = pa
            t = 0.5*(M - m) + resid[e]
            if t < 0.0:
                elev[e] = m + t
            else:
                elev[e] = m + 0.0
        order_v = np.argsort(-perm_np, kind="stable").astype(np.int64)
        order_e = np.argsort(-np.asarray(elev), kind="stable").astype(np.int64)
        nm = _sweep_edges(perm, ea, eb, elev, order_v, order_e, parent, birth, root_mode,
                          mode_parent, mode_birth, mode_prom, vertex_mode, pb, pd, &npairs, False)
        maxs[p] = _max_fin(mode_prom, nm)
    return float(np.quantile(maxs, 1.0-alpha))

def count_significant(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao, double tau):
    """Count persistent modes whose prominence meets the supplied threshold."""
    cdef int64_t n = dens.shape[0], i, nm, c = 0
    cdef int64_t[::1] order = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef unsigned char[::1] active = np.empty(n, np.uint8)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef int64_t[::1] mode_parent = np.empty(n, np.int64)
    cdef double[::1] mode_birth = np.empty(n)
    cdef double[::1] mode_prom = np.empty(n)
    cdef int64_t[::1] vertex_mode = np.empty(n, np.int64)
    cdef double[::1] pb = np.empty(n)
    cdef double[::1] pd = np.empty(n)
    cdef int64_t* roots = <int64_t*>malloc(n * sizeof(int64_t))
    if roots == NULL:
        raise MemoryError()
    cdef int64_t npairs = 0
    nm = _sweep(dens, ai, ao, parent, active, birth, root_mode, mode_parent,
                mode_birth, mode_prom, vertex_mode, pb, pd, order, roots, &npairs, False)
    free(roots)
    for i in range(nm):
        if mode_prom[i] >= tau:
            c += 1
    return c

def assign_basins(int64_t[::1] vertex_mode, int64_t[::1] mode_parent,
                  double[::1] mode_prom, double tau):
    """Assign vertices to persistent mode basins at the requested filtration threshold."""
    cdef int64_t n = vertex_mode.shape[0], M = mode_prom.shape[0], i, m, t, plen
    cdef int64_t[::1] cache = np.full(M, -1, dtype=np.int64)
    cdef int64_t[::1] path = np.empty(M, np.int64)
    cdef int64_t[::1] out = np.empty(n, np.int64)
    for i in range(n):
        m = vertex_mode[i]
        plen = 0
        while mode_prom[m] < tau and mode_parent[m] != -1:
            if cache[m] != -1:
                m = cache[m]
                break
            path[plen] = m
            plen += 1
            m = mode_parent[m]
        for t in range(plen):
            cache[path[t]] = m
        out[i] = m
    return np.asarray(out)



def max_prominence_arrays(double[::1] dens, int64_t[::1] ai, int64_t[::1] ao):
    """Calculate maximum finite H0 prominence on a vertex filtration without
    building a merge tree."""
    cdef int64_t n = dens.shape[0]
    if n == 0:
        return 0.0
    cdef int64_t[::1] order = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef unsigned char[::1] active = np.empty(n, np.uint8)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t* roots = <int64_t*>malloc(n * sizeof(int64_t))
    if roots == NULL:
        raise MemoryError()
    cdef double ans
    with nogil:
        ans = _sweep_max(dens, ai, ao, parent, active, birth, order, roots)
    free(roots)
    return float(ans)


def max_prominence_edges(double[::1] dens, int64_t[::1] ea, int64_t[::1] eb, double[::1] elev):
    """Calculate maximum finite H0 prominence on an edge filtration without
    building a merge tree."""
    cdef int64_t n = dens.shape[0]
    if n == 0:
        return 0.0
    cdef int64_t[::1] order_v = np.argsort(-np.asarray(dens), kind="stable").astype(np.int64)
    cdef int64_t[::1] order_e = np.argsort(-np.asarray(elev), kind="stable").astype(np.int64)
    cdef int64_t[::1] parent = np.empty(n, np.int64)
    cdef double[::1] birth = np.empty(n)
    cdef int64_t[::1] root_mode = np.empty(n, np.int64)
    cdef double ans
    with nogil:
        ans = _sweep_edges_max(dens, ea, eb, elev, order_v, order_e,
                               parent, birth, root_mode)
    return float(ans)

# ---------------------------------------------------------------------------
# Memory-conscious graph kernels.  The public Python API keeps convenient
# adjacency-list compatibility, while ENTRAP's internal path stores CSR only.
# ---------------------------------------------------------------------------
def edges_to_csr(int64_t n, int64_t[:, ::1] edges):
    """Convert undirected edge pairs into symmetric compressed sparse row arrays."""
    cdef int64_t m = edges.shape[0]
    cdef int64_t i, a, b, row
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty(0, np.int64), np.zeros(1, np.int64)
    if m == 0:
        return np.empty(0, np.int64), np.zeros(n + 1, np.int64)
    if n > 3037000499:
        raise OverflowError("n is too large for int64 CSR key encoding")
    cdef cnp.ndarray[cnp.int64_t, ndim=1] keys_arr = np.empty(2 * m, dtype=np.int64)
    cdef int64_t[::1] keys = keys_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] degree_arr = np.zeros(n, dtype=np.int64)
    cdef int64_t[::1] degree = degree_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ao_arr = np.empty(n + 1, dtype=np.int64)
    cdef int64_t[::1] ao = ao_arr
    cdef int64_t[:, ::1] ev = edges
    with nogil:
        for i in range(m):
            a = ev[i, 0]
            b = ev[i, 1]
            keys[2 * i] = a * n + b
            keys[2 * i + 1] = b * n + a
    keys_arr.sort(kind="quicksort")
    with nogil:
        for i in range(2 * m):
            row = keys[i] // n
            if row < 0 or row >= n:
                continue
            degree[row] += 1
        ao[0] = 0
        for i in range(n):
            ao[i + 1] = ao[i] + degree[i]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ai_arr = np.empty(2 * m, dtype=np.int64)
    cdef int64_t[::1] ai = ai_arr
    with nogil:
        for i in range(2 * m):
            ai[i] = keys[i] % n
    return ai_arr, ao_arr


def knn_edges_csr(int64_t[:, ::1] knn_idx, bint return_first=False):
    """Convert kNN indices into unique undirected edges and symmetric CSR arrays."""
    cdef int64_t n = knn_idx.shape[0]
    cdef int64_t k = knn_idx.shape[1]
    cdef int64_t m = n * k
    cdef int64_t i, j, dst, a, b, e, ne, pos, previous_key
    if n == 0 or k == 0:
        base = (np.empty((0, 2), np.int64), np.empty(0, np.int64),
                np.zeros(n + 1, np.int64))
        if return_first:
            return base + (np.empty(0, np.int64),)
        return base
    if n > 3037000499:
        raise OverflowError("n is too large for int64 edge-key encoding")
    cdef cnp.ndarray[cnp.int64_t, ndim=1] keys_arr = np.empty(m, dtype=np.int64)
    cdef int64_t[::1] keys = keys_arr
    with nogil:
        for i in range(n):
            for j in range(k):
                dst = knn_idx[i, j]
                if i < dst:
                    a = i
                    b = dst
                else:
                    a = dst
                    b = i
                keys[i * k + j] = a * n + b
    cdef object unique_keys, first_indices, unique_key_buffer
    cdef int64_t[::1] uk
    if return_first:
        # This compatibility path needs the first directed occurrence for each
        # undirected pair, so preserve NumPy's first-index semantics exactly.
        unique_keys, first_indices = np.unique(keys_arr, return_index=True)
        unique_key_buffer = np.ascontiguousarray(unique_keys, dtype=np.int64)
        ne = unique_key_buffer.shape[0]
    else:
        # The common graph-construction path needs only sorted unique keys.
        # Sorting and compacting in-place avoids np.unique's hash-table and
        # temporary-array overhead while preserving its sorted output order.
        keys_arr.sort(kind="quicksort")
        ne = 0
        previous_key = -1
        with nogil:
            for e in range(m):
                if e == 0 or keys[e] != previous_key:
                    keys[ne] = keys[e]
                    ne += 1
                previous_key = keys[e]
        unique_key_buffer = keys_arr
        first_indices = None
    uk = unique_key_buffer
    cdef cnp.ndarray[cnp.int64_t, ndim=2] edge_arr = np.empty((ne, 2), dtype=np.int64)
    cdef int64_t[:, ::1] edge_view = edge_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] degree_arr = np.zeros(n, dtype=np.int64)
    cdef int64_t[::1] degree = degree_arr
    for e in range(ne):
        a = uk[e] // n
        b = uk[e] % n
        edge_view[e, 0] = a
        edge_view[e, 1] = b
        degree[a] += 1
        degree[b] += 1
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ao_arr = np.empty(n + 1, dtype=np.int64)
    cdef int64_t[::1] ao = ao_arr
    ao[0] = 0
    for i in range(n):
        ao[i + 1] = ao[i] + degree[i]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ai_arr = np.empty(2 * ne, dtype=np.int64)
    cdef int64_t[::1] ai = ai_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cursor_arr = ao_arr[:n].copy()
    cdef int64_t[::1] cursor = cursor_arr
    for e in range(ne):
        a = edge_view[e, 0]
        b = edge_view[e, 1]
        pos = cursor[a]
        ai[pos] = b
        cursor[a] += 1
        pos = cursor[b]
        ai[pos] = a
        cursor[b] += 1
    if return_first:
        return edge_arr, ai_arr, ao_arr, np.ascontiguousarray(first_indices, dtype=np.int64)
    return edge_arr, ai_arr, ao_arr


def edge_fields_from_dist(double[::1] ell, int64_t[:, ::1] edges,
                          double[::1] distances):
    """Calculate mutual-reachability residuals and filtration levels from edge distances."""
    cdef int64_t ne = edges.shape[0]
    cdef int64_t e, a, b
    cdef double da, db, lo, hi, d, r, t
    cdef cnp.ndarray[cnp.float64_t, ndim=1] resid_arr = np.empty(ne, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elev_arr = np.empty(ne, dtype=np.float64)
    cdef double[::1] resid = resid_arr
    cdef double[::1] elev = elev_arr
    with nogil:
        for e in range(ne):
            a = edges[e, 0]
            b = edges[e, 1]
            da = ell[a]
            db = ell[b]
            if da < db:
                lo = da
                hi = db
            else:
                lo = db
                hi = da
            d = distances[e]
            if d < 1e-300:
                d = 1e-300
            r = -log(d) - 0.5 * (da + db)
            resid[e] = r
            t = 0.5 * (hi - lo) + r
            elev[e] = lo + (t if t < 0.0 else 0.0)
    return resid_arr, elev_arr


def edge_fields(double[:, ::1] X, int64_t[:, ::1] edges,
                double[::1] ell):
    """Calculate mutual-reachability residuals and filtration levels from
    feature coordinates."""
    cdef int64_t ne = edges.shape[0]
    cdef int64_t d = X.shape[1]
    cdef int64_t e, a, b, j
    cdef double da, db, lo, hi, delta, sq, dist, r, t
    cdef cnp.ndarray[cnp.float64_t, ndim=1] resid_arr = np.empty(ne, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elev_arr = np.empty(ne, dtype=np.float64)
    cdef double[::1] resid = resid_arr
    cdef double[::1] elev = elev_arr
    with nogil:
        for e in range(ne):
            a = edges[e, 0]
            b = edges[e, 1]
            sq = 0.0
            for j in range(d):
                delta = X[a, j] - X[b, j]
                sq += delta * delta
            dist = sqrt(sq)
            if dist < 1e-300:
                dist = 1e-300
            da = ell[a]
            db = ell[b]
            if da < db:
                lo = da
                hi = db
            else:
                lo = db
                hi = da
            r = -log(dist) - 0.5 * (da + db)
            resid[e] = r
            t = 0.5 * (hi - lo) + r
            elev[e] = lo + (t if t < 0.0 else 0.0)
    return resid_arr, elev_arr


def edge_levels_from_resid(double[::1] ell, int64_t[::1] ea,
                           int64_t[::1] eb, double[::1] resid):
    """Recalculate edge levels for a changed density field while keeping edge residuals fixed."""
    cdef int64_t ne = ea.shape[0]
    cdef int64_t e, a, b
    cdef double da, db, lo, hi, t
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elev_arr = np.empty(ne, dtype=np.float64)
    cdef double[::1] elev = elev_arr
    with nogil:
        for e in range(ne):
            a = ea[e]
            b = eb[e]
            da = ell[a]
            db = ell[b]
            if da < db:
                lo = da
                hi = db
            else:
                lo = db
                hi = da
            t = 0.5 * (hi - lo) + resid[e]
            elev[e] = lo + (t if t < 0.0 else 0.0)
    return elev_arr


def induced_edges_csr(int64_t n, int64_t[:, ::1] edges, int64_t[::1] indices):
    """Extract edges induced by a vertex subset and remap endpoints to local indices."""
    cdef int64_t ne = edges.shape[0]
    cdef int64_t ns = indices.shape[0]
    cdef int64_t i, e, a, b, count = 0
    if n < 0 or ns > 2147483647:
        raise ValueError("unsupported induced-subgraph dimensions")
    cdef cnp.ndarray[cnp.int32_t, ndim=1] loc_arr = np.full(n, -1, dtype=np.int32)
    cdef int32_t[::1] loc = loc_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=2] out_arr = np.empty((ne, 2), dtype=np.int64)
    cdef int64_t[:, ::1] out = out_arr
    with nogil:
        for i in range(ns):
            loc[indices[i]] = <int32_t>i
        for e in range(ne):
            a = loc[edges[e, 0]]
            b = loc[edges[e, 1]]
            if a >= 0 and b >= 0:
                out[count, 0] = a
                out[count, 1] = b
                count += 1
    return out_arr[:count].copy()



def multiscale_graph_statistics(double[:, ::1] X,
                                int64_t[:, ::1] knn_idx,
                                double[::1] ell,
                                int64_t[::1] scales,
                                bint mreach=True):
    """Calculate maximum H0 prominence and component counts across candidate graph scales."""
    cdef int64_t n = X.shape[0]
    cdef int64_t dim = X.shape[1]
    cdef int64_t kmax = knn_idx.shape[1]
    cdef int64_t nscales = scales.shape[0]
    cdef int64_t max_m = n * kmax
    cdef int64_t s, k, m, i, j, dst, a, b, e, ne, pos
    cdef int64_t key, previous_key, ncomp, ra, rb
    cdef double sq, delta, dist, da, db, lo, hi, r, t

    if knn_idx.shape[0] != n or ell.shape[0] != n:
        raise ValueError("X, knn_idx, and ell must have matching row counts")
    if n == 0 or nscales == 0:
        return np.zeros(nscales, dtype=np.float64), np.zeros(nscales, dtype=np.int64)
    if n > 3037000499:
        raise OverflowError("n is too large for int64 edge-key encoding")
    for s in range(nscales):
        if scales[s] < 1 or scales[s] > kmax:
            raise ValueError("every scale must be between 1 and knn_idx.shape[1]")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] maxima_arr = np.empty(nscales, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] components_arr = np.empty(nscales, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] keys_arr = np.empty(max_m, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ea_arr = np.empty(max_m, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] eb_arr = np.empty(max_m, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] elev_arr = np.empty(max_m, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] parent_arr = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] birth_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] root_mode_arr = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] csr_keys_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] degree_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ai_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ao_arr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] active_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] roots_arr

    cdef int64_t[::1] csr_keys
    cdef int64_t[::1] degree
    cdef int64_t[::1] ai
    cdef int64_t[::1] ao
    cdef unsigned char[::1] active
    cdef int64_t[::1] roots

    if not mreach:
        csr_keys_arr = np.empty(2 * max_m, dtype=np.int64)
        degree_arr = np.empty(n, dtype=np.int64)
        ai_arr = np.empty(2 * max_m, dtype=np.int64)
        ao_arr = np.empty(n + 1, dtype=np.int64)
        active_arr = np.empty(n, dtype=np.uint8)
        roots_arr = np.empty(n, dtype=np.int64)
        csr_keys = csr_keys_arr
        degree = degree_arr
        ai = ai_arr
        ao = ao_arr
        active = active_arr
        roots = roots_arr

    cdef int64_t[::1] keys = keys_arr
    cdef int64_t[::1] ea_all = ea_arr
    cdef int64_t[::1] eb_all = eb_arr
    cdef double[::1] elev_all = elev_arr
    cdef int64_t[::1] parent = parent_arr
    cdef double[::1] birth = birth_arr
    cdef int64_t[::1] root_mode = root_mode_arr
    cdef int64_t[::1] order_v = np.argsort(-np.asarray(ell), kind="stable").astype(np.int64)
    cdef int64_t[::1] order_e
    cdef int64_t[::1] ea
    cdef int64_t[::1] eb
    cdef double[::1] elev
    cdef double[::1] maxima = maxima_arr
    cdef int64_t[::1] components = components_arr

    for s in range(nscales):
        k = scales[s]
        m = n * k
        # Encode each directed neighbour relation as its canonical undirected
        # edge key. Reusing a max-K buffer avoids per-scale key allocations.
        with nogil:
            for i in range(n):
                for j in range(k):
                    dst = knn_idx[i, j]
                    if i < dst:
                        a = i
                        b = dst
                    else:
                        a = dst
                        b = i
                    keys[i * k + j] = a * n + b

        keys_arr[:m].sort(kind="quicksort")
        ne = 0
        previous_key = -1
        with nogil:
            for i in range(m):
                key = keys[i]
                if i == 0 or key != previous_key:
                    keys[ne] = key
                    ne += 1
                previous_key = key

            # Decode sorted unique edges. For edge filtration, also compute
            # levels identically to topology.edge_fields().
            for e in range(ne):
                key = keys[e]
                a = key // n
                b = key % n
                ea_all[e] = a
                eb_all[e] = b
                if mreach:
                    sq = 0.0
                    for j in range(dim):
                        delta = X[a, j] - X[b, j]
                        sq += delta * delta
                    dist = sqrt(sq)
                    if dist < 1e-300:
                        dist = 1e-300
                    da = ell[a]
                    db = ell[b]
                    if da < db:
                        lo = da
                        hi = db
                    else:
                        lo = db
                        hi = da
                    r = -log(dist) - 0.5 * (da + db)
                    t = 0.5 * (hi - lo) + r
                    elev_all[e] = lo + (t if t < 0.0 else 0.0)

        ea = ea_all[:ne]
        eb = eb_all[:ne]
        elev = elev_all[:ne]

        # Count connected components with union-find. _sweep_edges_max resets
        # the same parent array before persistence, so no extra O(N) array is
        # needed for this second pass.
        with nogil:
            for i in range(n):
                parent[i] = i
            ncomp = n
            for e in range(ne):
                ra = _find(parent, ea[e])
                rb = _find(parent, eb[e])
                if ra != rb:
                    if ra < rb:
                        parent[rb] = ra
                    else:
                        parent[ra] = rb
                    ncomp -= 1
        components[s] = ncomp

        if mreach:
            order_e = np.argsort(-np.asarray(elev), kind="stable").astype(np.int64)
            with nogil:
                maxima[s] = _sweep_edges_max(
                    ell, ea, eb, elev, order_v, order_e,
                    parent, birth, root_mode,
                )
        else:
            # Build the same sorted symmetric CSR as edges_to_csr(), without
            # materialising intermediate Python edge/adjacency objects.
            with nogil:
                for i in range(n):
                    degree[i] = 0
                for e in range(ne):
                    a = ea_all[e]
                    b = eb_all[e]
                    csr_keys[2 * e] = a * n + b
                    csr_keys[2 * e + 1] = b * n + a
            csr_keys_arr[:2 * ne].sort(kind="quicksort")
            with nogil:
                for i in range(2 * ne):
                    key = csr_keys[i]
                    degree[key // n] += 1
                    ai[i] = key % n
                ao[0] = 0
                for i in range(n):
                    ao[i + 1] = ao[i] + degree[i]
                maxima[s] = _sweep_max(
                    ell, ai, ao, parent, active, birth, order_v, &roots[0]
                )

    return maxima_arr, components_arr


cdef void _sweep_ranked_edges(
    double[::1] dens,
    int64_t[::1] edge_keys,
    double[::1] elev,
    uint32_t[::1] edge_rank,
    int64_t[::1] order_v,
    int64_t[::1] order_e,
    int64_t k_limit,
    int64_t[::1] parent,
    double[::1] birth,
    int64_t[::1] root_mode,
    double* max_prominence,
    int64_t* component_count,
) noexcept nogil:
    """Sweep one ranked graph scale while implicitly constructing its maximum spanning forest."""
    cdef int64_t n = dens.shape[0]
    cdef int64_t ne = edge_keys.shape[0]
    cdef int64_t iv = 0
    cdef int64_t ie = 0
    cdef int64_t v, e, edge_key, a, b, ra, rb, elder, young
    cdef int64_t next_mode = 0
    cdef int64_t n_components = 0
    cdef double maximum = 0.0
    cdef double persistence
    cdef bint take_vertex

    for v in range(n):
        parent[v] = v
        root_mode[v] = -1

    while iv < n or ie < ne:
        while ie < ne and edge_rank[order_e[ie]] >= k_limit:
            ie += 1

        if iv >= n and ie >= ne:
            break
        if ie >= ne:
            take_vertex = True
        elif iv >= n:
            take_vertex = False
        else:
            take_vertex = dens[order_v[iv]] >= elev[order_e[ie]]

        if take_vertex:
            v = order_v[iv]
            iv += 1
            birth[v] = dens[v]
            root_mode[v] = next_mode
            next_mode += 1
            n_components += 1
            continue

        e = order_e[ie]
        ie += 1
        edge_key = edge_keys[e]
        a = edge_key // n
        b = edge_key % n
        ra = _find(parent, a)
        rb = _find(parent, b)

        if ra == rb:
            continue

        if (birth[ra] > birth[rb] or
                (birth[ra] == birth[rb] and root_mode[ra] < root_mode[rb])):
            elder = ra
            young = rb
        else:
            elder = rb
            young = ra

        persistence = birth[young] - elev[e]
        if persistence > maximum:
            maximum = persistence

        parent[young] = elder
        n_components -= 1

    max_prominence[0] = maximum
    component_count[0] = n_components


def multiscale_msf_statistics(
    double[:, ::1] X,
    int64_t[:, ::1] knn_idx,
    double[::1] ell,
    int64_t[::1] scales,
    bint mreach=False,
):
    """Compute exact multi-K H0 summaries from one ranked edge table and one filtration ordering."""
    cdef int64_t n = X.shape[0]
    cdef int64_t dim = X.shape[1]
    cdef int64_t kmax = knn_idx.shape[1]
    cdef int64_t nscales = scales.shape[0]
    cdef int64_t max_directed = n * kmax
    cdef int64_t s, k, i, j, dst, a, b, code, key, rank
    cdef int64_t ne = 0
    cdef int64_t previous_key = -1
    cdef int64_t e, coordinate
    cdef double squared_distance, delta, distance, da, db, lo, hi, residual, level
    cdef double maximum_value
    cdef int64_t component_value

    if knn_idx.shape[0] != n or ell.shape[0] != n:
        raise ValueError("X, knn_idx, and ell must have matching row counts")
    if n == 0 or nscales == 0:
        return (
            np.zeros(nscales, dtype=np.float64),
            np.zeros(nscales, dtype=np.int64),
        )
    if n > 3037000499:
        raise OverflowError("n is too large for int64 edge-key encoding")
    if n * n > (9223372036854775807 // (kmax + 1)):
        raise OverflowError("ranked edge-key encoding exceeds int64 capacity")
    for s in range(nscales):
        if scales[s] < 1 or scales[s] > kmax:
            raise ValueError("every scale must be between 1 and knn_idx.shape[1]")

    cdef cnp.ndarray[cnp.int64_t, ndim=1] ranked_keys_arr = np.empty(max_directed, dtype=np.int64)
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] edge_rank_arr = np.empty(max_directed, dtype=np.uint32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] edge_level_arr = np.empty(max_directed, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] order_e_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] order_v_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] parent_arr = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] root_mode_arr = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] birth_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] maxima_arr = np.empty(nscales, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] components_arr = np.empty(nscales, dtype=np.int64)

    cdef int64_t[::1] ranked_keys = ranked_keys_arr
    cdef uint32_t[::1] edge_rank = edge_rank_arr
    cdef double[::1] edge_level = edge_level_arr
    cdef int64_t[::1] order_v = np.argsort(-np.asarray(ell), kind="stable").astype(np.int64)
    cdef int64_t[::1] order_e
    cdef int64_t[::1] parent = parent_arr
    cdef int64_t[::1] root_mode = root_mode_arr
    cdef double[::1] birth = birth_arr
    cdef double[::1] maxima = maxima_arr
    cdef int64_t[::1] components = components_arr
    cdef int64_t sort_base = kmax + 1

    # Encode edge identity and directed neighbour rank in one sortable key.
    with nogil:
        for i in range(n):
            for j in range(kmax):
                dst = knn_idx[i, j]
                if i <= dst:
                    a = i
                    b = dst
                else:
                    a = dst
                    b = i
                key = a * n + b
                ranked_keys[i * kmax + j] = key * sort_base + j

    ranked_keys_arr.sort(kind="quicksort")

    # Deduplicate undirected pairs and retain their earliest directed K rank.
    with nogil:
        for i in range(max_directed):
            code = ranked_keys[i]
            key = code // sort_base
            rank = code - key * sort_base
            if i == 0 or key != previous_key:
                ranked_keys[ne] = key
                edge_rank[ne] = <uint32_t>rank
                ne += 1
            previous_key = key

        # Calculate each edge filtration level only once for the maximum-K graph.
        for e in range(ne):
            key = ranked_keys[e]
            a = key // n
            b = key % n
            da = ell[a]
            db = ell[b]
            if da < db:
                lo = da
                hi = db
            else:
                lo = db
                hi = da

            if mreach:
                squared_distance = 0.0
                for coordinate in range(dim):
                    delta = X[a, coordinate] - X[b, coordinate]
                    squared_distance += delta * delta
                distance = sqrt(squared_distance)
                if distance < 1e-300:
                    distance = 1e-300
                residual = -log(distance) - 0.5 * (da + db)
                level = lo + min(0.0, 0.5 * (hi - lo) + residual)
                edge_level[e] = level
            else:
                edge_level[e] = lo

    order_e_arr = np.argsort(-np.asarray(edge_level_arr[:ne]), kind="stable").astype(np.int64)
    order_e = order_e_arr

    for s in range(nscales):
        k = scales[s]
        _sweep_ranked_edges(
            ell,
            ranked_keys_arr[:ne],
            edge_level_arr[:ne],
            edge_rank_arr[:ne],
            order_v,
            order_e,
            k,
            parent,
            birth,
            root_mode,
            &maximum_value,
            &component_value,
        )
        maxima[s] = maximum_value
        components[s] = component_value

    return maxima_arr, components_arr
