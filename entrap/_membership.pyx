# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native Mahalanobis rescue and geometric tail-neighbour propagation."""
cimport numpy as cnp
import numpy as np

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t I64_t
ctypedef cnp.uint8_t U8_t


def rescue_assign(const double[:, ::1] X,
                  const cnp.int64_t[::1] labels_in,
                  const double[:, ::1] pi_pc,
                  const double[:, ::1] means,
                  const double[:, :, ::1] inv_covs,
                  const double[::1] prominence,
                  double threshold):
    """Assign eligible noise points using squared Mahalanobis distance (Mahalanobis, 1936)."""
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t C = means.shape[0]
    cdef Py_ssize_t x, c, a, b
    cdef double diff, tmp, d2, pc, prom, best_pc, best_prom
    cdef bint found
    cdef cnp.ndarray[cnp.int64_t, ndim=1] labels_arr = np.asarray(labels_in, dtype=np.int64).copy()
    cdef cnp.int64_t[::1] labels = labels_arr
    cdef int rescued = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] delta_arr = np.empty(d, dtype=np.float64)
    cdef double[::1] delta = delta_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] product_arr = np.empty(d, dtype=np.float64)
    cdef double[::1] product = product_arr

    with nogil:
        for x in range(n):
            if labels[x] >= 0:
                continue
            found = False
            best_pc = 0.0
            best_prom = 0.0
            for c in range(C):
                for a in range(d):
                    delta[a] = X[x, a] - means[c, a]
                # Match NumPy's diff @ inv_cov @ diff (row-vector first).
                for b in range(d):
                    tmp = 0.0
                    for a in range(d):
                        tmp += delta[a] * inv_covs[c, a, b]
                    product[b] = tmp
                d2 = 0.0
                for a in range(d):
                    d2 += product[a] * delta[a]
                if d2 <= threshold:
                    pc = pi_pc[x, c]
                    prom = prominence[c]
                    if (not found) or pc > best_pc or (pc == best_pc and prom > best_prom):
                        found = True
                        best_pc = pc
                        best_prom = prom
                        labels[x] = c
            if found:
                rescued += 1
    return labels_arr, rescued


def propagate_tail_candidates(cnp.ndarray[cnp.int64_t, ndim=1] labels_arr,
                              cnp.ndarray[cnp.uint8_t, ndim=1] tail_arr,
                              const cnp.uint8_t[::1] geometry_ok,
                              const double[:, ::1] keep_dist,
                              const cnp.int64_t[:, ::1] keep_nbr,
                              const cnp.uint8_t[::1] accepted_initial,
                              double link_limit,
                              int max_hops,
                              cnp.int64_t cluster_id):
    """Batch-synchronous graph propagation, preserving ascending candidate order."""
    cdef Py_ssize_t n = labels_arr.shape[0]
    cdef Py_ssize_t k = keep_nbr.shape[1]
    cdef Py_ssize_t i, j, t, count, hop
    cdef double best_dist
    cdef bint linked
    cdef cnp.int64_t[::1] labels = labels_arr
    cdef cnp.uint8_t[::1] tail = tail_arr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] accepted_arr = np.asarray(
        accepted_initial, dtype=np.uint8
    ).copy()
    cdef cnp.uint8_t[::1] accepted = accepted_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] newly_arr = np.empty(n, dtype=np.int64)
    cdef cnp.int64_t[::1] newly = newly_arr
    cdef int added = 0

    with nogil:
        for hop in range(max(1, max_hops)):
            count = 0
            for i in range(n):
                if labels[i] >= 0 or tail[i] != 0 or geometry_ok[i] == 0:
                    continue
                linked = False
                best_dist = 1.7976931348623157e308
                for j in range(k):
                    t = keep_nbr[i, j]
                    if accepted[t] != 0:
                        linked = True
                        if keep_dist[i, j] < best_dist:
                            best_dist = keep_dist[i, j]
                if linked and best_dist <= link_limit:
                    newly[count] = i
                    count += 1
            if count == 0:
                break
            for t in range(count):
                i = newly[t]
                labels[i] = cluster_id
                tail[i] = 1
                accepted[i] = 1
                added += 1
    return labels_arr, tail_arr, added
