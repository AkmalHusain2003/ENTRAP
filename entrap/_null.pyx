# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native CSR breadth-first traversal for block-scale estimation."""
from libc.math cimport sqrt
cimport numpy as cnp
import numpy as np

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t I64_t


def estimate_block_scale(const double[::1] density,
                         const cnp.int64_t[::1] indices,
                         const cnp.int64_t[::1] indptr,
                         const cnp.int64_t[::1] sources,
                         int max_hop=6):
    """Estimate graph correlation length with native CSR traversal and online Pearson moments."""
    cdef Py_ssize_t n = density.shape[0]
    cdef Py_ssize_t ns = sources.shape[0]
    cdef Py_ssize_t sidx, h, q, v, pos, u, next_count, cur_count
    cdef Py_ssize_t mark
    cdef double xval, yval, dx, dy, inv_count, corr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] visited_arr = np.zeros(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] front_a = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] front_b = np.empty(n, dtype=np.int64)
    cdef cnp.int64_t[::1] visited = visited_arr
    cdef cnp.int64_t[::1] front1 = front_a
    cdef cnp.int64_t[::1] front2 = front_b
    cdef cnp.ndarray[cnp.int64_t, ndim=1] count_arr = np.zeros(max_hop, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] meanx_arr = np.zeros(max_hop, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] meany_arr = np.zeros(max_hop, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] m2x_arr = np.zeros(max_hop, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] m2y_arr = np.zeros(max_hop, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] cov_arr = np.zeros(max_hop, dtype=np.float64)
    cdef cnp.int64_t[::1] counts = count_arr
    cdef double[::1] meanx = meanx_arr
    cdef double[::1] meany = meany_arr
    cdef double[::1] m2x = m2x_arr
    cdef double[::1] m2y = m2y_arr
    cdef double[::1] cov = cov_arr
    cdef int result = max(1, max_hop)

    if n < 4:
        return 1
    if ns == 0 or max_hop <= 0:
        return result

    with nogil:
        for sidx in range(ns):
            v = sources[sidx]
            mark = sidx + 1
            visited[v] = mark
            front1[0] = v
            cur_count = 1
            xval = density[v]
            for h in range(max_hop):
                next_count = 0
                for q in range(cur_count):
                    v = front1[q]
                    for pos in range(indptr[v], indptr[v + 1]):
                        u = indices[pos]
                        if visited[u] != mark:
                            visited[u] = mark
                            front2[next_count] = u
                            next_count += 1
                            yval = density[u]
                            counts[h] += 1
                            inv_count = 1.0 / counts[h]
                            dx = xval - meanx[h]
                            meanx[h] += dx * inv_count
                            dy = yval - meany[h]
                            meany[h] += dy * inv_count
                            m2x[h] += dx * (xval - meanx[h])
                            m2y[h] += dy * (yval - meany[h])
                            cov[h] += dx * (yval - meany[h])
                cur_count = next_count
                tmp = front1
                front1 = front2
                front2 = tmp
                if cur_count == 0:
                    break

        for h in range(max_hop):
            if counts[h] < 5:
                continue
            if sqrt(m2x[h] / counts[h]) < 1e-12 or sqrt(m2y[h] / counts[h]) < 1e-12:
                continue
            corr = cov[h] / sqrt(m2x[h] * m2y[h])
            if corr < 0.36787944117144233:
                result = h + 1
                break
    return result
