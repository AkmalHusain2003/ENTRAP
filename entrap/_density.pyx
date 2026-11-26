# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native adaptive-neighbourhood and kNN-density kernels with Python fallbacks."""
from libc.math cimport log, fmax, pow
cimport numpy as cnp
import numpy as np
from scipy.special import gammaln, gamma as scipy_gamma

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t I32_t


def adaptive_k_from_distances(const double[:, ::1] dist,
                              int k_min=4,
                              double sensitivity=2.5,
                              int window=4,
                              int patience=2):
    """Select each sample local K from the persistent slope transition in its log-distance curve."""
    cdef Py_ssize_t n = dist.shape[0]
    cdef Py_ssize_t kmax = dist.shape[1]
    cdef Py_ssize_t i, j, end, start, b0, b1, count
    cdef int kmin, w, pat, knee
    cdef double eps = np.finfo(np.float64).tiny
    cdef double denom = 0.0, meanx = 0.0, meany = 0.0
    cdef double xval, yval, xc, yy, numerator, slope
    cdef double baseline, threshold, high_count
    cdef double base_mean, base_num, base
    cdef double win_denom, win_mean_x, win_xc
    cdef cnp.ndarray[cnp.float64_t, ndim=1] baseline_arr
    cdef double[::1] baseline_view
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xcenter
    cdef cnp.ndarray[cnp.int32_t, ndim=1] out_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] high_arr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] unresolved_arr
    cdef double[::1] x, xc_arr
    cdef cnp.int32_t[::1] out, high
    cdef cnp.uint8_t[::1] unresolved

    if kmax == 0:
        return np.zeros(n, dtype=np.int32)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    kmin = min(max(2, k_min), kmax)
    w = min(max(2, window), max(2, kmax - kmin + 1))
    pat = max(1, patience)
    out_arr = np.full(n, kmax, dtype=np.int32)
    if kmax <= kmin + w - 1:
        return out_arr

    x_arr = np.empty(kmax, dtype=np.float64)
    x = x_arr
    for j in range(kmax):
        x[j] = log(j + 1.0)

    b0 = kmin - 1
    b1 = min(kmax, kmin + w - 1)
    count = b1 - b0
    meanx = 0.0
    for j in range(b0, b1):
        meanx += x[j]
    meanx /= count
    xcenter = np.empty(count, dtype=np.float64)
    xc_arr = xcenter
    denom = 0.0
    for j in range(count):
        xc_arr[j] = x[b0 + j] - meanx
        denom += xc_arr[j] * xc_arr[j]
    if denom <= 1e-15:
        return out_arr

    high_arr = np.zeros(n, dtype=np.int32)
    unresolved_arr = np.ones(n, dtype=np.uint8)
    baseline_arr = np.empty(n, dtype=np.float64)
    baseline_view = baseline_arr
    out = out_arr
    high = high_arr
    unresolved = unresolved_arr

    with nogil:
        # Baseline slope for every row.
        for i in range(n):
            base_mean = 0.0
            for j in range(b0, b1):
                yval = log(fmax(dist[i, j], eps))
                base_mean += yval
            base_mean /= count
            base_num = 0.0
            for j in range(count):
                yval = log(fmax(dist[i, b0 + j], eps))
                base_num += (yval - base_mean) * xc_arr[j]
            base = base_num / denom
            if base < 0.0:
                base = 0.0
            baseline_view[i] = base

        # Same overlapping-window/patience rule as the NumPy reference.
        for end in range(kmin + w, kmax + 1):
            start = end - w
            win_mean_x = 0.0
            for j in range(start, end):
                win_mean_x += x[j]
            win_mean_x /= w
            win_denom = 0.0
            for j in range(start, end):
                win_xc = x[j] - win_mean_x
                win_denom += win_xc * win_xc
            if win_denom <= 1e-15:
                continue
            knee = end - pat - 1
            if knee < kmin:
                knee = kmin
            for i in range(n):
                if unresolved[i] == 0:
                    high[i] = 0
                    continue
                meany = 0.0
                for j in range(start, end):
                    meany += log(fmax(dist[i, j], eps))
                meany /= w
                numerator = 0.0
                for j in range(start, end):
                    win_xc = x[j] - win_mean_x
                    yval = log(fmax(dist[i, j], eps))
                    numerator += (yval - meany) * win_xc
                slope = numerator / win_denom
                base = baseline_view[i]
                threshold = fmax(base * sensitivity, base + 0.08)
                if slope > threshold:
                    high[i] += 1
                else:
                    high[i] = 0
                if high[i] >= pat:
                    out[i] = knee
                    unresolved[i] = 0

        for i in range(n):
            if out[i] < kmin:
                out[i] = kmin
            elif out[i] > kmax:
                out[i] = kmax
    return out_arr


def adaptive_density(const double[:, ::1] dist,
                     int k_min=4,
                     double sensitivity=2.5,
                     int window=4,
                     int patience=2,
                     bint normalized=False,
                     d_hat=None,
                     bint k_bias_correction=True):
    """Fused adaptive-K + density calculation from a cached max-K distance matrix."""
    cdef Py_ssize_t n = dist.shape[0]
    cdef Py_ssize_t kmax = dist.shape[1]
    cdef Py_ssize_t i, j, k
    cdef double dh, volume, rk, denom, total, mean_r, correction
    cdef bint apply_bias = False
    cdef cnp.ndarray[cnp.int32_t, ndim=1] ks_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] density_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] correction_arr
    cdef cnp.int32_t[::1] ks
    cdef double[::1] density, corr

    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32)
    if kmax == 0:
        return np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.int32)

    ks_arr = adaptive_k_from_distances(dist, k_min, sensitivity, window, patience)
    ks = ks_arr
    density_arr = np.empty(n, dtype=np.float64)
    density = density_arr
    dh = float(d_hat) if d_hat is not None else 1.0

    if normalized:
        volume = (np.pi ** (dh / 2.0)) / float(scipy_gamma(dh / 2.0 + 1.0))
        with nogil:
            for i in range(n):
                k = ks[i]
                if k < 1:
                    k = 1
                rk = dist[i, k - 1]
                if rk < 1e-12:
                    rk = 1e-12
                if k_bias_correction:
                    denom = k - 1
                    if denom < 1:
                        denom = 1
                else:
                    denom = k
                total = n * volume * pow(rk, dh)
                if total < 1e-300:
                    total = 1e-300
                density[i] = denom / total
    else:
        apply_bias = bool(k_bias_correction and d_hat is not None and
                          int(ks_arr.min()) != int(ks_arr.max()))
        if apply_bias:
            jj = np.arange(1, kmax + 1, dtype=np.float64)
            moments = np.exp(gammaln(jj + 1.0 / max(dh, 1.0)) - gammaln(jj))
            correction_arr = (
                np.cumsum(moments, dtype=np.float64)
                / np.arange(1, kmax + 1, dtype=np.float64)
            )
            corr = correction_arr
        with nogil:
            for i in range(n):
                k = ks[i]
                if k < 1:
                    k = 1
                total = 0.0
                for j in range(k):
                    total += dist[i, j]
                mean_r = total / k
                if mean_r < 1e-12:
                    mean_r = 1e-12
                density[i] = 1.0 / mean_r
                if apply_bias:
                    density[i] *= corr[k - 1]
    return density_arr, ks_arr
