# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Memory-bounded Poisson halo reference transforms.

Fixed-K draws use NumPy's vectorized ufuncs in bounded chunks because those
outperform scalar libm calls. Variable-K draws use a fused Cython row kernel,
avoiding the three full M-by-K intermediate matrices from the reference implementation.
"""
from libc.math cimport pow, log, fmax, sqrt, cbrt
cimport numpy as cnp
import numpy as np


def _poisson_g_variable_loop(const double[:, ::1] draws,
                             const cnp.int32_t[::1] selected_k,
                             int dimension):
    """Transform exponential Poisson arrivals into variable-K radial-mean statistics."""
    cdef Py_ssize_t m = draws.shape[0]
    cdef Py_ssize_t kmax = draws.shape[1]
    cdef Py_ssize_t i, j, stop
    cdef double gamma_sum, radial_sum, radial, mean_radius, exponent
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty(m, dtype=np.float64)
    cdef double[::1] out = out_arr
    exponent = 1.0 / max(1, dimension)
    with nogil:
        for i in range(m):
            stop = selected_k[i]
            if stop < 1:
                stop = 1
            if stop > kmax:
                stop = kmax
            gamma_sum = 0.0
            radial_sum = 0.0
            for j in range(stop):
                gamma_sum += draws[i, j]
                if dimension == 1:
                    radial = gamma_sum
                elif dimension == 2:
                    radial = sqrt(gamma_sum)
                elif dimension == 3:
                    radial = cbrt(gamma_sum)
                elif dimension == 4:
                    radial = sqrt(sqrt(gamma_sum))
                elif dimension == 6:
                    radial = cbrt(sqrt(gamma_sum))
                else:
                    radial = pow(gamma_sum, exponent)
                radial_sum += radial
            mean_radius = radial_sum / stop
            out[i] = -log(fmax(mean_radius, 1e-300))
    return out_arr


def poisson_g_values(exponential_draws, int dimension, selected_k=None):
    """Transform one bounded chunk of exponential increments into g values."""
    draws = np.asarray(exponential_draws, dtype=np.float64, order="C")
    if draws.ndim != 2:
        raise ValueError("exponential_draws must be a 2D array")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if selected_k is not None:
        chosen = np.asarray(selected_k, dtype=np.int32)
        if chosen.ndim != 1 or chosen.shape[0] != draws.shape[0]:
            raise ValueError("selected_k must contain one K value per row")
        if np.any(chosen < 1) or np.any(chosen > draws.shape[1]):
            raise ValueError("selected_k values must be within [1, kmax]")
        return _poisson_g_variable_loop(draws, chosen, dimension)
    gamma_arrivals = np.cumsum(draws, axis=1)
    radial = np.power(gamma_arrivals, 1.0 / float(dimension))
    means = np.mean(radial, axis=1)
    return -np.log(np.maximum(means, 1e-300))
