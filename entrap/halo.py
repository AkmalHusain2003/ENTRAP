"""Apply a pointwise Poisson or fixed-fraction halo criterion to log-density values."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
try:
    from . import _halo as _halo_native
    _HAS_HALO_NATIVE = True
except Exception:
    _halo_native = None
    _HAS_HALO_NATIVE = False


def _stream_poisson_g(rng, M, kmax, dimension, selected_k=None, chunk_size=65536):
    """Generate Poisson halo reference statistics in bounded memory chunks."""
    M = int(M)
    kmax = int(kmax)
    out = np.empty(M, dtype=np.float64)
    for start in range(0, M, int(chunk_size)):
        stop = min(M, start + int(chunk_size))
        draws = np.ascontiguousarray(
            rng.exponential(1.0, size=(stop - start, kmax)), dtype=np.float64
        )
        local_k = None if selected_k is None else np.ascontiguousarray(
            selected_k[start:stop], dtype=np.int32
        )
        out[start:stop] = _halo_native.poisson_g_values(
            draws, int(dimension), local_k
        )
    return out


@lru_cache(maxsize=64)
def poisson_null_quantiles(k, d, alpha, M=200000, seed=12345):
    """Estimate fixed-K Poisson halo reference quantiles by Monte Carlo."""
    rng = np.random.default_rng(seed)
    if _HAS_HALO_NATIVE:
        g = _stream_poisson_g(rng, M, int(k), int(d), selected_k=None)
    else:
        gam = np.cumsum(rng.exponential(1.0, size=(M, int(k))), axis=1)
        g = -np.log(np.mean(gam ** (1.0/float(d)), axis=1))
    return float(np.median(g)), float(np.quantile(g, 1.0-alpha)), float(g.std())



@lru_cache(maxsize=64)
def poisson_null_quantiles_variable(k_values, d, alpha, M=200000, seed=12345):
    """Estimate Poisson halo quantiles for an empirical mixture of local K values."""
    ks = np.asarray(tuple(int(v) for v in k_values), dtype=np.int32)
    if ks.size == 0 or np.any(ks < 1):
        raise ValueError("k_values must contain positive integers")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ks, size=int(M), replace=True)
    kmax = int(ks.max())
    if _HAS_HALO_NATIVE:
        g = _stream_poisson_g(
            rng, int(M), kmax, int(d),
            selected_k=np.asarray(chosen, dtype=np.int32)
        )
    else:
        gaps = np.cumsum(rng.exponential(1.0, size=(int(M), kmax)), axis=1)
        radial = gaps ** (1.0 / float(d))
        cumulative = np.cumsum(radial, axis=1)
        means = cumulative[np.arange(int(M)), chosen - 1] / chosen
        g = -np.log(np.maximum(means, 1e-300))
    return float(np.median(g)), float(np.quantile(g, 1.0-alpha)), float(g.std())


def _two_means_1d(x, iters=100):
    """Split one-dimensional density values into two groups using iterative means."""
    lo, hi = float(x.min()), float(x.max())
    for _ in range(iters):
        thr = 0.5*(lo+hi)
        a = x[x <= thr]
        b = x[x > thr]
        if len(a) == 0 or len(b) == 0:
            break
        nlo, nhi = float(a.mean()), float(b.mean())
        if abs(nlo-lo) < 1e-12 and abs(nhi-hi) < 1e-12:
            break
        lo, hi = nlo, nhi
    return 0.5*(lo+hi)


def poisson_core(ell, k, d, alpha=0.001, kappa=2.5, contrast=2.0, min_frac=0.03):
    """Identify dense core points using a homogeneous-Poisson reference distribution."""
    ell = np.asarray(ell, dtype=np.float64)
    n = len(ell)
    if np.ndim(k) == 0:
        g50, ghi, sg = poisson_null_quantiles(int(k), int(d), float(alpha))
    else:
        kvals = tuple(sorted(int(v) for v in np.asarray(k).ravel()))
        g50, ghi, sg = poisson_null_quantiles_variable(kvals, int(d), float(alpha))
    all_core = np.ones(n, bool)
    info = {"ok": False, "c": float("nan"), "t": float("nan"), "n_noise": 0, "reason": ""}
    if n < 8 or float(ell.max() - ell.min()) < 1e-12:
        info["reason"] = "degenerate"
        return all_core, info
    N = ell <= _two_means_1d(ell)
    c = t = float("nan")
    for _ in range(100):
        if N.sum() < 3:
            break
        c = float(np.median(ell[N]) - g50)
        t = c + ghi
        N2 = ell <= t
        if np.array_equal(N2, N):
            break
        N = N2
    info.update(c=c, t=t, n_noise=int(N.sum()))
    if N.sum() < max(3, int(np.ceil(min_frac*n))):
        info["reason"] = "too few background points"
        return all_core, info
    if (~N).sum() < 3:
        info["reason"] = "no dense component"
        return all_core, info
    mad = 1.4826*float(np.median(np.abs(ell[N] - np.median(ell[N]))))
    if mad > kappa*sg:
        info["reason"] = "background spread not Poisson-like"
        return all_core, info
    if (
        float(np.median(ell[~N])) - (c + g50)
        < contrast * (ghi - g50)
    ):  # core median vs background median
        info["reason"] = "no density contrast"
        return all_core, info
    info["ok"] = True
    return ~N, info


def fixed_core(base, eta):
    """Select core points using a fixed density-range fraction."""
    b = np.asarray(base, dtype=np.float64)
    span = float(b.max() - b.min())
    if span <= 1e-12:
        return np.ones(len(b), bool)
    return (b - b.min())/span >= eta


def halo_core(X, ell_dist, base, cfg, k):
    """Apply the configured halo filter to one hierarchical node."""
    if cfg.halo_mode == "fixed":
        return fixed_core(base, cfg.halo_eta)
    d = int(cfg.halo_dim) if cfg.halo_dim else int(np.asarray(X).shape[1])
    core, _ = poisson_core(
        ell_dist, k, max(d, 1), cfg.halo_alpha, cfg.halo_kappa, cfg.halo_contrast
    )
    if core.sum() < cfg.min_cluster_size:
        return np.ones(len(core), bool)  # Never remove a cluster already accepted by persistence.
    return core
