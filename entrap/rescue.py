"""Reassign selected noise points using cluster-level Gaussian models."""

import numpy as np
from .utils import chi2_threshold
try:
    from . import _membership as _membership_native
    _HAS_MEMBERSHIP_NATIVE = True
except Exception:
    _membership_native = None
    _HAS_MEMBERSHIP_NATIVE = False


def _gauss(X, idx, ridge):
    """Fit a regularized Gaussian model to the selected cluster members."""
    p = X[idx]
    mu = p.mean(0)
    cov = np.cov(p, rowvar=False)+ridge*np.eye(X.shape[1])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.eye(X.shape[1])/(np.trace(cov)/X.shape[1]+1e-6)
    return mu, inv


def rescue(X, labels, pi_pc, basins, proms, cfg):
    """Reassign eligible noise points using squared Mahalanobis distance (Mahalanobis, 1936)."""
    labels = labels.copy()
    C = len(basins)
    if C == 0:
        return labels, 0
    thr = chi2_threshold(X.shape[1], cfg.alpha_gate)
    gauss = {c: _gauss(X, b, cfg.ridge) for c, b in basins.items()}
    if _HAS_MEMBERSHIP_NATIVE:
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        labelc = np.ascontiguousarray(labels, dtype=np.int64)
        pic = np.ascontiguousarray(pi_pc, dtype=np.float64)
        means = np.empty((C, Xc.shape[1]), dtype=np.float64)
        invs = np.empty((C, Xc.shape[1], Xc.shape[1]), dtype=np.float64)
        prominence = np.empty(C, dtype=np.float64)
        for c in range(C):
            mu, inv = gauss[c]
            means[c] = mu
            invs[c] = inv
            prominence[c] = float(proms.get(c, 0.0))
        return _membership_native.rescue_assign(
            Xc, labelc, pic, means, invs, prominence, float(thr)
        )

    rescued = 0
    for x in np.where(labels < 0)[0]:
        best = None
        for c in range(C):
            mu, inv = gauss[c]
            diff = X[x]-mu
            D2 = float(diff @ inv @ diff)
            if D2 <= thr:
                key = (pi_pc[x, c], proms.get(c, 0.0))
                if best is None or key > best[0]:
                    best = (key, c)
        if best is not None:
            labels[x] = best[1]
            rescued += 1
    return labels, rescued
