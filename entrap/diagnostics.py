"""Provide compact topological diagnostics for persistence outputs."""

import numpy as np


def persistence_entropy(pairs):
    """Compute Shannon entropy from finite persistence lifetimes (Shannon, 1948)."""
    if pairs.size == 0:
        return 0.0
    ell = np.abs(pairs[:, 1]-pairs[:, 0])
    ell = ell[ell > 1e-12]
    L = ell.sum()
    if L <= 1e-12:
        return 0.0
    p = ell/L
    return float(-np.sum(p*np.log(p+1e-12)))
