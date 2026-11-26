"""Provide input validation, random generators, and common utilities."""

import logging
import numpy as np
from scipy.stats import chi2


def get_logger(verbose=False):
    """Create a package logger with the requested verbosity."""
    log = logging.getLogger("entrap")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    return log


def validate_X(X):
    """Validate input features and return a contiguous finite-valued matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not np.all(np.isfinite(X)):
        raise ValueError("X has non-finite values")
    if X.shape[0] < 3:
        raise ValueError("need >= 3 samples")
    return np.ascontiguousarray(X)


def get_rng(seed):
    """Create a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def chi2_threshold(d, q):
    """Return the chi-square quantile used by membership rescue."""
    return float(chi2.ppf(q, df=d))
