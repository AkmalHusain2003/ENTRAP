"""Strict igraph/leidenalg integration used by ENTRAP.

The topological merge-tree supplies an initial partition and a target number of
modes. Leiden then optimises that partition with the Constant Potts Model (CPM)
on the same undirected, simple kNN graph.

Resolution search deliberately scans a deterministic set of candidate gamma
values instead of binary-searching community count: Leiden is a stochastic
heuristic, so the observed number of communities is not guaranteed to be a
monotone function of gamma across independently optimised runs.
"""
from __future__ import annotations

import math
import numpy as np

from .utils import get_logger

try:  # Optional at import time, required by default when cfg.use_leiden=True.
    import igraph as ig
    import leidenalg as la
    _HAS = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local optional packages
    ig = None
    la = None
    _HAS = False
    _IMPORT_ERROR = exc


def has_leiden() -> bool:
    """Check whether both igraph and leidenalg are available."""
    return bool(_HAS)


def backend_info() -> dict:
    """Return Leiden backend availability and installed version information."""
    if not _HAS:
        return {
            "available": False,
            "igraph_version": None,
            "leidenalg_version": None,
            "error": None if _IMPORT_ERROR is None else repr(_IMPORT_ERROR),
        }
    return {
        "available": True,
        "igraph_version": getattr(ig, "__version__", "unknown"),
        "leidenalg_version": getattr(la, "__version__", "unknown"),
        "error": None,
    }


def require_leiden() -> None:
    """Require the external Leiden backend before clustering starts."""
    if not _HAS:
        detail = f" Import error: {_IMPORT_ERROR!r}." if _IMPORT_ERROR else ""
        raise ImportError(
            "ENTRAP is configured with use_leiden=True, but the igraph and/or "
            "leidenalg backend is unavailable. Install the required packages "
            "with `python -m pip install 'igraph>=0.11' 'leidenalg>=0.10'`. "
            "For an intentional topology-only run set use_leiden=False." + detail
        )


def _canonical_graph_arrays(n: int, edges, weights):
    """Validate and canonicalize an undirected simple graph."""
    n = int(n)
    if n < 1:
        raise ValueError("Leiden graph must contain at least one vertex")
    if n > 3037000499:
        raise OverflowError("n is too large for int64 canonical edge-key encoding")
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        e = np.empty((0, 2), dtype=np.int64)
    else:
        if e.ndim != 2 or e.shape[1] != 2:
            raise ValueError("edges must have shape (n_edges, 2)")
        if np.any(e < 0) or np.any(e >= n):
            raise ValueError("edge endpoint is outside [0, n)")
        e = np.sort(np.ascontiguousarray(e), axis=1)

    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(e):
            raise ValueError("weights length must match edges length")
        if not np.all(np.isfinite(w)):
            raise ValueError("Leiden edge weights must be finite")
        if np.any(w <= 0.0):
            raise ValueError("CPM kNN edge weights must be strictly positive")

    keep = e[:, 0] != e[:, 1]
    e = e[keep]
    if w is not None:
        w = w[keep]

    if len(e) == 0:
        return np.empty((0, 2), dtype=np.int64), (None if w is None else np.empty(0))

    # Canonical sorted pairs make an undirected simple graph deterministic.
    keys = e[:, 0] * np.int64(n) + e[:, 1]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    if w is not None and len(unique_keys) != len(keys):
        reduced = np.full(len(unique_keys), -np.inf, dtype=np.float64)
        np.maximum.at(reduced, inverse, w)
        w = reduced
    elif w is not None:
        # np.unique sorts keys, preserve aligned weight order.
        order = np.argsort(keys, kind="stable")
        w = w[order]

    out = np.column_stack((unique_keys // n, unique_keys % n)).astype(np.int64, copy=False)
    if w is not None and len(w) == len(keys) and len(unique_keys) == len(keys):
        # The no-duplicate branch above sorted weights to match unique_keys.
        pass
    elif w is not None and len(w) != len(unique_keys):
        raise RuntimeError("internal edge-weight canonicalisation error")
    return np.ascontiguousarray(out), (None if w is None else np.ascontiguousarray(w))


def _make_igraph(n: int, edges: np.ndarray, weights):
    """Construct an undirected igraph object from canonical graph arrays."""
    g = ig.Graph(n=int(n), edges=[(int(a), int(b)) for a, b in edges], directed=False)
    weight_arg = None
    if weights is not None and len(weights):
        # Unit-weight kNN graph: igraph/CPM's implicit default is exactly 1.
        if not np.all(weights == 1.0):
            # leidenalg accepts a weight iterable, this avoids edge-attribute
            # lookups and is the documented `weights` interface.
            weight_arg = weights.tolist()
    return g, weight_arg


def _seed_labels(seed, n: int):
    """Validate and normalize initial community assignments."""
    if seed is None:
        return None
    s = np.asarray(seed)
    if s.ndim != 1 or len(s) != n:
        raise ValueError("initial Leiden membership must be a 1D array of length n")
    if not np.issubdtype(s.dtype, np.integer):
        if not np.all(np.isfinite(s)) or not np.all(s == np.floor(s)):
            raise ValueError("initial Leiden membership must contain integer labels")
    _, inverse = np.unique(s.astype(np.int64, copy=False), return_inverse=True)
    return np.ascontiguousarray(inverse, dtype=np.int64)


def _partition_once(graph, initial_membership, weights, gamma, rng_seed, n_iterations):
    """Run one seeded Leiden optimization with the CPM objective (Traag et al., 2019)."""
    part = la.find_partition(
        graph,
        la.CPMVertexPartition,
        initial_membership=(None if initial_membership is None
                            else initial_membership.tolist()),
        weights=weights,
        n_iterations=int(n_iterations),
        seed=int(rng_seed),
        resolution_parameter=float(gamma),
    )
    membership = np.asarray(part.membership, dtype=np.int64)
    _, membership = np.unique(membership, return_inverse=True)
    quality = float(part.quality())
    if membership.ndim != 1 or len(membership) != graph.vcount():
        raise RuntimeError("leidenalg returned a membership vector with invalid shape")
    if not np.isfinite(quality):
        raise RuntimeError("leidenalg returned a non-finite CPM quality")
    return np.ascontiguousarray(membership), quality


def _seed_internal_density(n, edges, weights, seed):
    """Estimate the internal edge density of an initial community partition."""
    counts = np.bincount(seed, minlength=int(seed.max()) + 1)
    possible_pairs = float(np.sum(counts.astype(np.float64) * (counts - 1.0) / 2.0))
    if possible_pairs > 0 and len(edges):
        same = seed[edges[:, 0]] == seed[edges[:, 1]]
        if weights is None:
            internal_weight = float(np.count_nonzero(same))
        else:
            internal_weight = float(np.sum(weights[same]))
        if internal_weight > 0.0:
            return internal_weight / possible_pairs
    # If the seed contains no internal edges, use the overall graph density as
    # a neutral scale reference. This is still in the CPM edge-weight units.
    denom = float(n * (n - 1) / 2.0)
    if denom <= 0.0 or len(edges) == 0:
        return 0.0
    total_weight = float(len(edges)) if weights is None else float(np.sum(weights))
    return total_weight / denom


def _gamma_candidates(gamma_range, steps, gamma_reference):
    """Generate deterministic CPM resolution candidates around the seed density."""
    lo, hi = map(float, gamma_range)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 <= lo < hi):
        raise ValueError("gamma_range must be finite and satisfy 0 <= low < high")
    steps = int(steps)
    if steps < 5:
        raise ValueError("leiden_gamma_steps must be >= 5")

    if lo == 0.0:
        positive_floor = max(hi * 1e-5, np.finfo(np.float64).tiny)
        base = np.geomspace(positive_floor, hi, steps)
        values = [0.0, *base.tolist()]
    else:
        values = np.geomspace(lo, hi, steps).tolist()

    ref = float(np.clip(gamma_reference, lo, hi))
    if ref > 0.0:
        # CPM gamma has a direct density interpretation. Probe around the
        # density of the topological seed, not only an arbitrary [0, 1] grid.
        for factor in (0.5, 0.75, 1.0, 1.5, 2.0):
            values.append(float(np.clip(ref * factor, lo, hi)))
    return np.unique(np.asarray(values, dtype=np.float64))


def _run(n, edges, weights, gamma, seed, rng):
    """Run one compatibility Leiden optimization at a fixed resolution."""
    require_leiden()
    e, w = _canonical_graph_arrays(n, edges, weights)
    g, weight_arg = _make_igraph(n, e, w)
    s = _seed_labels(seed, n)
    run_seed = int(rng.integers(0, 2**31 - 1))
    membership, _ = _partition_once(
        g, s, weight_arg, gamma, run_seed, n_iterations=2
    )
    return membership


def leiden_partition(n, edges, weights, target, seed, gamma_range, cfg, rng,
                     return_info: bool = False):
    """Optimize a topological seed with Leiden and CPM (Traag et al., 2019)."""
    log = get_logger(getattr(cfg, "verbose", False))
    seed_arr = _seed_labels(seed, int(n))
    if seed_arr is None:
        raise ValueError("ENTRAP Leiden requires an initial topological membership")
    target = int(target)
    if target < 1 or target > int(n):
        raise ValueError("target community count must be in [1, n]")

    if not cfg.use_leiden:
        info = {
            "backend": "disabled", "partition_type": "none", "attempted_leiden": False,
            "used_leiden": False, "status": "disabled_by_config", "target_communities": target,
            "communities": int(np.unique(seed_arr).size), "gamma": None,
            "gamma_reference": None, "quality": None, "candidate_count": 0,
            "n_edges": int(len(edges)),
        }
        return (seed_arr, info) if return_info else seed_arr

    if not _HAS:
        if not cfg.allow_fallback:
            require_leiden()
        log.warning(
            "Leiden backend unavailable, explicit allow_fallback=True "
            "-> using merge-tree seed"
        )
        info = {
            "backend": "unavailable", "partition_type": "CPMVertexPartition",
            "attempted_leiden": False, "used_leiden": False,
            "status": "explicit_seed_fallback_backend_unavailable",
            "target_communities": target, "communities": int(np.unique(seed_arr).size),
            "gamma": None, "gamma_reference": None, "quality": None,
            "candidate_count": 0, "n_edges": int(len(edges)),
        }
        return (seed_arr, info) if return_info else seed_arr

    e, w = _canonical_graph_arrays(n, edges, weights)
    if len(e) == 0:
        info = {
            "backend": "igraph+leidenalg", "partition_type": "CPMVertexPartition",
            "attempted_leiden": False, "used_leiden": False,
            "status": "seed_fallback_empty_graph",
            "target_communities": target, "communities": int(np.unique(seed_arr).size),
            "gamma": None, "gamma_reference": None, "quality": None,
            "candidate_count": 0, "n_edges": 0,
        }
        return (seed_arr, info) if return_info else seed_arr

    graph, weight_arg = _make_igraph(n, e, w)
    gamma_reference = _seed_internal_density(n, e, w, seed_arr)
    gamma_values = _gamma_candidates(
        gamma_range,
        getattr(cfg, "leiden_gamma_steps", 17),
        gamma_reference,
    )
    restarts = max(1, int(getattr(cfg, "leiden_restarts", 1)))
    iterations = int(getattr(cfg, "leiden_iterations", -1))
    if iterations == 0:
        raise ValueError("leiden_iterations must be non-zero (negative means until stable)")

    exact = []
    best_near = None
    seed_count = int(np.unique(seed_arr).size)
    run_counter = 0
    for gamma in gamma_values:
        best_exact_at_gamma = None
        for _ in range(restarts):
            run_seed = int(rng.integers(0, 2**31 - 1))
            membership, quality = _partition_once(
                graph, seed_arr, weight_arg, float(gamma), run_seed, iterations
            )
            count = int(np.unique(membership).size)
            item = {
                "membership": membership,
                "gamma": float(gamma),
                "quality": float(quality),
                "communities": count,
            }
            run_counter += 1
            if count == target and (
                best_exact_at_gamma is None or quality > best_exact_at_gamma["quality"]
            ):
                best_exact_at_gamma = item
            if (best_near is None
                    or abs(count - target) < abs(best_near["communities"] - target)
                    or (abs(count - target) == abs(best_near["communities"] - target)
                        and quality > best_near["quality"])):
                best_near = item
        if best_exact_at_gamma is not None:
            exact.append(best_exact_at_gamma)

    if exact:
        # Select the CPM resolution nearest the seed partition's internal
        # density, within that resolution, keep the highest-quality restart.
        # This avoids comparing raw CPM quality values across different gamma.
        def exact_key(item):
            """Return deterministic metrics for comparing candidate partitions."""
            g = max(item["gamma"], np.finfo(np.float64).tiny)
            r = max(gamma_reference, np.finfo(np.float64).tiny)
            return (abs(math.log(g / r)), -item["quality"], item["gamma"])
        chosen = min(exact, key=exact_key)
        output = chosen["membership"]
        status = "optimized_exact_target"
        used = True
        chosen_gamma = chosen["gamma"]
        chosen_quality = chosen["quality"]
        chosen_count = chosen["communities"]
    else:
        # The topological mode count is a contract for the hierarchy. Do not
        # let the graph optimiser silently merge/split that count when no CPM
        # candidate preserves it, retain the valid mode-basin partition.
        output = seed_arr.copy()
        status = "seed_fallback_no_exact_target_partition"
        used = False
        chosen_gamma = None
        chosen_quality = None
        chosen_count = seed_count
        log.warning(
            "Leiden/CPM could not retain target=%d communities over %d candidate runs, "
            "keeping the topological seed (closest trial had %d).",
            target, run_counter,
            None if best_near is None else best_near["communities"],
        )

    info = {
        "backend": "igraph+leidenalg", "partition_type": "CPMVertexPartition",
        "attempted_leiden": bool(run_counter > 0), "used_leiden": bool(used),
        "status": status,
        "target_communities": target, "communities": int(chosen_count),
        "gamma": chosen_gamma, "gamma_reference": float(gamma_reference),
        "quality": chosen_quality, "candidate_count": int(len(gamma_values)),
        "optimisation_runs": int(run_counter), "n_edges": int(len(e)),
        "seed_communities": seed_count,
    }
    if return_info:
        return np.ascontiguousarray(output, dtype=np.int64), info
    return np.ascontiguousarray(output, dtype=np.int64)
