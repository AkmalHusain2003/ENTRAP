"""Estimate membership stability through repeated subsampling."""

from __future__ import annotations
import copy
import math
import numpy as np
from .tree import build_tree, extract_clusters
from .parallel import resolve_workers, ordered_thread_map
from .leiden import has_leiden


def compute_pi(graph, X, ref_labels, ref_basins, cfg, tracker, rng):
    """Estimate per-point and per-cluster membership stability across subsamples."""
    N = graph.N
    C = len(ref_basins)
    if C == 0 or not cfg.compute_pi:
        return (ref_labels >= 0).astype(float), np.zeros((N, max(C, 1))), ref_labels

    pi_counts = np.zeros((N, C), dtype=np.float64)
    present = np.zeros(N, dtype=np.float64)
    m = max(2, int(math.floor(cfg.f * N)))
    ref_sets = {c: set(map(int, b)) for c, b in ref_basins.items()}
    runs = int(cfg.pi_runs)
    bytes_per_worker = max(1, 32 * m * X.shape[1] + 48 * m * cfg.k_resolved
                            + 24 * len(graph.edges))
    workers = resolve_workers(cfg.n_jobs, runs, cfg.parallel_max_workers,
                              cfg.parallel_memory_mb, bytes_per_worker)
    # The external Leiden optimiser has process-global RNG internals on some
    # releases. Keep full tree runs serial when that backend is active, the
    # independent native/numeric Monte Carlo loops inside each run still use
    # the configured worker pool.
    if cfg.use_leiden and has_leiden():
        workers = 1

    def tasks():
        """Yield independent membership-stability tasks."""
        for _ in range(runs):
            idx = np.sort(rng.choice(N, size=m, replace=False))
            seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            yield idx, seed

    def one(task):
        """Evaluate reference-cluster overlap for one subsample."""
        idx, seed = task
        local_cfg = copy.copy(cfg)
        # Avoid nested worker pools: outer pi runs own the parallel capacity.
        local_cfg.n_jobs = 1 if workers > 1 else cfg.n_jobs
        local_cfg.parallel_max_workers = 1 if workers > 1 else cfg.parallel_max_workers
        local_rng = np.random.default_rng(seed)
        st = build_tree(graph, X, idx, local_cfg, None, local_rng,
                        internal_stability=cfg.pi_full_stability)
        _, sub_basins, _ = extract_clusters(st, N, halo_noise=cfg.halo_noise)
        return idx, sub_basins

    for idx, sub_basins in ordered_thread_map(one, tasks(), workers):
        idx_set = set(map(int, idx))
        present[idx] += 1.0
        for sb in sub_basins.values():
            sb_set = set(map(int, sb))
            bc, bo = -1, 0.0
            for rc, rset in ref_sets.items():
                rs = rset & idx_set
                if not rs:
                    continue
                ov = len(sb_set & rs) / min(len(sb_set), len(rs))
                if ov > bo:
                    bo, bc = ov, rc
            if bc >= 0 and bo >= cfg.overlap_threshold:
                pi_counts[sb, bc] += 1.0

    den = np.maximum(present, 1.0)
    return pi_counts.sum(1) / den, pi_counts / den[:, None], ref_labels
