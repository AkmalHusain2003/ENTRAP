"""Implement the Entropy-based Topological Rescue of Ambiguous Points estimator."""

import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from .config import EntrapConfig
from .utils import validate_X, get_rng, get_logger
from .memory import MemoryManager
from .graph import build_knn_graph
from .tree import build_tree, extract_clusters
from .stability import compute_pi
from .rescue import rescue
from .tails import extend_tidal_tail_candidates
from .results import EntrapResults
from .tracker import Tracker
from .leiden import backend_info, require_leiden
from . import topology


def _collect_leiden_diagnostics(root):
    """Collect Leiden execution diagnostics from the recursive hierarchy."""
    calls = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.leiden_info is not None:
            calls.append(
                dict(
                    node.leiden_info,
                    depth=int(node.depth),
                    node_size=int(len(node.indices)),
                )
            )
        stack.extend(node.children)
    return {
        "calls": len(calls),
        "attempted": sum(bool(item.get("attempted_leiden")) for item in calls),
        "optimised": sum(bool(item.get("used_leiden")) for item in calls),
        "seed_fallbacks": sum(str(item.get("status", "")).startswith("seed_fallback")
                               or str(item.get("status", "")).startswith("explicit_seed_fallback")
                               for item in calls),
        "nodes": calls,
    }


class ENTRAP(BaseEstimator, ClusterMixin):
    """Perform entropy-based topological clustering with hierarchical graph partitioning."""

    def __init__(self, min_cluster_size=10, significance=0.05, advanced=None):
        """Initialize the memory manager and its temporary-storage settings."""
        self.min_cluster_size = min_cluster_size
        self.significance = significance
        self.advanced = advanced

    def fit(self, X, y=None):
        """Fit the ENTRAP hierarchy and calculate final membership labels."""
        t0 = time.time()
        X = validate_X(X)
        N = X.shape[0]
        cfg = (
            EntrapConfig.from_params(
                self.min_cluster_size, self.significance, self.advanced
            ).resolve(N)
        )
        # When Leiden is requested it must be available before expensive work
        # starts. A merge-tree seed fallback is allowed only by explicit opt-in.
        if cfg.use_leiden and not cfg.allow_fallback:
            require_leiden()
        self.leiden_backend_ = backend_info()
        log = get_logger(cfg.verbose)
        rng = get_rng(cfg.seed)
        mem = MemoryManager()
        tracker = Tracker()
        if cfg.tau_null == "poisson":
            m_scales = len(cfg.k_scales_resolved)
            p_floor = 2.0 * m_scales / (cfg.P + 1.0)
            if p_floor > cfg.significance + 1e-12:
                log.warning(
                    "Poisson Monte-Carlo resolution is too coarse for the requested "
                    "multiscale family: minimum attainable adjusted p=%.4g > alpha=%.4g "
                    "(P=%d, K_scales=%d). Increase P or reduce the pre-declared K family.",
                    p_floor, cfg.significance, cfg.P, m_scales,
                )
        log.info(
            "Cython core: %s | normalized_density=%s mutual_reachability=%s "
            "halo_noise=%s halo_mode=%s block_ensemble=%s",
            topology.USING_CYTHON,
            cfg.normalized_density,
            cfg.mutual_reachability,
            cfg.halo_noise,
            cfg.halo_mode,
            cfg.block_ensemble,
        )
        graph = build_knn_graph(
            X,
            cfg.k_resolved,
            cfg.metric,
            n_jobs=cfg.n_jobs,
            store_knn=True,
        )
        root = build_tree(
            graph,
            X,
            np.arange(N),
            cfg,
            tracker,
            rng,
            internal_stability=True,
        )
        self.leiden_diagnostics_ = _collect_leiden_diagnostics(root)
        ref_labels, ref_basins, proms = extract_clusters(root, N, halo_noise=cfg.halo_noise)
        # The root kNN cache is no longer needed. Pi runs rebuild exact local
        # neighbourhoods on their own subsamples, free N*K arrays before them.
        graph.knn_idx = None
        graph.knn_dist = None
        pi, pi_pc, _ = compute_pi(graph, X, ref_labels, ref_basins, cfg, tracker, rng)
        labels = ref_labels.copy()
        labels[(labels >= 0) & (pi < cfg.pi_threshold_resolved)] = -1
        core_labels = labels.copy()
        # Rescue is calibrated by per-cluster subsample probabilities. If the
        # user disabled compute_pi, pi_pc carries no rescue evidence, do not
        # silently re-admit noise using an uncalibrated zero-probability matrix.
        if cfg.compute_pi:
            labels, rescued = rescue(X, labels, pi_pc, ref_basins, proms, cfg)
        else:
            rescued = 0
        tail_mask = np.zeros(N, dtype=bool)
        if cfg.tail_membership:
            # Only post-stability core members may seed geometric tail growth,
            # reference members rejected by pi filtering are not silently used
            # as hidden anchors for additional candidates.
            tail_basins = {
                int(cid): np.asarray(idx, dtype=np.int64)[
                    core_labels[np.asarray(idx, dtype=np.int64)] == int(cid)
                ]
                for cid, idx in ref_basins.items()
            }
            tail_basins = {cid: idx for cid, idx in tail_basins.items() if len(idx)}
            labels, tail_mask = extend_tidal_tail_candidates(
                X, labels, tail_basins, cfg.tail_feature_indices,
                k_neighbors=cfg.tail_k_neighbors,
                max_hops=cfg.tail_max_hops,
                radius_factor=cfg.tail_radius_factor,
                width_factor=cfg.tail_width_factor,
                min_anisotropy=cfg.tail_min_anisotropy,
            )
        nc = len(ref_basins)
        stats = {c: {"size": int(np.sum(labels == c)), "core": int(len(ref_basins[c])),
                     "prominence": proms.get(c, 0.0)} for c in range(nc)}
        self.labels_ = labels
        self.core_labels_ = core_labels
        self.tail_mask_ = tail_mask
        self.tail_labels_ = np.where(tail_mask, labels, -1)
        self.result_ = EntrapResults(labels, pi, pi_pc, nc, rescued, time.time()-t0,
                                     stats, root, cfg.to_dict(), tracker,
                                     core_labels=core_labels, tail_labels=self.tail_labels_)
        mem.collect()
        return self

    def fit_predict(self, X, y=None):
        """Fit ENTRAP and return the resulting labels."""
        return self.fit(X, y).labels_

    def get_summary(self):
        """Return summary statistics from the fitted estimator."""
        if not hasattr(self, "result_"):
            raise ValueError("call fit() first")
        return self.result_.summary()
