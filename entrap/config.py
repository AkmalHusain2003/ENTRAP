"""Define validated estimator parameters and data-size-dependent defaults."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import math


@dataclass
class EntrapConfig:
    """Store and validate ENTRAP algorithm and execution parameters."""
    min_cluster_size: int = 10
    significance: float = 0.05
    # Legacy explicit K. In V7 it caps the graph-scale family when supplied.
    k: Optional[int] = None
    B: int = 50
    P: int = 199
    f: float = 0.8
    metric: str = "euclidean"
    seed: int = 0
    n_jobs: int = -1
    parallel_max_workers: int = 8
    parallel_memory_mb: int = 1024
    overlap_threshold: float = 0.5
    block_scale: Optional[int] = None
    block_ensemble: bool = True
    gamma_range: Tuple[float, float] = (0.0, 1.0)
    leiden_gamma_steps: int = 17
    leiden_restarts: int = 1
    leiden_iterations: int = -1  # negative: continue until no quality improvement
    ridge: float = 1e-6
    pi_threshold: Optional[float] = None
    alpha_gate: float = 0.99
    stability_B: Optional[int] = None
    pi_runs: Optional[int] = None
    pi_full_stability: bool = True
    normalized_density: bool = False
    halo_noise: bool = True
    tau_null: str = "poisson"
    tau_null_dim: int | None = None
    halo_eta: float = 0.1
    halo_mode: str = "poisson"
    halo_alpha: float = 0.001
    halo_kappa: float = 2.5
    halo_contrast: float = 2.0
    halo_dim: int | None = None
    mutual_reachability: bool = False
    use_leiden: bool = True
    allow_fallback: bool = False
    compute_pi: bool = True
    verbose: bool = False

    # V7: graph-connectivity scale and density-estimation scale are distinct.
    adaptive_k: bool = True
    adaptive_k_min: int = 4
    adaptive_k_max: Optional[int] = None
    adaptive_k_sensitivity: float = 2.5
    adaptive_k_window: int = 4
    adaptive_k_patience: int = 2
    multi_k: bool = True
    k_scales: Optional[Tuple[int, ...]] = None
    # Optional tail extension is intentionally opt-in: feature coordinates must
    # have a meaningful metric before lower-density points can be called tails.
    tail_membership: bool = False
    tail_feature_indices: Optional[Tuple[int, ...]] = None
    tail_k_neighbors: int = 5
    tail_max_hops: int = 3
    tail_radius_factor: float = 2.0
    tail_width_factor: float = 2.5
    tail_min_anisotropy: float = 1.5

    # Resolved fields
    k_resolved: int = field(default=15, init=False)  # compatibility alias: max graph K
    k_scales_resolved: Tuple[int, ...] = field(default=(), init=False)
    adaptive_k_max_resolved: int = field(default=15, init=False)
    adaptive_k_min_resolved: int = field(default=4, init=False)
    block_scale_resolved: int = field(default=2, init=False)
    pi_threshold_resolved: float = field(default=0.95, init=False)

    @classmethod
    def from_params(cls, min_cluster_size, significance, advanced=None):
        """Build and validate configuration from estimator parameters and advanced options."""
        adv = dict(advanced or {})
        known = {f for f in cls.__dataclass_fields__ if not f.endswith("_resolved")}
        unknown = set(adv) - known
        if unknown:
            raise ValueError(f"Unknown advanced options: {sorted(unknown)}")
        if "gamma_range" in adv and adv["gamma_range"] is not None:
            adv["gamma_range"] = tuple(float(v) for v in adv["gamma_range"])
        if "k_scales" in adv and adv["k_scales"] is not None:
            adv["k_scales"] = tuple(int(v) for v in adv["k_scales"])
        if "tail_feature_indices" in adv and adv["tail_feature_indices"] is not None:
            adv["tail_feature_indices"] = tuple(int(v) for v in adv["tail_feature_indices"])
        cfg = cls(min_cluster_size=int(min_cluster_size), significance=float(significance), **adv)
        cfg.validate()
        return cfg

    def validate(self):
        """Validate configuration values before fitting."""
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size >= 2")
        if not (0 < self.significance < 1):
            raise ValueError("significance in (0,1)")
        if not (0 < self.f < 1):
            raise ValueError("f in (0,1)")
        if self.B < 1 or self.P < 1:
            raise ValueError("B,P >= 1")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.parallel_max_workers < 1:
            raise ValueError("parallel_max_workers >= 1")
        if self.parallel_memory_mb < 1:
            raise ValueError("parallel_memory_mb >= 1")
        if len(self.gamma_range) != 2:
            raise ValueError("gamma_range must contain (low, high)")
        if not (
            math.isfinite(float(self.gamma_range[0]))
            and math.isfinite(float(self.gamma_range[1]))
            and 0.0 <= float(self.gamma_range[0]) < float(self.gamma_range[1])
        ):
            raise ValueError("gamma_range must satisfy finite 0 <= low < high")
        if self.leiden_gamma_steps < 5:
            raise ValueError("leiden_gamma_steps >= 5")
        if self.leiden_restarts < 1:
            raise ValueError("leiden_restarts >= 1")
        if self.leiden_iterations == 0:
            raise ValueError("leiden_iterations cannot be zero")
        if not (0.0 <= self.halo_eta < 1.0):
            raise ValueError("halo_eta in [0,1)")
        if self.tau_null not in ("poisson", "permutation"):
            raise ValueError("tau_null in {poisson, permutation}")
        if self.tau_null_dim is not None and int(self.tau_null_dim) < 1:
            raise ValueError("tau_null_dim >= 1")
        if self.halo_mode not in ("poisson", "fixed"):
            raise ValueError("halo_mode in {poisson, fixed}")
        if not (1e-4 <= self.halo_alpha < 0.5):
            raise ValueError("halo_alpha in [1e-4, 0.5)")
        if self.adaptive_k_min < 2:
            raise ValueError("adaptive_k_min >= 2")
        if self.adaptive_k_max is not None and self.adaptive_k_max < self.adaptive_k_min:
            raise ValueError("adaptive_k_max must be >= adaptive_k_min")
        if self.adaptive_k_sensitivity <= 1.0:
            raise ValueError("adaptive_k_sensitivity must be > 1")
        if self.adaptive_k_window < 2:
            raise ValueError("adaptive_k_window >= 2")
        if self.adaptive_k_patience < 1:
            raise ValueError("adaptive_k_patience >= 1")
        if self.k is not None and int(self.k) < 2:
            raise ValueError("k must be >= 2")
        if self.k_scales is not None:
            if len(self.k_scales) == 0 or any(int(v) < 2 for v in self.k_scales):
                raise ValueError("k_scales must contain integer values >= 2")
        if self.tail_membership and not self.tail_feature_indices:
            raise ValueError("tail_feature_indices must be specified when tail_membership=True")
        if self.tail_k_neighbors < 1:
            raise ValueError("tail_k_neighbors >= 1")
        if self.tail_max_hops < 1:
            raise ValueError("tail_max_hops >= 1")
        if self.tail_radius_factor <= 0 or self.tail_width_factor <= 0:
            raise ValueError("tail radius/width factors must be > 0")
        if self.tail_min_anisotropy <= 1:
            raise ValueError("tail_min_anisotropy must be > 1")

    def resolve(self, N):
        """Resolve data-size-dependent neighbourhood and execution parameters."""
        N = int(N)
        if N < 2:
            raise ValueError("At least two samples are required")
        legacy = (
            int(self.k)
            if self.k is not None
            else min(50, max(15, math.ceil(2 * math.log(max(N, 2)))))
        )
        legacy_resolved = min(legacy, max(2, N - 1))
        self.k_resolved = legacy_resolved

        if self.k_scales is not None:
            # An explicitly supplied scale family may exceed the automatic
            # heuristic. Only an explicitly supplied legacy ``k`` acts as a cap.
            scale_cap = legacy_resolved if self.k is not None else max(2, N - 1)
            scales = sorted({min(int(v), scale_cap) for v in self.k_scales if int(v) >= 2})
        else:
            # A compact logarithmic-like scale family. The family is deliberately
            # bounded, family-wise calibration below corrects searching across it.
            fractions = (0.28, 0.42, 0.60, 0.80, 1.00)
            scales = sorted({max(2, min(legacy_resolved, int(round(legacy_resolved * f))))
                             for f in fractions})
        if not scales:
            scales = [min(2, N - 1)]
        self.k_scales_resolved = tuple(scales if self.multi_k else [scales[-1]])

        amax = self.adaptive_k_max
        if amax is None:
            amax = max(self.k_scales_resolved)
        self.adaptive_k_max_resolved = min(max(2, int(amax)), N - 1)
        self.adaptive_k_min_resolved = min(
            max(2, int(self.adaptive_k_min)),
            self.adaptive_k_max_resolved,
        )

        # Ensure graph cache contains the largest graph scale. Density estimation
        # can use a larger adaptive K if explicitly requested.
        self.k_resolved = max(self.k_resolved, max(self.k_scales_resolved),
                              self.adaptive_k_max_resolved)
        self.pi_threshold_resolved = (
            self.pi_threshold
            if self.pi_threshold is not None
            else 1.0 - self.significance
        )
        if self.stability_B is None:
            self.stability_B = self.B
        if self.pi_runs is None:
            self.pi_runs = self.B
        return self

    def to_dict(self):
        """Return the resolved configuration as a dictionary."""
        return asdict(self)
