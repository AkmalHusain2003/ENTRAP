"""Define the result container returned by the ENTRAP estimator."""

from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np


@dataclass
class EntrapResults:
    """Store clustering labels, probabilities, summary statistics, and diagnostics."""
    labels: np.ndarray
    probabilities: np.ndarray
    pi_per_cluster: np.ndarray
    n_clusters: int
    noise_rescued: int
    execution_time: float
    cluster_stats: Dict[int, Any] = field(default_factory=dict)
    tree: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    tracker: Any = None
    core_labels: Any = None
    tail_labels: Any = None

    def summary(self):
        """Return concise cluster and noise statistics."""
        return {"n_clusters": self.n_clusters, "n_noise": int(np.sum(self.labels < 0)),
                "noise_rescued": self.noise_rescued, "time_seconds": round(self.execution_time, 3),
                "cluster_sizes": [int(np.sum(self.labels == c)) for c in range(self.n_clusters)]}
