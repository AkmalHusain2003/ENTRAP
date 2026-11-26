"""Collect and plot hierarchical persistence diagnostics."""

from dataclasses import dataclass, field


@dataclass
class NodeRecord:
    """Store compact diagnostics for one hierarchical node."""
    depth: int
    size: int
    m_star: int
    tau: float
    entropy: float
    is_leaf: bool


@dataclass
class Tracker:
    """Collect diagnostic records during recursive clustering."""
    records: list = field(default_factory=list)

    def record(self, nd):
        """Append one hierarchical node to the diagnostic tracker."""
        self.records.append(
            NodeRecord(
                nd.depth,
                len(nd.indices),
                nd.m_star,
                nd.tau,
                nd.entropy,
                nd.is_leaf,
            )
        )

    def entropy_by_depth(self):
        """Group node persistence entropies by hierarchy depth."""
        ds = sorted({r.depth for r in self.records})
        return {d: [r.entropy for r in self.records if r.depth == d] for d in ds}

    def plot_entropy(self, save_path=None):
        """Plot persistence entropy against hierarchical depth."""
        import matplotlib.pyplot as plt
        data = self.entropy_by_depth()
        fig, ax = plt.subplots(figsize=(7, 4))
        for d, vals in data.items():
            ax.scatter([d]*len(vals), vals, s=12, alpha=0.6)
        ax.set_xlabel("tree depth")
        ax.set_ylabel("persistence entropy")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
