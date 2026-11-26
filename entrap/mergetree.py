"""Represent persistence merge trees and their mode assignments."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import topology
from .graph import adj_to_csr


@dataclass
class MergeTree:
    """Store H0 persistence pairs and persistent-mode assignments."""
    vertex_mode: np.ndarray
    mode_parent: np.ndarray
    mode_birth: np.ndarray
    mode_prom: np.ndarray
    pairs: np.ndarray

    def significant_modes(self, tau):
        """Return mode indices whose persistence meets the requested threshold."""
        return np.where(self.mode_prom >= tau)[0]

    def count_significant(self, tau):
        """Count significant modes in a vertex filtration."""
        return int(np.sum(self.mode_prom >= tau))

    def assign(self, tau):
        """Assign vertices to persistent mode basins at a specified threshold."""
        raw = topology.assign_basins(self.vertex_mode, self.mode_parent, self.mode_prom, tau)
        uniq = np.unique(raw)
        remap = {int(m): i for i, m in enumerate(uniq)}
        return np.array([remap[int(m)] for m in raw], dtype=np.int64)


def density_merge_tree(density, adj):
    # ``adj`` can be a CSR pair (internal path) or the legacy list-of-lists API.
    """Construct an H0 merge tree using elder-rule persistence (Edelsbrunner et al., 2002)."""
    ai, ao = adj_to_csr(adj)
    vm, mp, mb, mpr, pairs = topology.merge_tree_arrays(np.ascontiguousarray(density), ai, ao)
    return MergeTree(vm, mp, mb, mpr, pairs)
