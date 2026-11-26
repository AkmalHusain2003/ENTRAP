"""Filtration on the kNN graph: vertex log-density, optionally with mutual-reachability edge levels.

Vertex level  l_v   = log rho_v                  (rho = 1/mean kNN distance, or normalized)
Edge level    l_uv = min(l_u, l_v, -log d_uv), using mean kNN radius as the core-distance scale.

Persistence, the block-permutation null (tau) and the subsample stability counts must all be run on
the SAME filtration, every consumer takes a Filtration so that this holds by construction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import topology
from .graph import knn_density, log_density, adj_to_csr, edges_to_csr
from . import topology
from .mergetree import MergeTree, density_merge_tree


@dataclass
class Filtration:
    """Represent a vertex or mutual-reachability graph filtration."""
    ell: np.ndarray                  # vertex log density
    edges: np.ndarray                # (E, 2) local undirected edges (a < b)
    resid: np.ndarray | None = None  # edge-length residual (None -> plain vertex filtration)
    elev: np.ndarray | None = None   # edge levels  (None -> min of endpoints)

    @property
    def mreach(self):
        """Indicate whether the filtration includes mutual-reachability edge levels."""
        return self.resid is not None

    def edge_arrays(self):
        """Return the local undirected edge endpoints as contiguous arrays."""
        return (np.ascontiguousarray(self.edges[:, 0], dtype=np.int64),
                np.ascontiguousarray(self.edges[:, 1], dtype=np.int64))


def build_filtration(X, edges, k, d_hat=None, normalized=False, mreach=True, workers=-1):
    """Build vertex-density and optional mutual-reachability filtration levels."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    base = knn_density(X, k, d_hat, normalized, workers=workers)
    ell = log_density(base)
    if not mreach:
        return Filtration(ell, edges), base
    resid, elev = topology.edge_fields(X, edges, ell)
    return Filtration(ell, edges, resid, elev), base


def filtration_merge_tree(fil, adj):
    """Construct the H0 merge tree from the configured filtration."""
    if not fil.mreach:
        return density_merge_tree(fil.ell, adj)
    ea, eb = fil.edge_arrays()
    vm, mp, mb, mpr, pairs = topology.merge_tree_edges(np.ascontiguousarray(fil.ell), ea, eb,
                                                       np.ascontiguousarray(fil.elev))
    return MergeTree(vm, mp, mb, mpr, pairs)


def filtration_count(fil, tau):
    """Count significant H0 modes at the requested threshold."""
    if not fil.mreach:
        ai, ao = edges_to_csr(len(fil.ell), fil.edges)
        return topology.count_significant(np.ascontiguousarray(fil.ell), ai, ao, tau)
    ea, eb = fil.edge_arrays()
    return topology.count_significant_edges(np.ascontiguousarray(fil.ell), ea, eb,
                                            np.ascontiguousarray(fil.elev), tau)


def build_filtration_from_density(X, edges, base_density, mreach=True):
    """Build a graph filtration from precomputed vertex-density values."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    edges = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    base = np.ascontiguousarray(base_density, dtype=np.float64)
    if len(base) != len(X):
        raise ValueError("base_density must have one value per sample")
    ell = log_density(base)
    if not mreach:
        return Filtration(ell, edges), base
    resid, elev = topology.edge_fields(X, edges, ell)
    return Filtration(ell, edges, resid, elev), base
