"""Edge (mutual-reachability) filtration: independent oracle, parity, reduction to the vertex sweep."""
import importlib
import sys
import numpy as np
import pytest
from entrap import topology
from entrap.graph import adj_to_csr
from helpers import random_graph, edges_from_adj, oracle_persistence_edges, pairs_multiset


def _nonzero(pairs, tol=1e-9):
    """Return a nonzero baseline value for the test graph."""
    a = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    return pairs_multiset(a[(a[:, 0] - a[:, 1]) > tol])


def _graph(seed, n=16, p=0.3, lower=True):
    """Construct the graph fixture used by the edge-filtration test."""
    rng = np.random.default_rng(seed)
    dens, adj = random_graph(rng, n=n, p=p)
    edges = edges_from_adj(adj)
    mn = np.minimum(dens[edges[:, 0]], dens[edges[:, 1]]) if len(edges) else np.empty(0)
    elev = mn - (rng.exponential(0.7, len(edges)) if lower else 0.0)
    return dens, edges, np.ascontiguousarray(elev)


@pytest.mark.parametrize("seed", range(10))
def test_edge_sweep_matches_independent_oracle(seed):
    """Verify edge sweep matches independent oracle."""
    dens, edges, elev = _graph(seed)
    _, _, _, mpr, pairs = topology.merge_tree_edges(dens, np.ascontiguousarray(edges[:, 0]),
                                                    np.ascontiguousarray(edges[:, 1]), elev)
    o_pairs, o_ess = oracle_persistence_edges(dens, edges, elev)
    assert _nonzero(pairs) == _nonzero(o_pairs)
    assert int(np.isinf(mpr).sum()) == o_ess


@pytest.mark.parametrize("seed", range(6))
def test_edge_sweep_reduces_to_vertex_sweep_when_edge_level_is_min(seed):
    """Verify edge sweep reduces to vertex sweep when edge level is min."""
    dens, edges, elev = _graph(seed, lower=False)
    ai, ao = adj_to_csr([[] for _ in range(len(dens))]) if len(edges) == 0 else adj_to_csr(
        [list(np.concatenate([edges[edges[:, 0] == i, 1], edges[edges[:, 1] == i, 0]])) for i in range(len(dens))])
    _, _, _, mp_v, pairs_v = topology.merge_tree_arrays(dens, ai, ao)
    _, _, _, mp_e, pairs_e = topology.merge_tree_edges(dens, np.ascontiguousarray(edges[:, 0]),
                                                       np.ascontiguousarray(edges[:, 1]), elev)
    fv = np.sort(mp_v[np.isfinite(mp_v)])
    fe = np.sort(mp_e[np.isfinite(mp_e)])
    fe = fe[fe > 1e-9]
    np.testing.assert_allclose(fv[fv > 1e-9], fe, atol=1e-12)
    assert int(np.isinf(mp_v).sum()) == int(np.isinf(mp_e).sum())


def test_lowering_edges_never_decreases_prominence():
    """Verify lowering edges never decreases prominence."""
    dens, edges, elev = _graph(3, lower=False)
    ea, eb = np.ascontiguousarray(edges[:, 0]), np.ascontiguousarray(edges[:, 1])
    base = topology.merge_tree_edges(dens, ea, eb, elev)[3]
    low = topology.merge_tree_edges(dens, ea, eb, np.ascontiguousarray(elev - 0.4))[3]
    assert low[np.isfinite(low)].sum() >= base[np.isfinite(base)].sum() - 1e-12


def test_every_vertex_is_a_mode_and_basins_follow_parents():
    """Verify every vertex is a mode and basins follow parents."""
    dens, edges, elev = _graph(2, n=20, p=0.25)
    ea, eb = np.ascontiguousarray(edges[:, 0]), np.ascontiguousarray(edges[:, 1])
    vm, mpar, mb, mpr, pairs = topology.merge_tree_edges(dens, ea, eb, elev)
    assert len(mpr) == len(dens) and sorted(vm.tolist()) == list(range(len(dens)))
    big = topology.assign_basins(vm, mpar, mpr, float(np.sort(mpr[np.isfinite(mpr)])[-1] + 1.0))
    assert len(np.unique(big)) == int(np.isinf(mpr).sum())     # tau above all finite bars: roots only


def test_count_significant_edges_counts_infinite_and_large():
    """Verify count significant edges counts infinite and large."""
    dens = np.array([3.0, 1.0, 2.0])
    ea = np.array([0, 1], np.int64)
    eb = np.array([1, 2], np.int64)
    elev = np.array([1.0, 1.0])                                  # path 3-1-2, merge at level 1
    n_hi = topology.count_significant_edges(dens, ea, eb, elev, 1.5)
    n_lo = topology.count_significant_edges(dens, ea, eb, elev, 0.5)
    assert n_hi == 1 and n_lo == 2        # {mode 0: inf} , {+ mode 2: prom 1.0}


@pytest.fixture
def parity_pair_edges():
    """Construct equivalent edge arrays for the parity comparison."""
    if not topology.USING_CYTHON:
        pytest.skip("Cython core not built")
    cy = (topology.merge_tree_edges, topology.count_significant_edges, topology.prominence_null_edges,
          topology.distance_blocks)
    saved = sys.modules.get("entrap._topology")
    sys.modules["entrap._topology"] = None
    try:
        importlib.reload(topology)
        fb = (topology.merge_tree_edges, topology.count_significant_edges, topology.prominence_null_edges,
              topology.distance_blocks)
        yield cy, fb
    finally:
        if saved is not None:
            sys.modules["entrap._topology"] = saved
        else:
            sys.modules.pop("entrap._topology", None)
        importlib.reload(topology)


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(6))
def test_edge_sweep_parity(parity_pair_edges, seed):
    """Verify edge sweep parity."""
    cy, fb = parity_pair_edges
    dens, edges, elev = _graph(seed)
    ea, eb = np.ascontiguousarray(edges[:, 0]), np.ascontiguousarray(edges[:, 1])
    for x, y in zip(cy[0](dens, ea, eb, elev), fb[0](dens, ea, eb, elev)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)
    for tau in (0.0, 0.4, 1.5):
        assert cy[1](dens, ea, eb, elev, tau) == fb[1](dens, ea, eb, elev, tau)


@pytest.mark.parity
def test_edge_null_parity(parity_pair_edges):
    """Verify edge null parity."""
    cy, fb = parity_pair_edges
    dens, edges, _ = _graph(5, n=24, p=0.25)
    ea, eb = np.ascontiguousarray(edges[:, 0]), np.ascontiguousarray(edges[:, 1])
    resid = np.random.default_rng(1).normal(-0.1, 0.3, len(ea))
    ai, ao = adj_to_csr([list(np.concatenate([edges[edges[:, 0] == i, 1], edges[edges[:, 1] == i, 0]])) for i in range(len(dens))])
    blocks = cy[3](ai, ao, 2)
    a = cy[2](dens, ea, eb, resid, blocks, 29, 0.1, np.random.default_rng(7))
    b = fb[2](dens, ea, eb, resid, blocks, 29, 0.1, np.random.default_rng(7))
    assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
