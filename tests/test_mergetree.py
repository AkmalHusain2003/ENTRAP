"""Merge-tree correctness on hand-checkable graphs + independent oracle."""
import numpy as np
import pytest
from entrap import topology
from entrap.graph import adj_to_csr
from entrap.mergetree import density_merge_tree
from helpers import random_graph, path_graph, oracle_persistence, pairs_multiset, partition_equal


def _mt(dens, adj):
    """Construct the merge tree used by the test."""
    return density_merge_tree(dens, adj)


def test_single_node():
    """Verify single node."""
    dens, adj = path_graph([5.0], [])
    mt = _mt(dens, adj)
    assert len(mt.mode_prom) == 1
    assert np.isinf(mt.mode_prom[0])
    assert mt.pairs.size == 0
    assert mt.count_significant(1e9) == 1


def test_two_nodes_single_mode():
    # The lower vertex is not a local maximum: a single essential H0 class,
    # no finite pair (agrees with the independent oracle).
    """Verify two nodes single mode."""
    dens, adj = path_graph([2.0, 1.0], [(0, 1)])
    mt = _mt(dens, adj)
    assert len(mt.mode_prom) == 1
    assert np.isinf(mt.mode_prom).sum() == 1
    assert mt.pairs.size == 0
    o_pairs, o_ess = oracle_persistence(dens, adj)
    assert o_pairs == [] and o_ess == 1


def test_two_nodes_disconnected_two_modes():
    """Verify two nodes disconnected two modes."""
    dens, adj = path_graph([2.0, 1.0], [])
    mt = _mt(dens, adj)
    assert len(mt.mode_prom) == 2
    assert np.isinf(mt.mode_prom).sum() == 2
    assert mt.pairs.size == 0


def test_three_node_path_two_modes():
    # densities 3,1,2 on path 0-1-2: modes at 0 and 2 merge at vertex 1 (level 1)
    """Verify three node path two modes."""
    dens, adj = path_graph([3.0, 1.0, 2.0], [(0, 1), (1, 2)])
    mt = _mt(dens, adj)
    assert pairs_multiset(mt.pairs) == [(2.0, 1.0)]
    assert mt.count_significant(0.5) == 2
    assert mt.count_significant(1.5) == 1
    lab_lo = mt.assign(0.5)
    lab_hi = mt.assign(1.5)
    assert len(np.unique(lab_lo)) == 2
    assert len(np.unique(lab_hi)) == 1
    assert lab_lo[0] == lab_lo[1]      # vertex 1 joins the elder basin
    assert lab_lo[2] != lab_lo[0]


def test_disconnected_components_all_essential():
    """Verify disconnected components all essential."""
    dens, adj = path_graph([4.0, 3.0, 2.0, 1.0], [(0, 1), (2, 3)])
    mt = _mt(dens, adj)
    assert np.isinf(mt.mode_prom).sum() == 2
    assert mt.count_significant(1e6) == 2


def test_star_center_saddle():
    # leaves 5,4,3 attach through center 1 -> two deaths at level 1
    """Verify star center saddle."""
    dens, adj = path_graph([1.0, 5.0, 4.0, 3.0], [(0, 1), (0, 2), (0, 3)])
    mt = _mt(dens, adj)
    assert pairs_multiset(mt.pairs) == [(3.0, 1.0), (4.0, 1.0)]
    assert np.isinf(mt.mode_prom).sum() == 1


@pytest.mark.parametrize("seed", range(12))
def test_matches_independent_oracle(seed):
    """Verify matches independent oracle."""
    rng = np.random.default_rng(seed)
    dens, adj = random_graph(rng, n=14, p=0.3)
    mt = _mt(dens, adj)
    o_pairs, o_ess = oracle_persistence(dens, adj)
    assert pairs_multiset(mt.pairs) == pairs_multiset(o_pairs)
    assert int(np.isinf(mt.mode_prom).sum()) == o_ess


@pytest.mark.parametrize("seed", range(8))
def test_assign_count_invariant_and_monotone(seed):
    """Verify assign count invariant and monotone."""
    rng = np.random.default_rng(100 + seed)
    dens, adj = random_graph(rng, n=16, p=0.3)
    mt = _mt(dens, adj)
    prev = None
    for tau in [0.0, 0.5, 1.0, 2.0, 5.0, 1e9]:
        lab = mt.assign(tau)
        assert len(np.unique(lab)) == mt.count_significant(tau)
        if prev is not None:
            assert len(np.unique(lab)) <= prev
        prev = len(np.unique(lab))


def test_prominence_equals_birth_minus_death():
    """Verify prominence equals birth minus death."""
    rng = np.random.default_rng(7)
    dens, adj = random_graph(rng, n=18, p=0.3)
    mt = _mt(dens, adj)
    finite = np.isfinite(mt.mode_prom)
    births = mt.mode_birth[finite]
    deaths = np.array([mt.pairs[i, 1] for i in range(len(mt.pairs))])
    assert np.all(mt.pairs[:, 0] > mt.pairs[:, 1])
    assert np.allclose(np.sort(mt.mode_prom[finite]), np.sort(births - deaths))


def test_empty_graph():
    """Verify empty graph."""
    ai, ao = adj_to_csr([])
    vm, mp, mb, mpr, pairs = topology.merge_tree_arrays(
        np.ascontiguousarray(np.empty(0)), ai, ao)
    assert len(vm) == 0 and len(mpr) == 0 and pairs.shape == (0, 2)
