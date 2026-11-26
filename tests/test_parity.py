"""Cython vs pure-Python backend equivalence."""
import importlib
import sys
import numpy as np
import pytest
from entrap import topology
from entrap.graph import adj_to_csr
from helpers import random_graph

FN = ["merge_tree_arrays", "distance_blocks", "count_significant", "assign_basins"]


@pytest.fixture
def parity_pair():
    """Create matching inputs for the Python and native implementations."""
    if not topology.USING_CYTHON:
        pytest.skip("Cython core not built, parity is trivially satisfied")
    cy = {n: getattr(topology, n) for n in FN}
    cy_null = topology.prominence_null
    saved = sys.modules.get("entrap._topology")
    sys.modules["entrap._topology"] = None      # forces ImportError -> fallback
    try:
        importlib.reload(topology)
        fb = {n: getattr(topology, n) for n in FN}
        fb_null = topology.prominence_null
        yield cy, fb, cy_null, fb_null
    finally:
        if saved is not None:
            sys.modules["entrap._topology"] = saved
        else:
            sys.modules.pop("entrap._topology", None)
        importlib.reload(topology)


@pytest.mark.parity
@pytest.mark.parametrize("seed", range(8))
def test_merge_tree_parity(parity_pair, seed):
    """Verify merge tree parity."""
    cy, fb, _, _ = parity_pair
    rng = np.random.default_rng(seed)
    dens, adj = random_graph(rng, n=16, p=0.3)
    ai, ao = adj_to_csr(adj)
    a = cy["merge_tree_arrays"](dens, ai, ao)
    b = fb["merge_tree_arrays"](dens, ai, ao)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)


@pytest.mark.parity
def test_blocks_and_count_parity(parity_pair):
    """Verify blocks and count parity."""
    cy, fb, _, _ = parity_pair
    rng = np.random.default_rng(99)
    dens, adj = random_graph(rng, n=20, p=0.3)
    ai, ao = adj_to_csr(adj)
    for r in (0, 1, 3):
        np.testing.assert_array_equal(cy["distance_blocks"](ai, ao, r),
                                      fb["distance_blocks"](ai, ao, r))
    for tau in (0.0, 0.5, 2.0):
        assert cy["count_significant"](dens, ai, ao, tau) == fb["count_significant"](dens, ai, ao, tau)


@pytest.mark.parity
def test_null_parity(parity_pair):
    """Verify null parity."""
    cy, fb, cy_null, fb_null = parity_pair
    rng = np.random.default_rng(5)
    dens, adj = random_graph(rng, n=18, p=0.3)
    ai, ao = adj_to_csr(adj)
    blocks = cy["distance_blocks"](ai, ao, 2)
    a = cy_null(dens, ai, ao, blocks, 29, 0.1, np.random.default_rng(7))
    b = fb_null(dens, ai, ao, blocks, 29, 0.1, np.random.default_rng(7))
    assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
