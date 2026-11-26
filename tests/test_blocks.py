import numpy as np
import pytest
from entrap import topology
from entrap.graph import adj_to_csr
from entrap.significance import graph_distance_blocks
from helpers import random_graph, path_graph


def test_all_vertices_assigned():
    """Verify all vertices assigned."""
    rng = np.random.default_rng(3)
    dens, adj = random_graph(rng, n=25, p=0.2)
    blocks = graph_distance_blocks(adj, 2)
    assert len(blocks) == len(adj)
    assert np.all(blocks >= 0)


def test_large_radius_one_block_per_component():
    """Verify large radius one block per component."""
    dens, adj = path_graph([5., 4., 3., 2., 1., 0.], [(0, 1), (1, 2), (3, 4), (4, 5)])
    blocks = graph_distance_blocks(adj, 100)
    assert len(np.unique(blocks)) == 2
    assert blocks[0] == blocks[1] == blocks[2]
    assert blocks[3] == blocks[4] == blocks[5]
    assert blocks[0] != blocks[3]


def test_blocks_are_connected():
    """Verify blocks are connected."""
    rng = np.random.default_rng(11)
    dens, adj = random_graph(rng, n=30, p=0.25)
    blocks = graph_distance_blocks(adj, 2)
    for b in np.unique(blocks):
        idx = set(np.where(blocks == b)[0].tolist())
        start = next(iter(idx))
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u in idx and u not in seen:
                    seen.add(u)
                    stack.append(u)
        assert seen == idx, "block must induce a connected subgraph"


def test_radius_zero_gives_singletons():
    """Verify radius zero gives singletons."""
    rng = np.random.default_rng(5)
    dens, adj = random_graph(rng, n=12, p=0.4)
    ai, ao = adj_to_csr(adj)
    blocks = topology.distance_blocks(ai, ao, 0)
    # radius 0: seed only: every vertex its own block
    assert len(np.unique(blocks)) == 12
