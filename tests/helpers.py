"""Independent oracles and graph generators for the ENTRAP test suite."""
import numpy as np


def random_graph(rng, n=15, p=0.25):
    """Generate a random undirected graph for topology tests."""
    dens = rng.permutation(np.arange(1, n + 1)).astype(np.float64) + rng.uniform(0, 0.5, n)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    return np.ascontiguousarray(dens), adj


def path_graph(densities, edges):
    """Generate a path graph with a configurable number of vertices."""
    n = len(densities)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return np.ascontiguousarray(np.asarray(densities, dtype=np.float64)), adj


def oracle_persistence(density, adj):
    """Compute reference H0 persistence pairs with a transparent Python implementation."""
    n = len(density)
    levels = np.unique(density)[::-1]
    prev = []          # list of (component_set, birth_level)
    pairs = []
    for lam in levels:
        active = set(int(i) for i in range(n) if density[i] >= lam - 1e-12)
        seen, comps = set(), []
        for s in sorted(active):
            if s in seen:
                continue
            stack, comp = [s], set()
            seen.add(s)
            while stack:
                v = stack.pop()
                comp.add(v)
                for u in adj[v]:
                    if u in active and u not in seen:
                        seen.add(u)
                        stack.append(u)
            comps.append(comp)
        cur = []
        for comp in comps:
            parents = [(ps, pb) for (ps, pb) in prev if ps & comp]
            if not parents:
                cur.append((comp, float(lam)))
            elif len(parents) == 1:
                cur.append((comp, parents[0][1]))
            else:
                parents.sort(key=lambda t: -t[1])
                for _, pb in parents[1:]:
                    pairs.append((float(pb), float(lam)))
                cur.append((comp, parents[0][1]))
        prev = cur
    return pairs, len(prev)


def pairs_multiset(pairs):
    """Normalize persistence pairs for order-independent comparison."""
    arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    return sorted((round(float(a), 9), round(float(b), 9)) for a, b in arr)


def make_blobs(n_per=40, std=0.2, centers=((0., 0., 0.), (10., 0., 0.), (20., 0., 0.)), seed=0):
    """Generate Gaussian blobs with known class labels."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(c, std, size=(n_per, 3)) for c in centers]
    X = np.vstack(parts)
    truth = np.repeat(np.arange(len(centers)), n_per)
    return X, truth


def uniform_field(n=120, dim=3, seed=1):
    """Generate a homogeneous uniform point field."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, dim))


def partition_equal(a, b):
    """Compare two partitions while ignoring arbitrary label IDs."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        return False
    mapping = {}
    for x, y in zip(a, b):
        if x in mapping and mapping[x] != y:
            return False
        mapping[x] = y
    return len(set(mapping.values())) == len(set(mapping.keys()))


def edges_from_adj(adj):
    """Convert adjacency lists into undirected edge pairs."""
    return np.array([(i, j) for i in range(len(adj)) for j in adj[i] if i < j], dtype=np.int64).reshape(-1, 2)


def oracle_persistence_edges(density, edges, elev):
    """Compute reference H0 persistence on an edge filtration."""
    n = len(density)
    edges = np.asarray(edges).reshape(-1, 2)
    levels = np.unique(np.concatenate([np.asarray(density), np.asarray(elev)]))[::-1]
    prev, pairs = [], []
    for lam in levels:
        active = set(int(i) for i in range(n) if density[i] >= lam - 1e-12)
        nbr = {v: [] for v in active}
        for (a, b), le in zip(edges, elev):
            if le >= lam - 1e-12 and int(a) in active and int(b) in active:
                nbr[int(a)].append(int(b))
                nbr[int(b)].append(int(a))
        seen, comps = set(), []
        for s in sorted(active):
            if s in seen:
                continue
            stack, comp = [s], set()
            seen.add(s)
            while stack:
                v = stack.pop()
                comp.add(v)
                for u in nbr[v]:
                    if u not in seen:
                        seen.add(u)
                        stack.append(u)
            comps.append(comp)
        cur = []
        for comp in comps:
            parents = [(ps, pb) for (ps, pb) in prev if ps & comp]
            if not parents:
                cur.append((comp, float(lam)))
            elif len(parents) == 1:
                cur.append((comp, parents[0][1]))
            else:
                parents.sort(key=lambda t: -t[1])
                for _, pb in parents[1:]:
                    pairs.append((float(pb), float(lam)))
                cur.append((comp, parents[0][1]))
        prev = cur
    return pairs, len(prev)
