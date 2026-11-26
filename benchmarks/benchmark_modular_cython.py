#!/usr/bin/env python3
"""Compare modular Cython kernels with their retained Python/NumPy references.

Examples:
    python benchmarks/benchmark_modular_cython.py --repeat 3
    python benchmarks/benchmark_modular_cython.py --repeat 5 --n 50000 --halo-M 200000

The benchmark is implementation-only. It does not alter ENTRAP parameters or
statistical procedures. The halo path is memory-bounded
its vectorized NumPy
reference can be slightly faster on some machines, so report time and memory
trade-offs rather than assuming every extension must win every metric.
"""
from __future__ import annotations
import argparse
import statistics
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entrap import graph, significance, halo, rescue, tails
from entrap.graph import edges_to_csr
from entrap.config import EntrapConfig


def timed(fn, repeat):
    """Run one timed benchmark operation and return its result."""
    values, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t0)
    return statistics.median(values), result


def main():
    """Run the demonstration workflow and save the clustering comparison figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--halo-M", type=int, default=200000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261124)

    print("Native modules:", {
        "density": graph._HAS_DENSITY_NATIVE,
        "null": significance._HAS_NULL_NATIVE,
        "halo": halo._HAS_HALO_NATIVE,
        "membership": rescue._HAS_MEMBERSHIP_NATIVE,
    })

    # Density: compare compiled adaptive K+density with retained NumPy reference.
    dist = np.sort(rng.lognormal(size=(args.n, 28)), axis=1)
    tn, rn = timed(lambda: graph.adaptive_knn_density_from_distances(
        dist, 4, 2.5, 4, 2, False, 3.0, True), args.repeat)
    tp, rp = timed(lambda: graph._adaptive_knn_density_from_distances_python(
        dist, 4, 2.5, 4, 2, False, 3.0, True), args.repeat)
    density_exact = np.array_equal(rn[1], rp[1]) and np.allclose(rn[0], rp[0], rtol=2e-13, atol=2e-13)
    print(f"density: native={tn:.6f}s reference={tp:.6f}s speedup={tp/tn:.2f}x parity={density_exact}")

    # Null: same fixed graph and same RNG-selected source sequence.
    n = min(args.n, 20000)
    X = rng.normal(size=(n, 3))
    _, ki = graph._query_knn_euclidean(X, 12)
    edges = graph.topology.knn_edges_csr(ki)[0]
    ai, ao = edges_to_csr(n, edges)
    density_values = rng.normal(size=n)
    tn, rn = timed(lambda: significance.estimate_block_scale(
        density_values, (ai, ao), 6, 100, np.random.default_rng(818)), args.repeat)
    tp, rp = timed(lambda: significance._estimate_block_scale_python(
        density_values, (ai, ao), 6, 100, np.random.default_rng(818)), args.repeat)
    print(f"null/BFS: native={tn:.6f}s reference={tp:.6f}s speedup={tp/tn:.2f}x parity={rn == rp}")

    # Halo: compare the exact same seeded NumPy stream. Cython route processes
    # bounded chunks, reference route materializes the original full matrices.
    M, k, d, seed = args.halo_M, 20, 3, 7331

    def halo_native():
        """Benchmark the chunked native fixed-K halo transform."""
        r = np.random.default_rng(seed)
        g = halo._stream_poisson_g(r, M, k, d, chunk_size=65536)
        return np.median(g), np.quantile(g, 0.999), g.std()

    def halo_reference():
        """Benchmark the NumPy reference fixed-K halo transform."""
        r = np.random.default_rng(seed)
        arrivals = np.cumsum(r.exponential(1.0, size=(M, k)), axis=1)
        g = -np.log(np.mean(arrivals ** (1.0 / d), axis=1))
        return np.median(g), np.quantile(g, 0.999), g.std()
    tn, rn = timed(halo_native, args.repeat)
    tp, rp = timed(halo_reference, args.repeat)
    halo_exact = np.allclose(rn, rp, rtol=2e-13, atol=2e-13)
    print(f"halo/fixed-K: native-chunked={tn:.6f}s reference-full={tp:.6f}s ratio={tp/tn:.2f}x parity={halo_exact}, chunked path bounds peak temporaries")

    # Variable-K halo draws use a fused Cython row kernel and avoid the full
    # gaps/radial/cumulative matrices needed by the reference implementation.
    ks = np.asarray([4, 6, 9, 12, 15, 20], dtype=np.int32)

    def halo_native_variable():
        """Benchmark the native variable-K halo transform."""
        r = np.random.default_rng(seed)
        chosen = r.choice(ks, size=M, replace=True)
        g = halo._stream_poisson_g(r, M, int(ks.max()), d, chosen, chunk_size=65536)
        return np.median(g), np.quantile(g, 0.999), g.std()

    def halo_reference_variable():
        """Benchmark the NumPy variable-K halo reference."""
        r = np.random.default_rng(seed)
        chosen = r.choice(ks, size=M, replace=True)
        gaps = np.cumsum(r.exponential(1.0, size=(M, int(ks.max()))), axis=1)
        radial = gaps ** (1.0 / d)
        cumulative = np.cumsum(radial, axis=1)
        means = cumulative[np.arange(M), chosen - 1] / chosen
        g = -np.log(np.maximum(means, 1e-300))
        return np.median(g), np.quantile(g, 0.999), g.std()
    tvn, rvn = timed(halo_native_variable, args.repeat)
    tvp, rvp = timed(halo_reference_variable, args.repeat)
    print(f"halo/variable-K: native={tvn:.6f}s reference={tvp:.6f}s speedup={tvp/tvn:.2f}x parity={np.allclose(rvn,rvp,rtol=2e-13,atol=2e-13)}")

    # Rescue: Gaussian fitting remains NumPy/SciPy, candidate Mahalanobis loops
    # are native vs the retained Python loop.
    nr, dim, C = min(args.n, 30000), 6, 8
    XR = rng.normal(size=(nr, dim))
    lab = np.full(nr, -1, dtype=np.int64)
    basins = {}
    cluster_size = max(100, nr // 20)
    for c in range(C):
        ix = np.arange(c * cluster_size, (c + 1) * cluster_size, dtype=np.int64)
        basins[c] = ix
        lab[ix] = c
    pi = rng.random((nr, C))
    proms = {c: float(rng.random()) for c in range(C)}
    cfg = EntrapConfig.from_params(10, 0.05, {"alpha_gate": 0.99, "ridge": 1e-5}).resolve(nr)
    native_flag = rescue._HAS_MEMBERSHIP_NATIVE
    tn, rn = timed(lambda: rescue.rescue(XR, lab, pi, basins, proms, cfg), args.repeat)
    rescue._HAS_MEMBERSHIP_NATIVE = False
    try:
        tp, rp = timed(lambda: rescue.rescue(XR, lab, pi, basins, proms, cfg), args.repeat)
    finally:
        rescue._HAS_MEMBERSHIP_NATIVE = native_flag
    print(f"rescue: native={tn:.6f}s reference={tp:.6f}s speedup={tp/tn:.2f}x parity={np.array_equal(rn[0], rp[0]) and rn[1] == rp[1]}")

    # Tail: compare only the batch propagation after identical neighbour query
    # and geometric mask construction via public function, both functions do
    # the same SciPy PCA/quantile/kNN setup.
    ncore = max(100, nr // 4)
    core = np.column_stack([rng.normal(0.0, 1.3, ncore), rng.normal(0.0, 0.08, ncore)])
    ext = np.column_stack([np.linspace(2.5, 5.0, max(20, nr // 6)), rng.normal(0.0, 0.04, max(20, nr // 6))])
    other = rng.normal([0.0, 2.5], [1.0, 0.1], size=(max(20, nr - len(core) - len(ext)), 2))
    XT = np.vstack([core, ext, other])
    LT = np.full(len(XT), -1, dtype=np.int64)
    LT[:len(core)] = 0
    BT = {0: np.arange(len(core), dtype=np.int64)}
    kw = dict(k_neighbors=5, max_hops=3, radius_factor=3.0, width_factor=3.0, min_anisotropy=1.5)
    tail_flag = tails._HAS_MEMBERSHIP_NATIVE
    tn, rn = timed(lambda: tails.extend_tidal_tail_candidates(XT, LT, BT, (0, 1), **kw), args.repeat)
    tails._HAS_MEMBERSHIP_NATIVE = False
    try:
        tp, rp = timed(lambda: tails.extend_tidal_tail_candidates(XT, LT, BT, (0, 1), **kw), args.repeat)
    finally:
        tails._HAS_MEMBERSHIP_NATIVE = tail_flag
    print(f"tails: native={tn:.6f}s reference={tp:.6f}s speedup={tp/tn:.2f}x parity={np.array_equal(rn[0],rp[0]) and np.array_equal(rn[1],rp[1])}")


if __name__ == "__main__":
    main()
