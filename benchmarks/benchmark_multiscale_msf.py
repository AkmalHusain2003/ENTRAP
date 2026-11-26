"""Benchmark the original and exact shared-edge multi-K H0 kernels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrap import topology


def make_knn(X: np.ndarray, k: int) -> np.ndarray:
    """Build the exact self-excluding neighbour-index matrix used by ENTRAP."""
    n = len(X)
    _, indices = cKDTree(X).query(X, k=k + 1)
    indices = np.asarray(indices, dtype=np.int64).reshape(n, k + 1)
    output = np.empty((n, k), dtype=np.int64)
    for row in range(n):
        selected = indices[row, indices[row] != row]
        if len(selected) < k:
            _, extra = cKDTree(X).query(X[row], k=min(n, k + 2))
            extra = np.atleast_1d(extra)
            selected = extra[extra != row]
        output[row] = selected[:k]
    return output


def timed_call(function, repeats: int) -> tuple[float, tuple]:
    """Return the median runtime and last result for repeated kernel calls."""
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result


def main() -> None:
    """Run an exact parity and runtime comparison for both filtration modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32000)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--kmax", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(20261126)
    X = rng.normal(size=(args.n, args.dimension))
    knn_idx = make_knn(X, min(args.kmax, args.n - 1))
    ell = rng.normal(size=args.n)
    scales = np.unique(
        np.array(
            [6, 9, 13, 17, args.kmax],
            dtype=np.int64,
        )
    )

    print(f"N={args.n}, D={args.dimension}, Kmax={args.kmax}")
    print(f"Scales={scales.tolist()}, repeats={args.repeats}")
    print(f"Cython active={topology.USING_CYTHON}")

    for mreach in (False, True):
        reference_time, reference_result = timed_call(
            lambda: topology.multiscale_graph_statistics_reference(
                X, knn_idx, ell, scales, mreach=mreach
            ),
            args.repeats,
        )
        optimized_time, optimized_result = timed_call(
            lambda: topology.multiscale_graph_statistics(
                X, knn_idx, ell, scales, mreach=mreach
            ),
            args.repeats,
        )
        np.testing.assert_allclose(
            optimized_result[0], reference_result[0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_array_equal(
            optimized_result[1], reference_result[1]
        )
        print(
            f"mreach={mreach}: reference={reference_time:.4f}s, "
            f"optimized={optimized_time:.4f}s, "
            f"speedup={reference_time / optimized_time:.3f}x"
        )


if __name__ == "__main__":
    main()
