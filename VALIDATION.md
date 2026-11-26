# ENTRAP 1.0.0 validation report

## Environment

The validation was run in the local Linux build environment with Python 3.13.5,
NumPy 2.3.5, SciPy 1.17.0 and Cython 3.2.4. GCC compilation succeeded.
Timings below are kernel-level microbenchmarks on this environment; they are not
end-to-end demo timings and should not be generalized to other hardware.

## Regression and parity

- Full suite with all five Cython extensions built: **225 passed, 5 skipped**.
- Selected fallback suite with the four new modules (`_density`, `_null`, `_halo`,
  `_membership`) temporarily unavailable and `_topology` still built: **29 passed,
  6 skipped**. The skips include native-module-specific tests.
- A complete suite with every native module removed was not used as a release
  gate; the pure-Python topology fallback is substantially slower.
- Cython/Python parity checks cover adaptive K and density (including duplicate
  distances, normalized and inverse-mean estimators), CSR block-scale traversal,
  fixed- and variable-K Poisson halo transforms with the same seeded NumPy RNG
  stream, rescue labels/counts, and tail-candidate labels/masks.
- Regression tests for the established multi-scale implementation remained green.
- Five tests requiring a live `igraph`/`leidenalg` installation were skipped in
  this build environment. The Leiden/CPM implementation was not changed by 1.0.0.

## Kernel benchmark

The following figures are medians from five local runs (N=20,000 for density and
CSR traversal; rescue up to N=30,000; 200,000 Poisson draws for halo). The exact
input and flags are defined in `benchmarks/benchmark_modular_cython.py`.

| Kernel | Native path | Python/NumPy reference | Speed-up | Parity |
|---|---:|---:|---:|---|
| Adaptive density | 0.02332 s | 0.03031 s | 1.30x | Yes |
| CSR block-scale BFS | 0.00569 s | 0.20837 s | 36.64x | Yes |
| Mahalanobis rescue loop | 0.00381 s | 0.27135 s | 71.18x | Yes |
| Tail propagation | 0.11394 s | 0.15754 s | 1.38x | Yes |
| Variable-K halo transform | 0.07393 s | 0.09291 s | 1.26x | Yes, within 3.4e-16 |

Fixed-K halo is a deliberate time/memory trade-off: bounded NumPy chunks took
0.08689 s compared with 0.07354 s for the full-matrix NumPy reference (about
15% slower), but peak RSS in a separate M=200,000, K=20 test fell from 226.1 MiB
to 195.7 MiB (13.4% lower). The variable-K fused Cython path reduced peak RSS
from 262.2 MiB to 185.4 MiB (29.3% lower) in its corresponding test.

## Interpretation and limitations

- BFS and rescue show large speed-ups because the reference implementations
  perform Python-level graph traversal or per-point/per-cluster loops. Actual
  end-to-end gains depend on how often these functions are called.
- Density and tail propagation show more modest, workload-dependent gains
  because surrounding NumPy/SciPy operations already execute in native code.
  In a shorter three-repeat exploratory run, density was approximately at
  parity; the five-repeat median reported above was 1.30x. Rebenchmark on the
  target machine before assuming a consistent density speed-up.
- Fixed-K halo is not faster in this workload. It is chunked to constrain peak
  temporary memory. Users prioritising wall time over RAM can compare the
  retained fallback path on their own workload.
- The figures are not an HDBSCAN comparison, not a full `demo_clustering.py`
  runtime comparison, and not a guarantee for astronomical catalogue sizes.
- The live Leiden backend was not available in this test environment; five
  integration tests were skipped. 1.0.0 does not modify Leiden code or CPM
  parameters.
- Windows/MSVC compilation was not performed in this Linux environment. The
  source package includes `.pyx` files and compiles extensions locally.
