# ENTRAP 1.1.0.dev1 — Development Validation

This development build introduces the exact shared-edge multi-K H0/MSF summary engine. It does not alter the ENTRAP statistical model or its public default parameters.

## What changed

- Builds the maximum-K undirected kNN edge table once.
- Stores the first directed-neighbour rank for each undirected edge.
- Computes each edge filtration value once.
- Sorts the maximum-K edge family once.
- Reuses that stable ordering for every requested K.
- Uses a Kruskal/Union-Find sweep that implicitly constructs the maximum spanning forest for each K while calculating the H0 summary.
- Retains the previous multi-K kernel as `topology.multiscale_graph_statistics_reference()`.

The full selected graph and full selected merge tree are still constructed by the existing code. Leiden CPM continues to receive the full selected-K graph. No approximate nearest neighbours, approximate MST, reduced Monte Carlo count, or altered significance threshold is introduced.

## Exactness checks

- 1,000 randomized graph datasets were compared in both vertex-filtration and mutual-reachability modes: 2,000 graph-filtration comparisons passed.
- Regression cases include random point clouds, duplicate coordinates, zero-distance edges, quantized density values, fully tied density values, unsorted scale lists, and component-count monotonicity.
- End-to-end Blobs and Two Moons comparisons matched the reference kernel for labels, core labels, cluster count, selected K, `m_star`, adjusted p-values, component counts, and recursive tree structure.
- Full regression suite: **235 passed, 5 skipped**.
- The five skipped tests require `igraph` and `leidenalg`, which are not installed in the build environment. The new H0 engine does not change the Leiden input graph or Leiden implementation.

## Kernel benchmark

Environment: Linux x86_64, Python 3.13, GCC `-O3`, five timing repetitions.

Input preparation and kNN construction were performed before timing. The measurements include the multi-K edge-table construction, edge filtration calculation, ordering, and all scale sweeps. They exclude null-cloud generation, density estimation, recursive tree construction, Leiden, and global membership resampling.

| Filtration | Reference median | Optimized median | Speed-up |
|---|---:|---:|---:|
| Vertex filtration (`mreach=False`) | 0.2882 s | 0.1041 s | 2.77x |
| Mutual reachability (`mreach=True`) | 0.2932 s | 0.1149 s | 2.55x |

Benchmark dimensions:

- `N = 32,000`
- `D = 5`
- `K_max = 21`
- Graph scales = `{6, 9, 13, 17, 21}`

These are kernel-level results, not an end-to-end ENTRAP fit speed-up. Overall speed-up will be smaller because kNN queries, null generation, stability, recursive nodes, and Leiden still contribute runtime.

## Complexity

Let `E_max <= N*K_max` denote the unique edge count in the maximum-K graph and `M` the number of graph scales. The shared edge table and filtration ordering cost approximately

\[
O(NK_{\max}\log(NK_{\max}) + E_{\max}D + E_{\max}\log E_{\max}).
\]

The per-scale filtered Union-Find sweeps cost

\[
O(M(N + E_{\max}\alpha(N))).
\]

For fixed low-dimensional `D`, fixed `K_max`, and fixed `M`, the multi-K H0 kernel is sorting-dominated and behaves near `O(N log N)`. This does not make the entire ENTRAP fit worst-case `O(N log N)`. Exact kNN, null simulations, resampling, recursive hierarchy depth, and Leiden retain data-dependent costs.

## Reproducibility

The reference kernel can be selected temporarily for parity experiments:

```python
from entrap import topology

optimized = topology.multiscale_graph_statistics
try:
    topology.multiscale_graph_statistics = (
        topology.multiscale_graph_statistics_reference
    )
    # Run a reference experiment here.
finally:
    topology.multiscale_graph_statistics = optimized
```

For reproducible performance comparisons, record the ENTRAP version, Python/NumPy/SciPy/Cython versions, compiler, CPU, seed, data representation, graph scales, and filtration mode.
