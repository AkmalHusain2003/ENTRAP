# ENTRAP 1.1.0.dev1

> **Development build:** `1.1.0.dev1` introduces an exact shared-edge multi-K H0/MSF sweep. The original V1.0.0 multi-scale kernel remains available as a reference implementation for parity testing. This development build is not a new stable release.


**Entropy-based Topological Rescue of Ambiguous Points**

ENTRAP is a hierarchical graph-clustering method designed to identify compact
and extended structures in noisy point clouds. It combines adaptive
nearest-neighbour density estimation, zero-dimensional persistent homology,
Monte Carlo significance calibration, multi-scale stability, and optional
Leiden community refinement.


> **Current status:** research software. ENTRAP is intended for controlled
> experiments and scientific evaluation. It is not a guarantee of cluster
> recovery, astrophysical membership, or tidal-tail confirmation.

## Contents

- [What ENTRAP does](#what-entrap-does)
- [Algorithm overview](#algorithm-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Demo and benchmarks](#demo-and-benchmarks)
- [Project structure](#project-structure)
- [Developer guide](#developer-guide)
- [Native Cython modules](#native-cython-modules)
- [Experimental exact multi-K MSF engine](#experimental-exact-multi-k-msf-engine)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Important limitations](#important-limitations)
- [References](#references)
- [License](#license)

## What ENTRAP does

ENTRAP builds a neighbourhood graph from a finite feature matrix
\(X\in\mathbb{R}^{N\times D}\). It searches for persistent density modes,
calibrates those modes against a null model, and recursively partitions
significant structures. A later membership stage estimates subsample
stability, filters low-density halo points, and can optionally rescue or extend
candidate memberships.

The method is designed around three distinctions:

1. **Graph scale and density scale are separate.** `K_graph` controls graph
   connectivity, while adaptive `K_density` controls the local density
   estimate for each sample.
2. **A topological split must be statistically supported.** Candidate graph
   scales are evaluated against a null ensemble before a recursive split is
   requested.
3. **Core, rescued, and tail-candidate labels are different outputs.** A point
   added by geometric tail propagation is not automatically a dynamically
   confirmed tidal-tail member.

Persistence entropy is calculated from finite persistence lifetimes and stored
as a hierarchical diagnostic. It is not, by itself, the split-selection
statistic.

## Algorithm overview

The implementation follows this sequence.

1. **Input validation.** Convert the feature matrix to a finite, contiguous
   floating-point array and resolve data-size-dependent configuration values.
2. **Neighbourhood construction.** Query exact Euclidean k-nearest neighbours
   with `scipy.spatial.cKDTree`, then store the undirected graph in compressed
   sparse row (CSR) form.
3. **Adaptive density estimation.** Select a local neighbourhood size for
   each point from a persistent slope transition in its log-distance curve.
   Estimate local density and apply the configured Gamma-moment correction
   when applicable.
4. **Multi-scale graph filtration.** Evaluate a predeclared family of graph
   scales. The filtration can use vertex density alone or the optional
   mutual-reachability edge level.
5. **H0 persistent homology.** Process vertices and graph edges in filtration
   order with an elder-rule union-find sweep. The method implemented here is
   zero-dimensional persistent homology, which tracks connected components.
6. **Null calibration.** Estimate persistence and component-count evidence
   using either a homogeneous Poisson reference or graph-block permutation.
   The selected graph scale is corrected for the tested multi-scale family.
7. **Multi-K stability.** For a node proposing multiple modes, each subsample
   rebuilds its local graph and density estimate, evaluates feasible scales,
   and selects its own graph scale. One-mode nodes use the selected-scale
   stability test.
8. **Leiden refinement.** When a node requests a topological split, the
   merge-tree assignment initializes Leiden optimization on an undirected
   simple graph using the Constant Potts Model (CPM).
9. **Recursive children.** Accepted communities become new nodes and receive
   locally rebuilt neighbourhood graphs.
10. **Membership processing.** Extract accepted cores, estimate subsample
    membership probabilities, apply the configured halo filter, and perform
    optional Mahalanobis rescue and geometric tail-candidate propagation.

The detailed equations and implementation notes are described in the
accompanying preprint source when available. The code is the authoritative
specification for a particular release.

### Null models and significance

`tau_null="poisson"` generates homogeneous Poisson reference point clouds and
uses their maximum finite H0 prominence and component-count statistics.
`tau_null="permutation"` estimates graph-distance blocks, permutes density
values within blocks, and recalculates persistence statistics. For the
multi-scale Poisson path, the implementation combines prominence and
component-count empirical p-values at each scale and applies a Bonferroni
correction across the predeclared scale family at each hierarchical node.

For \(M\) graph scales and \(P\) Poisson simulations, the smallest attainable
adjusted p-value under the implemented two-statistic correction is

\[
 p_{\min,\mathrm{adj}}=\frac{2M}{P+1}.
\]

Choose `P` so this Monte Carlo resolution is finer than the requested
significance level. For five scales and `significance=0.05`, `P=199` is only
the theoretical minimum resolution, while `P=399` gives a smaller attainable
step. Larger simulation budgets are recommended for final scientific
experiments.

The correction is applied within each hierarchical node and across that
node's scale family. It is not a global family-wise correction across every
node in the complete recursive tree.

### Mutual reachability

With `mutual_reachability=True`, the edge level is derived from the endpoint
log-densities and the Euclidean edge distance. This can help preserve
non-convex structures such as Two Moons when a large graph neighbourhood
would otherwise create a low-density bridge. It is not enabled by default,
and its usefulness depends on feature scaling and the geometry of the data.

### Leiden and igraph

`igraph` provides the graph representation and `leidenalg` performs Leiden
community optimization. ENTRAP uses a merge-tree partition as the initial
membership and a CPM resolution scan to seek a partition with the requested
number of communities. Leiden is not used as a replacement for topological
significance testing.

Leiden is required by default. If the backend is unavailable, fitting raises
an import error unless the user explicitly sets `allow_fallback=True`. A
fallback partition is not a Leiden result. Set `use_leiden=False` for an
intentional topology-only run.

## Installation

ENTRAP requires Python 3.10 or newer and a C compiler supported by Cython.

### 1. Create an environment

```bash
python -m venv .venv
```

Activate the environment before installing packages.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The runtime requirements include NumPy, SciPy, scikit-learn, Cython,
`igraph`, `leidenalg`, and Matplotlib.

### 3. Build and install ENTRAP

From the repository root:

```bash
python -m pip install -e .
```

The build compiles five Cython extensions. A C/C++ build toolchain is required.
On Windows, install Visual Studio Build Tools with the C++ build tools
workload. On Linux, install the distribution's compiler and Python development
headers. On macOS, install the Xcode Command Line Tools.

To compile the extensions explicitly in place:

```bash
python setup.py build_ext --inplace
```

The project avoids `-ffast-math` so that floating-point comparisons used by
persistence ordering are not relaxed by the compiler.

## Quick start

The estimator accepts a finite two-dimensional NumPy-compatible feature
matrix with shape `(n_samples, n_features)`.

```python
import numpy as np
from entrap import ENTRAP

rng = np.random.default_rng(42)
centers = np.array([
    [0.0, 0.0, 0.0],
    [5.0, 5.0, 0.0],
    [0.0, 6.0, 5.0],
])

clusters = [
    rng.normal(center, 0.35, size=(100, 3))
    for center in centers
]
noise = rng.uniform(-3.0, 8.0, size=(80, 3))
X = np.vstack(clusters + [noise])

model = ENTRAP(
    min_cluster_size=15,
    significance=0.05,
    advanced={
        "seed": 42,
        "P": 399,
        "B": 50,
        "stability_B": 50,
        "pi_runs": 50,
        "n_jobs": -1,
        "parallel_max_workers": 8,
        "parallel_memory_mb": 1024,
    },
)

labels = model.fit_predict(X)
print(model.get_summary())
print("Number of clusters:", model.result_.n_clusters)
print("Leiden diagnostics:", model.leiden_diagnostics_)
```

The example uses the default vertex-density filtration. To test non-convex
structures, enable mutual reachability explicitly:

```python
model = ENTRAP(
    min_cluster_size=15,
    significance=0.05,
    advanced={
        "mutual_reachability": True,
        "P": 399,
        "seed": 42,
    },
)
```

For an intentional run without Leiden optimization:

```python
model = ENTRAP(
    min_cluster_size=15,
    significance=0.05,
    advanced={
        "use_leiden": False,
        "seed": 42,
    },
)
```

## Configuration

The first two parameters are passed directly to the estimator:

```python
ENTRAP(min_cluster_size=10, significance=0.05, advanced={...})
```

Additional parameters are supplied through `advanced`.

| Parameter | Default | Meaning |
|---|---:|---|
| `B` | `50` | Default number of subsamples for stability and membership runs |
| `P` | `199` | Number of Monte Carlo null realizations |
| `f` | `0.8` | Fraction of samples retained in each subsample |
| `metric` | `"euclidean"` | Metric used by the kNN graph builder |
| `seed` | `0` | Random seed |
| `n_jobs` | `-1` | Requested worker count, with memory and worker caps |
| `parallel_max_workers` | `8` | Maximum worker count |
| `parallel_memory_mb` | `1024` | Approximate memory budget used for worker selection |
| `tau_null` | `"poisson"` | Null model, either `"poisson"` or `"permutation"` |
| `mutual_reachability` | `False` | Include edge-distance information in the filtration |
| `normalized_density` | `False` | Use the volume-based kNN density option |
| `adaptive_k` | `True` | Enable per-sample density-neighbourhood selection |
| `adaptive_k_min` | `4` | Minimum adaptive density neighbourhood |
| `adaptive_k_max` | `None` | Maximum adaptive K, resolved from graph scales when omitted |
| `adaptive_k_sensitivity` | `2.5` | Slope-transition sensitivity in the adaptive-K heuristic |
| `adaptive_k_window` | `4` | Number of neighbours in each slope window |
| `adaptive_k_patience` | `2` | Consecutive high-slope windows required to select a knee |
| `multi_k` | `True` | Evaluate the configured graph-scale family |
| `k_scales` | `None` | Optional explicit graph-scale family |
| `k` | `None` | Optional legacy graph-scale cap |
| `significance` | `0.05` | Per-node significance level used by null calibration |
| `stability_B` | `None` | Stability subsample count, defaults to `B` |
| `pi_runs` | `None` | Global membership subsample count, defaults to `B` |
| `pi_threshold` | `None` | Membership probability threshold, defaults to `1 - significance` |
| `use_leiden` | `True` | Enable Leiden CPM refinement |
| `allow_fallback` | `False` | Explicitly allow merge-tree seed fallback if Leiden is unavailable |
| `gamma_range` | `(0.0, 1.0)` | CPM resolution search interval |
| `leiden_gamma_steps` | `17` | Number of log-spaced CPM resolution candidates |
| `leiden_restarts` | `1` | Optimizer restarts per candidate resolution |
| `leiden_iterations` | `-1` | Leiden iteration limit, negative means until no quality improvement |
| `halo_noise` | `True` | Apply the configured halo/core filter |
| `halo_mode` | `"poisson"` | Halo mode, either `"poisson"` or `"fixed"` |
| `compute_pi` | `True` | Calculate global subsample membership stability |
| `tail_membership` | `False` | Enable optional geometric tail-candidate propagation |
| `tail_feature_indices` | `None` | Feature columns used by the geometric tail model |

The complete parameter definitions and validation rules are in
[`entrap/config.py`](entrap/config.py).

### Tail-candidate settings

Tail extension is disabled by default. To enable it, specify feature columns
whose distances have a meaningful interpretation for the structure being
studied:

```python
advanced = {
    "tail_membership": True,
    "tail_feature_indices": (0, 1, 2),
    "tail_k_neighbors": 5,
    "tail_max_hops": 3,
    "tail_radius_factor": 2.0,
    "tail_width_factor": 2.5,
    "tail_min_anisotropy": 1.5,
}
```

These are geometric candidates only. They should not be described as
astrophysically confirmed tidal-tail members without independent physical or
kinematic validation.

## Outputs

After `model.fit(X)` or `model.fit_predict(X)`, the main attributes are:

| Attribute | Description |
|---|---|
| `labels_` | Final labels after probability filtering and optional post-processing; noise is `-1` |
| `core_labels_` | Labels retained after global membership-probability filtering and before rescue |
| `tail_mask_` | Boolean mask identifying points added by geometric tail propagation |
| `tail_labels_` | Labels for tail-candidate points and `-1` elsewhere |
| `result_.probabilities` | Per-sample global membership stability |
| `result_.pi_per_cluster` | Per-sample, per-cluster subsample membership probabilities |
| `result_.noise_rescued` | Number of noise points reassigned by the rescue stage |
| `result_.tree` | Root of the recursive hierarchy with node diagnostics |
| `leiden_diagnostics_` | Counts and per-node records for Leiden calls and accepted optimizations |
| `result_.config` | Resolved estimator configuration |

`model.get_summary()` returns a compact summary of cluster count, noise count,
rescued points, runtime, and cluster sizes.

## Demo and benchmarks

Run the three-dataset demonstration from the repository root:

```bash
python demo_clustering.py
```

The demonstration includes 3D Gaussian blobs with uniform noise, Two Moons,
and standardized Iris data. It prints Leiden diagnostics and saves a
comparison plot. The demo requires `igraph` and `leidenalg` because it is
configured to test the Leiden backend explicitly. It may take several minutes
because its Monte Carlo budget is larger than a quick unit-test configuration.

Run the test suite:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -ra
```

Run the modular Cython benchmark:

```bash
python benchmarks/benchmark_modular_cython.py --repeat 5 --n 20000 --halo-M 200000
```

Benchmark results depend on CPU, compiler, Python, NumPy, SciPy, and worker
configuration. Do not treat the benchmark notes in this repository as a
performance guarantee for another machine or astronomical catalogue.

## Project structure

```text
ENTRAP-1.1.0-dev1/
├── .github/workflows/tests.yml
├── entrap/
│   ├── __init__.py
│   ├── config.py
│   ├── estimator.py
│   ├── graph.py
│   ├── filtration.py
│   ├── mergetree.py
│   ├── significance.py
│   ├── stability.py
│   ├── leiden.py
│   ├── halo.py
│   ├── rescue.py
│   ├── tails.py
│   ├── tree.py
│   ├── results.py
│   ├── diagnostics.py
│   ├── memory.py
│   ├── parallel.py
│   ├── tracker.py
│   ├── topology.py
│   ├── _topology.pyx
│   ├── _density.pyx
│   ├── _null.pyx
│   ├── _halo.pyx
│   └── _membership.pyx
├── tests/
├── benchmarks/
├── docs/ALGORITHM_FLOWCHART.md
├── demo_clustering.py
├── example.py
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-test.txt
├── requirements-benchmark.txt
├── pytest.ini
├── MANIFEST.in
├── README.md
├── CONTRIBUTING.md
├── VALIDATION.md
└── VERSION.txt
```

## Developer guide

The contributor instructions are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
They define the one-statement-per-line style, one-sentence function
documentation, literature-attribution rules, and numerical-parity requirements
for native kernels.

The stage-by-stage algorithm diagram is available in
[`docs/ALGORITHM_FLOWCHART.md`](docs/ALGORITHM_FLOWCHART.md).

## Native Cython modules

| Extension | Main operations |
|---|---|
| `_topology.pyx` | Union-find sweeps, H0 persistence, merge-tree construction, edge filtration, CSR construction, and multi-scale graph statistics |
| `_density.pyx` | Adaptive-K selection and local density calculation from cached kNN distances |
| `_null.pyx` | CSR breadth-first traversal and online correlation accumulation for graph-block scale estimation |
| `_halo.pyx` | Variable-K Poisson halo transforms and bounded-chunk fixed-K transforms |
| `_membership.pyx` | Mahalanobis rescue assignment and synchronous geometric tail propagation |

Python modules provide validation, orchestration, configuration, and fallback
implementations. SciPy performs kNN queries, NumPy performs vectorized array
operations, and `leidenalg` performs the Leiden optimization in its own native
backend. ENTRAP does not contain a separate Cython implementation of Leiden.

## Validation and reproducibility

For reproducible experiments, record at least:

- ENTRAP version and repository commit.
- Python, NumPy, SciPy, scikit-learn, Cython, igraph, and leidenalg versions.
- Operating system and C/C++ compiler.
- Input feature definition, units, scaling, and any uncertainty treatment.
- All explicit estimator parameters and the random seed.
- Whether the Cython extensions and Leiden backend were active.
- The null model, Monte Carlo budgets, graph-scale family, and mutual-reachability setting.

The Cython implementation retains Python/NumPy reference paths for supported
kernels. Numerical comparisons can differ at floating-point round-off level
because operation order may differ, so scientific regression tests should
compare both numerical tolerances and discrete decisions such as selected K,
mode counts, and membership labels.

## Experimental exact multi-K MSF engine

The `1.1.0.dev1` development branch adds a shared-edge, exact maximum-spanning-forest H0 summary kernel. It builds the nested maximum-K kNN edge table once, computes filtration levels once, and reuses one stable edge ordering across all requested K scales. The original implementation remains available through `topology.multiscale_graph_statistics_reference()` for parity checks.

This optimization is restricted to multi-K H0 prominence and connected-component summaries. The selected full kNN graph, full merge tree, recursive hierarchy, null-calibration procedure, stability resampling, and Leiden CPM input graph are unchanged. The implementation uses an implicit Kruskal maximum-spanning-forest sweep on the sparse ENTRAP kNN graphs. It does not implement HDBSCAN's Dual-Tree Boruvka algorithm or construct a complete mutual-reachability graph. See [`docs/EXACT_MULTI_K_MSF.md`](docs/EXACT_MULTI_K_MSF.md) for the equivalence argument, complexity, and scope.

## Important limitations

- The implemented persistent homology is H0 on graph filtrations. ENTRAP does
  not currently compute higher-dimensional H1 or H2 persistence.
- Adaptive K is an ENTRAP overlapping-window slope heuristic. It is not a
  direct implementation of the ABIDE estimator or a universal statistically
  optimal neighbourhood selector.
- The Gamma-moment correction is a leading-order scale correction under its
  stated Poisson approximation, not a general proof of unbiased density
  estimation.
- The multi-K empirical correction is applied within each hierarchical
  node. Recursive testing across all nodes is not covered by one global
  family-wise correction.
- Multi-K stability reuses parent-calibrated Poisson prominence thresholds
  and component evidence for efficiency. It is an approximation rather than
  a full subsample-size-specific null recalibration.
- The Poisson null assumes a homogeneous reference process. Strongly
  non-uniform backgrounds, boundaries, selection effects, and measurement
  uncertainties require dedicated validation or a suitable alternative null.
- Leiden optimizes graph communities after topological evidence proposes a
  split. It does not establish the physical reality of an astronomical group.
- Halo and rescue decisions depend on the selected feature metric and model
  assumptions. They should be validated against simulations and known members.
- Geometric tail propagation is optional and exploratory. It does not prove
  that candidate points are dynamically associated with a cluster.
- ENTRAP does not automatically account for heteroscedastic astrometric
  uncertainties. Users must choose and document their feature representation
  and metric.
- No claim is made that ENTRAP universally outperforms HDBSCAN. Comparisons
  require matched preprocessing, parameters, null assumptions, and metrics.

## References

1. **Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002).** Topological
   Persistence and Simplification. *Discrete & Computational Geometry*, 28,
   511–533. [https://doi.org/10.1007/s00454-002-2885-2](https://doi.org/10.1007/s00454-002-2885-2)
2. **Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).** Estimating
   the intrinsic dimension of datasets by a minimal neighborhood information.
   *Scientific Reports*, 7, 12140.
   [https://doi.org/10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y)
3. **Traag, V. A., Waltman, L., & van Eck, N. J. (2019).** From Louvain to
   Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9,
   5233. [https://doi.org/10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)
4. **Mahalanobis, P. C. (1936).** On the generalized distance in statistics.
   *Proceedings of the National Institute of Sciences of India*, 2, 49–55.
5. **Shannon, C. E. (1948).** A Mathematical Theory of Communication.
   *Bell System Technical Journal*, 27, 379–423 and 623–656.
6. **Csardi, G., & Nepusz, T. (2006).** The igraph software package for
   complex network research. *InterJournal, Complex Systems*, 1695.
7. **Behnel, S., Bradshaw, R., Citro, C., Dalcin, L., Seljebotn, D. S., & Smith,
   K. (2011).** Cython: The Best of Both Worlds. *Computing in Science &
   Engineering*, 13(2), 31–39.
   [https://doi.org/10.1109/MCSE.2010.118](https://doi.org/10.1109/MCSE.2010.118)
8. **Harris, C. R., et al. (2020).** Array programming with NumPy. *Nature*,
   585, 357–362. [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
9. **Virtanen, P., et al. (2020).** SciPy 1.0: fundamental algorithms for
   scientific computing in Python. *Nature Methods*, 17, 261–272.
   [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
10. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python.
    *Journal of Machine Learning Research*, 12, 2825–2830.
11. **Loftsgaarden, D. O., & Quesenberry, C. P. (1965).** A Nonparametric Estimate of a Multivariate Density Function. *The Annals of Mathematical Statistics*, 36(3), 1049–1051. [https://doi.org/10.1214/aoms/1177700079](https://doi.org/10.1214/aoms/1177700079)

12. **McInnes, L., & Healy, J. (2017).** Accelerated Hierarchical Density Based Clustering. *2017 IEEE International Conference on Data Mining Workshops*, 33–42. [https://doi.org/10.1109/ICDMW.2017.12](https://doi.org/10.1109/ICDMW.2017.12)
13. **Kruskal, J. B. (1956).** On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem. *Proceedings of the American Mathematical Society*, 7(1), 48–50. [https://doi.org/10.1090/S0002-9939-1956-0078686-7](https://doi.org/10.1090/S0002-9939-1956-0078686-7)

The references above describe the mathematical foundations or software
components used by ENTRAP. They do not imply that every ENTRAP design choice is
part of those original publications. In particular, adaptive-K knee selection,
multi-K significance selection, and the ENTRAP hierarchy are project-specific
combinations and should be cited to the ENTRAP manuscript when it is formally
published.

## License

No `LICENSE` file is included in this repository. The project license has not
yet been selected. Before publishing the repository as an open-source project,
choose a license and add its complete text. Without a license, users should not
assume that they have permission to redistribute or modify the source code.
