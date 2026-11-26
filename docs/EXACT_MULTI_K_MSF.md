# Exact Multi-K Maximum-Spanning-Forest H0 Engine

## Status

This document describes the first experimental scalability change after ENTRAP 1.0.0. The implementation is exposed by `topology.multiscale_graph_statistics()` and compiled as `entrap._topology.multiscale_msf_statistics()`.

The original V1.0.0 implementation remains available as `topology.multiscale_graph_statistics_reference()` for parity testing.

## Scope of the optimization

The optimization is deliberately restricted to the repeated multi-K H0 summary calculation used by observed-data and null-data evaluations. It returns the same two statistics as the reference kernel for every candidate graph scale:

1. Maximum finite H0 persistence prominence.
2. Number of connected components in the undirected kNN graph.

The selected full graph is still constructed by the original graph code. The full selected merge tree, basin assignment, recursive child graphs, and Leiden CPM optimization are unchanged. In particular, Leiden always receives the full selected-K graph, never the spanning forest.

## Nested kNN graph family

Let `knn_idx[i, j]` be the vertex at zero-based neighbour rank `j` for vertex `i`. The undirected graph at scale `k` is the symmetrized union of the first `k` directed neighbours:

\[
E_k = \bigcup_{i=1}^{N}\{\{i,\operatorname{knn}(i,j)\}:0\leq j<k\}.
\]

For an undirected pair \(e=\{u,v\}\), define its first-inclusion rank as

\[
r(e)=\min\bigl(r_u(v),r_v(u)\bigr),
\]

where a missing directed occurrence is treated as having infinite rank. The pair belongs to `E_k` exactly when \(r(e)<k\). Therefore,

\[
E_{k_1}\subseteq E_{k_2}\quad\text{for }k_1<k_2.
\]

The new kernel builds the maximum-K edge table once, stores each canonical undirected pair once, and retains its first-inclusion rank. It calculates the edge filtration value once and creates one stable descending filtration ordering.

## Filtration values

For the vertex filtration, the current ENTRAP vertex level is

\[
\ell_i=\log\widehat{\rho}_i.
\]

The activation level of an edge in the vertex-only filtration is

\[
w_{ij}^{(0)}=\min(\ell_i,\ell_j).
\]

For mutual reachability, the current ENTRAP edge level is

\[
w_{ij}^{(\mathrm{MR})}
=\min\left(\ell_i,\ell_j,-\log\max(d_{ij},10^{-300})\right).
\]

Both edge-level definitions satisfy

\[
w_{ij}\leq\min(\ell_i,\ell_j),
\]

so an edge cannot become active before both endpoint vertices have been born.

## Why a maximum spanning forest is sufficient for H0

Consider an undirected weighted graph \(G=(V,E,w)\). Let \(F\) be a maximum spanning forest of `G`, obtained by processing edges in non-increasing weight order and accepting an edge only when it joins two distinct disjoint-set components.

For every threshold \(t\), the subgraphs containing edges with weight at least \(t\) have the same connected-component partition:

\[
\operatorname{CC}\bigl(V,\{e\in E:w(e)\geq t\}\bigr)
=
\operatorname{CC}\bigl(V,\{e\in F:w(e)\geq t\}\bigr).
\]

This follows from the maximum-spanning-forest bottleneck property: for any pair of vertices connected in the thresholded original graph, the path between them in a maximum spanning forest has minimum edge weight at least the threshold. The reverse inclusion holds because \(F\subseteq E\).

For the ENTRAP filtrations above, vertex births are retained as separate events and every edge weight is no greater than either endpoint's birth level. Thus the descending vertex/edge event sweep produces the same H0 connected-component filtration as the full weighted graph. The successful Union-Find merges are precisely the edges of the maximum spanning forest; the implementation does not need to materialize the forest explicitly.

### Scope of the equivalence

The theorem concerns H0 connectivity and the persistence summary computed by this kernel. It does not authorize replacing the full graph in Leiden CPM, whose objective depends on all internal graph edges. It also does not replace the selected full merge-tree construction used for mode identities and basin assignment.

Equal-weight events are processed with a stable canonical edge order and vertices are activated before edges at equal filtration levels. This preserves deterministic behavior of the summary kernel. Tests include quantized density values and duplicate coordinates.

## Shared multi-K sweep

The implementation performs the following operations:

1. Encode each directed neighbour relation as a canonical undirected edge key plus its neighbour rank.
2. Sort the encoded keys once.
3. Deduplicate each undirected pair, retaining its minimum directed rank.
4. Calculate each edge filtration value once.
5. Sort the maximum-K edge table once by descending filtration value using a stable sort.
6. For each requested `k`, scan this ordering while ignoring edges whose first-inclusion rank is not less than `k`.
7. Interleave eligible edges with the stable descending vertex-birth order.
8. Use Union-Find to calculate maximum finite persistence and final connected-component count.

The scan is an implicit Kruskal maximum-spanning-forest construction for each nested graph scale. The selected forest edges are not stored because the current consumer only needs the H0 summary. This avoids allocating one forest object per scale.

## Complexity

Let \(K_{\max}\) be the largest requested graph scale, `M` the number of scales, and \(E_{\max}\leq NK_{\max}\) the number of unique undirected edges in the maximum-K graph.

The shared edge table requires

\[
O(NK_{\max}\log(NK_{\max}))
\]

for sorting encoded directed-neighbour records. Edge-level calculation costs \(O(E_{\max}D)\), and the shared filtration sort costs

\[
O(E_{\max}\log E_{\max}).
\]

Each scale then performs one filtered Union-Find event scan:

\[
O(N+E_{\max}\alpha(N)).
\]

The combined cost is therefore

\[
O\left(NK_{\max}\log(NK_{\max})
+E_{\max}D
+E_{\max}\log E_{\max}
+M(N+E_{\max}\alpha(N))\right).
\]

For fixed low-dimensional `D`, fixed `K_max`, and a fixed number of scales `M`, this kernel has sorting-dominated expected/implementation complexity near \(O(N\log N)\). This statement applies to the multi-K H0 summary kernel, not to the complete ENTRAP fit. Exact KD-tree queries, recursive hierarchy depth, null simulations, stability resampling, and Leiden optimization retain their own data-dependent costs.

Memory use is \(O(NK_{\max}+N)\), with one packed rank array, one canonical edge-key array, one edge-level array, one edge ordering, and reusable Union-Find buffers.

## Exactness and fallback

The optimized kernel does not use approximate nearest neighbours, approximate MSTs, reduced Monte Carlo counts, or altered significance thresholds. It consumes the exact neighbour-index matrix already produced by the ENTRAP graph builder.

`topology.multiscale_graph_statistics_reference()` preserves the previous multi-K implementation. The optimized public wrapper falls back to this reference path if the Cython extension is unavailable.

The parity suite compares both mutual-reachability modes, including random graphs, duplicate coordinates, equal density values, unsorted scale arrays, and end-to-end hierarchy outputs.
