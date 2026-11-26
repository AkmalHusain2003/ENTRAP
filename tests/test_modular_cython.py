"""Parity tests for the four modular Cython kernels added in ENTRAP 1.0.0."""
from __future__ import annotations

import numpy as np
import pytest

from entrap import graph, significance, halo, rescue, tails

_NATIVE_AVAILABLE = all((
    graph._HAS_DENSITY_NATIVE,
    significance._HAS_NULL_NATIVE,
    halo._HAS_HALO_NATIVE,
    rescue._HAS_MEMBERSHIP_NATIVE,
    tails._HAS_MEMBERSHIP_NATIVE,
))
pytestmark = pytest.mark.skipif(
    not _NATIVE_AVAILABLE, reason="modular Cython extensions are not built"
)


def test_density_cython_matches_numpy_reference_random_and_duplicate_distances():
    """Verify density cython matches numpy reference random and duplicate distances."""
    assert graph._HAS_DENSITY_NATIVE
    for seed in range(12):
        rng = np.random.default_rng(seed)
        dist = np.sort(rng.lognormal(mean=-0.2, sigma=0.8, size=(173, 24)), axis=1)
        if seed % 2 == 0:
            dist[:5, :4] = 0.0
        for k_min, sens, window, patience in ((4, 2.5, 4, 2), (3, 1.7, 5, 3)):
            got_k = graph.adaptive_k_from_distances(dist, k_min, sens, window, patience)
            ref_k = graph._adaptive_k_from_distances_python(dist, k_min, sens, window, patience)
            np.testing.assert_array_equal(got_k, ref_k)
            for normalized in (False, True):
                got_d, got_ks = graph.adaptive_knn_density_from_distances(
                    dist, k_min, sens, window, patience, normalized, 2.6, True
                )
                ref_d, ref_ks = graph._adaptive_knn_density_from_distances_python(
                    dist, k_min, sens, window, patience, normalized, 2.6, True
                )
                np.testing.assert_array_equal(got_ks, ref_ks)
                np.testing.assert_allclose(got_d, ref_d, rtol=2e-13, atol=2e-13)


def test_null_cython_block_scale_matches_python_reference():
    """Verify null cython block scale matches python reference."""
    assert significance._HAS_NULL_NATIVE
    from entrap.graph import edges_to_csr
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 90
        edges = np.asarray([(i, j) for i in range(n) for j in (i - 1, i + 1, i + 3)
                            if 0 <= j < n], dtype=np.int64)
        ai, ao = edges_to_csr(n, edges)
        density = rng.normal(size=n)
        s_native = significance.estimate_block_scale(
            density, (ai, ao), max_hop=6, n_samples=35,
            rng=np.random.default_rng(1000 + seed)
        )
        s_python = significance._estimate_block_scale_python(
            density, (ai, ao), max_hop=6, n_samples=35,
            rng=np.random.default_rng(1000 + seed)
        )
        assert s_native == s_python


def _poisson_reference_fixed(M, k, d, seed):
    """Evaluate the fixed-K NumPy Poisson reference transform."""
    rng = np.random.default_rng(seed)
    gamma_arrivals = np.cumsum(rng.exponential(1.0, size=(M, k)), axis=1)
    g = -np.log(np.mean(gamma_arrivals ** (1.0 / float(d)), axis=1))
    return float(np.median(g)), float(np.quantile(g, 0.99)), float(g.std())


def _poisson_reference_variable(M, ks, d, seed):
    """Evaluate the variable-K NumPy Poisson reference transform."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(ks, dtype=np.int32), size=M, replace=True)
    kmax = int(max(ks))
    gaps = np.cumsum(rng.exponential(1.0, size=(M, kmax)), axis=1)
    radial = gaps ** (1.0 / float(d))
    cumulative = np.cumsum(radial, axis=1)
    g = -np.log(np.maximum(cumulative[np.arange(M), chosen - 1] / chosen, 1e-300))
    return float(np.median(g)), float(np.quantile(g, 0.99)), float(g.std())


def test_halo_cython_streaming_preserves_numpy_rng_and_quantiles():
    """Verify halo cython streaming preserves numpy rng and quantiles."""
    assert halo._HAS_HALO_NATIVE
    M, seed = 12000, 1234
    for k, d in ((7, 2), (15, 3), (21, 5)):
        got = halo.poisson_null_quantiles.__wrapped__(k, d, 0.01, M, seed)
        ref = _poisson_reference_fixed(M, k, d, seed)
        np.testing.assert_allclose(got, ref, rtol=2e-13, atol=2e-13)
    ks = (3, 5, 9, 15)
    got = halo.poisson_null_quantiles_variable.__wrapped__(ks, 3, 0.01, M, seed)
    ref = _poisson_reference_variable(M, ks, 3, seed)
    np.testing.assert_allclose(got, ref, rtol=2e-13, atol=2e-13)


def _rescue_cfg():
    """Build the rescue configuration used by the test."""
    from entrap.config import EntrapConfig
    return EntrapConfig.from_params(5, 0.05, {"alpha_gate": 0.99}).resolve(61)


def test_membership_cython_rescue_matches_python_reference(monkeypatch):
    """Verify membership cython rescue matches python reference."""
    assert rescue._HAS_MEMBERSHIP_NATIVE
    rng = np.random.default_rng(13)
    A = rng.normal(0.0, 0.3, size=(30, 3))
    B = rng.normal(3.0, 0.3, size=(30, 3))
    X = np.vstack([A, B, [[0.1, 0.0, 0.0]]])
    labels = np.array([0] * 30 + [1] * 30 + [-1], dtype=np.int64)
    pi_pc = np.zeros((61, 2), dtype=np.float64)
    pi_pc[-1] = [0.8, 0.9]
    basins = {0: np.arange(30), 1: np.arange(30, 60)}
    proms = {0: 1.0, 1: 1.0}
    native = rescue.rescue(X, labels, pi_pc, basins, proms, _rescue_cfg())
    monkeypatch.setattr(rescue, "_HAS_MEMBERSHIP_NATIVE", False)
    reference = rescue.rescue(X, labels, pi_pc, basins, proms, _rescue_cfg())
    np.testing.assert_array_equal(native[0], reference[0])
    assert native[1] == reference[1]


def test_membership_cython_tail_propagation_matches_python_reference(monkeypatch):
    """Verify membership cython tail propagation matches python reference."""
    assert tails._HAS_MEMBERSHIP_NATIVE
    rng = np.random.default_rng(31)
    core = np.column_stack([rng.normal(0.0, 1.3, 60), rng.normal(0.0, 0.08, 60)])
    tail = np.column_stack([np.linspace(2.5, 5.0, 12), rng.normal(0.0, 0.04, 12)])
    other = rng.normal([0.0, 2.5], [1.0, 0.1], size=(20, 2))
    X = np.vstack([core, tail, other])
    labels = np.full(len(X), -1, dtype=np.int64)
    labels[:len(core)] = 0
    basins = {0: np.arange(len(core))}
    kwargs = dict(k_neighbors=5, max_hops=3, radius_factor=3.0,
                  width_factor=3.0, min_anisotropy=1.5)
    native = tails.extend_tidal_tail_candidates(X, labels, basins, (0, 1), **kwargs)
    monkeypatch.setattr(tails, "_HAS_MEMBERSHIP_NATIVE", False)
    reference = tails.extend_tidal_tail_candidates(X, labels, basins, (0, 1), **kwargs)
    np.testing.assert_array_equal(native[0], reference[0])
    np.testing.assert_array_equal(native[1], reference[1])


def test_all_four_native_modules_are_importable():
    """Verify all four native modules are importable."""
    from entrap import _density, _null, _halo, _membership
    assert all(module is not None for module in (_density, _null, _halo, _membership))
