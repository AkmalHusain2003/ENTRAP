"""Regression tests for strict igraph/leidenalg integration in V7.1."""
import numpy as np
import pytest

import entrap.leiden as leiden
from entrap.config import EntrapConfig


class _FakeGraph:
    """Provide a minimal graph object for isolated Leiden integration tests."""
    instances = []

    def __init__(self, n, edges, directed=False):
        """Initialize the memory manager and its temporary-storage settings."""
        self._n = int(n)
        self.edges = list(edges)
        self.directed = bool(directed)
        self.vs = list(range(n))
        self.es = {}
        type(self).instances.append(self)

    def vcount(self):
        """Return the number of vertices in the fake graph."""
        return self._n


class _FakeIGraph:
    """Provide a deterministic stand-in for the igraph module."""
    Graph = _FakeGraph
    __version__ = "fake-igraph"


class _FakePartition:
    """Represent a deterministic Leiden partition and its quality score."""

    def __init__(self, membership, quality):
        """Initialize the memory manager and its temporary-storage settings."""
        self.membership = list(map(int, membership))
        self._quality = float(quality)

    def quality(self):
        """Return the stored quality of the fake partition."""
        return self._quality


class _FakeLeidenAlg:
    """Provide a deterministic stand-in for the leidenalg optimizer."""
    CPMVertexPartition = object
    __version__ = "fake-leidenalg"
    calls = []
    strategy = "seed"

    @classmethod
    def find_partition(cls, graph, partition_type, initial_membership=None,
                       weights=None, n_iterations=2, seed=None,
                       resolution_parameter=1.0):
        """Record and return a deterministic fake Leiden partition."""
        cls.calls.append({
            "edges": graph.edges,
            "weights": weights,
            "initial_membership": initial_membership,
            "n_iterations": n_iterations,
            "seed": seed,
            "gamma": float(resolution_parameter),
        })
        n = graph.vcount()
        if cls.strategy == "singletons":
            membership = np.arange(n)
        elif cls.strategy == "two_groups":
            membership = np.r_[np.zeros(n // 2, dtype=int),
                               np.ones(n - n // 2, dtype=int)]
        else:
            membership = np.asarray(initial_membership, dtype=int)
        # Highest fake quality is attained near gamma=0.1 for a deterministic
        # tie-break test, the implementation must choose only among exact-count
        # partitions and use the seed-density neighbourhood in its selection.
        quality = 1.0 - abs(float(resolution_parameter) - 0.1)
        return _FakePartition(membership, quality)


def _cfg(**advanced):
    """Build the Leiden configuration used by the test."""
    args = {
        "use_leiden": True,
        "allow_fallback": False,
        "leiden_gamma_steps": 7,
        "leiden_restarts": 1,
        "leiden_iterations": -1,
    }
    args.update(advanced)
    return EntrapConfig.from_params(4, 0.05, args).resolve(20)


def _install_fake(monkeypatch, strategy="seed"):
    """Install deterministic stand-ins for the external graph backend."""
    _FakeGraph.instances.clear()
    _FakeLeidenAlg.calls.clear()
    _FakeLeidenAlg.strategy = strategy
    monkeypatch.setattr(leiden, "ig", _FakeIGraph)
    monkeypatch.setattr(leiden, "la", _FakeLeidenAlg)
    monkeypatch.setattr(leiden, "_HAS", True)
    monkeypatch.setattr(leiden, "_IMPORT_ERROR", None)


def test_default_configuration_does_not_allow_silent_leiden_fallback():
    """Verify default configuration does not allow silent leiden fallback."""
    cfg = EntrapConfig.from_params(10, 0.05).resolve(100)
    assert cfg.use_leiden is True
    assert cfg.allow_fallback is False


def test_missing_backend_raises_when_leiden_is_required(monkeypatch):
    """Verify missing backend raises when leiden is required."""
    monkeypatch.setattr(leiden, "_HAS", False)
    monkeypatch.setattr(leiden, "_IMPORT_ERROR", ModuleNotFoundError("fake missing"))
    cfg = _cfg()
    seed = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ImportError, match="igraph.*leidenalg"):
        leiden.leiden_partition(
            4, np.array([[0, 1], [2, 3]]), None, 2, seed,
            (0.0, 1.0), cfg, np.random.default_rng(0),
        )


def test_missing_backend_fallback_requires_explicit_opt_in(monkeypatch):
    """Verify missing backend fallback requires explicit opt in."""
    monkeypatch.setattr(leiden, "_HAS", False)
    monkeypatch.setattr(leiden, "_IMPORT_ERROR", None)
    cfg = _cfg(allow_fallback=True)
    seed = np.array([4, 4, 8, 8], dtype=np.int64)
    output, info = leiden.leiden_partition(
        4, np.array([[0, 1], [2, 3]]), None, 2, seed,
        (0.0, 1.0), cfg, np.random.default_rng(0), return_info=True,
    )
    np.testing.assert_array_equal(output, np.array([0, 0, 1, 1]))
    assert info["used_leiden"] is False
    assert info["status"] == "explicit_seed_fallback_backend_unavailable"


def test_graph_is_canonical_simple_undirected_and_weight_aligned(monkeypatch):
    """Verify graph is canonical simple undirected and weight aligned."""
    _install_fake(monkeypatch)
    cfg = _cfg(leiden_gamma_steps=5)
    seed = np.array([0, 0, 1, 1], dtype=np.int64)
    edges = np.array([[1, 0], [0, 1], [1, 1], [3, 2]], dtype=np.int64)
    weights = np.array([0.2, 0.7, 9.0, 0.5])
    _, info = leiden.leiden_partition(
        4, edges, weights, 2, seed, (0.0, 1.0), cfg,
        np.random.default_rng(0), return_info=True,
    )
    graph = _FakeGraph.instances[0]
    assert graph.directed is False
    assert graph.edges == [(0, 1), (2, 3)]
    # Duplicate edge weights collapse to max(0.2, 0.7), self-loop is discarded.
    assert _FakeLeidenAlg.calls[0]["weights"] == [0.7, 0.5]
    assert info["n_edges"] == 2
    assert info["attempted_leiden"] is True
    assert info["used_leiden"] is True


def test_resolution_scan_is_log_spaced_seeded_and_exact_target(monkeypatch):
    """Verify resolution scan is log spaced seeded and exact target."""
    _install_fake(monkeypatch)
    cfg = _cfg(leiden_gamma_steps=9, leiden_iterations=-1)
    seed = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]])
    result, info = leiden.leiden_partition(
        6, edges, np.ones(len(edges)), 3, seed, (0.0, 1.0), cfg,
        np.random.default_rng(12), return_info=True,
    )
    assert len(_FakeLeidenAlg.calls) >= 9
    assert info["status"] == "optimized_exact_target"
    assert info["communities"] == 3
    assert info["used_leiden"] is True
    assert all(call["n_iterations"] == -1 for call in _FakeLeidenAlg.calls)
    assert all(call["seed"] is not None for call in _FakeLeidenAlg.calls)
    assert result.shape == (6,)
    assert len(np.unique(result)) == 3


def test_unreachable_target_keeps_topological_seed(monkeypatch):
    """Verify unreachable target keeps topological seed."""
    _install_fake(monkeypatch, strategy="singletons")
    cfg = _cfg(leiden_gamma_steps=5)
    seed = np.array([0, 0, 1, 1], dtype=np.int64)
    output, info = leiden.leiden_partition(
        4, np.array([[0, 1], [1, 2], [2, 3]]), None, 2, seed,
        (0.0, 1.0), cfg, np.random.default_rng(0), return_info=True,
    )
    np.testing.assert_array_equal(output, seed)
    assert info["status"] == "seed_fallback_no_exact_target_partition"
    assert info["attempted_leiden"] is True
    assert info["used_leiden"] is False
    assert info["communities"] == 2


def test_bad_seed_and_weights_are_rejected(monkeypatch):
    """Verify bad seed and weights are rejected."""
    _install_fake(monkeypatch)
    cfg = _cfg()
    with pytest.raises(ValueError, match="membership"):
        leiden.leiden_partition(
            4, np.array([[0, 1]]), None, 2, np.array([0, 1]),
            (0.0, 1.0), cfg, np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        leiden.leiden_partition(
            4, np.array([[0, 1]]), np.array([0.0]), 2,
            np.array([0, 0, 1, 1]), (0.0, 1.0), cfg,
            np.random.default_rng(0),
        )


def test_demo_requires_real_backend_and_adequate_null_resolution():
    """Verify demo requires real backend and adequate null resolution."""
    import demo_clustering
    assert demo_clustering.ADV["use_leiden"] is True
    assert demo_clustering.ADV["allow_fallback"] is False
    assert demo_clustering.ADV["P"] >= 199
    for _, X, _, mcs, _ in demo_clustering.make_datasets():
        cfg = EntrapConfig.from_params(mcs, 0.05, demo_clustering.ADV).resolve(len(X))
        floor = 2.0 * len(cfg.k_scales_resolved) / (cfg.P + 1.0)
        assert floor <= 0.05


def test_demo_fails_fast_when_real_backend_is_missing(monkeypatch, capsys):
    """Verify demo fails fast when real backend is missing."""
    import demo_clustering
    monkeypatch.setattr(demo_clustering, "has_leiden", lambda: False)
    monkeypatch.setattr(
        demo_clustering,
        "backend_info",
        lambda: {"available": False, "error": "test missing backend"},
    )
    with pytest.raises(SystemExit) as exc:
        demo_clustering.main()
    output = capsys.readouterr().out
    assert "Backend Cython aktif" in output
    assert "python -m pip install -r requirements.txt" in str(exc.value)
    assert "tidak mengizinkan fallback" in str(exc.value)
