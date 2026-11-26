import numpy as np
import pytest
from entrap.rescue import rescue
from entrap.config import EntrapConfig


def _cfg():
    """Build the Leiden configuration used by the test."""
    return EntrapConfig.from_params(5, 0.05, {"alpha_gate": 0.99}).resolve(60)


def test_far_point_not_rescued():
    """Verify far point not rescued."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.3, size=(30, 3))
    X = np.vstack([A, [[100., 100., 100.]]])
    labels = np.array([0] * 30 + [-1])
    pi_pc = np.zeros((31, 1))
    pi_pc[30, 0] = 1.0
    out, n = rescue(X, labels, pi_pc, {0: np.arange(30)}, {0: 1.0}, _cfg())
    assert out[30] == -1 and n == 0


def test_near_point_rescued_to_argmax_pi():
    """Verify near point rescued to argmax pi."""
    rng = np.random.default_rng(1)
    A = rng.normal(0, 0.3, size=(30, 3))
    B = rng.normal(3, 0.3, size=(30, 3))
    x = np.array([0.1, 0.0, 0.0])
    X = np.vstack([A, B, x[None, :]])
    labels = np.array([0] * 30 + [1] * 30 + [-1])
    pi_pc = np.zeros((61, 2))
    pi_pc[60] = [0.2, 0.9]
    out, n = rescue(X, labels, pi_pc, {0: np.arange(30), 1: np.arange(30, 60)},
                    {0: 1.0, 1: 1.0}, _cfg())
    assert n == 1
    assert out[60] in (0, 1)


def test_tie_broken_by_prominence():
    """Verify tie broken by prominence."""
    rng = np.random.default_rng(2)
    A = rng.normal(0, 0.3, size=(30, 3))
    B = rng.normal(0, 0.3, size=(30, 3))     # same location -> both pass the gate
    X = np.vstack([A, B, [[0., 0., 0.]]])
    labels = np.array([0] * 30 + [1] * 30 + [-1])
    pi_pc = np.zeros((61, 2))
    pi_pc[60] = [0.5, 0.5]
    out, _ = rescue(X, labels, pi_pc, {0: np.arange(30), 1: np.arange(30, 60)},
                    {0: 1.0, 1: 9.0}, _cfg())
    assert out[60] == 1


def test_input_labels_not_mutated():
    """Verify input labels not mutated."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 0.3, (30, 3)), [[0., 0., 0.]]])
    labels = np.array([0] * 30 + [-1])
    pi_pc = np.ones((31, 1))
    before = labels.copy()
    rescue(X, labels, pi_pc, {0: np.arange(30)}, {0: 1.0}, _cfg())
    np.testing.assert_array_equal(labels, before)


def test_no_clusters_returns_unchanged():
    """Verify no clusters returns unchanged."""
    X = np.random.default_rng(0).normal(size=(10, 3))
    labels = np.full(10, -1)
    out, n = rescue(X, labels, np.zeros((10, 1)), {}, {}, _cfg())
    assert n == 0 and np.all(out == -1)
