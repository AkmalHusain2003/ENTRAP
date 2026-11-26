import numpy as np
import pytest
from entrap.diagnostics import persistence_entropy


def test_empty_is_zero():
    """Verify empty is zero."""
    assert persistence_entropy(np.empty((0, 2))) == 0.0


def test_single_bar_is_zero():
    """Verify single bar is zero."""
    assert persistence_entropy(np.array([[1.0, 3.0]])) == pytest.approx(0.0, abs=1e-9)


def test_uniform_lifetimes_is_log_n():
    """Verify uniform lifetimes is log n."""
    n = 5
    pairs = np.column_stack([np.zeros(n), np.ones(n) * 2.0])
    assert persistence_entropy(pairs) == pytest.approx(np.log(n), rel=1e-9)


def test_matches_definition():
    """Verify matches definition."""
    pairs = np.array([[0., 1.], [0., 3.], [0., 6.]])
    ell = pairs[:, 1] - pairs[:, 0]
    p = ell / ell.sum()
    expected = -np.sum(p * np.log(p + 1e-12))
    assert persistence_entropy(pairs) == pytest.approx(expected, rel=1e-12)


def test_zero_lifetimes_ignored():
    """Verify zero lifetimes ignored."""
    pairs = np.array([[1., 1.], [0., 2.]])
    assert persistence_entropy(pairs) == pytest.approx(0.0, abs=1e-9)
