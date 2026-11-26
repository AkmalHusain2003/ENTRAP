import numpy as np
import pytest
from sklearn.base import clone
from entrap import ENTRAP
from helpers import make_blobs

FAST = {"B": 4, "P": 19, "pi_runs": 4, "stability_B": 4,
        "pi_full_stability": False, "seed": 0, "use_leiden": False,
        "multi_k": False}


def test_end_to_end_smoke():
    """Verify end to end smoke."""
    X, truth = make_blobs(n_per=30)
    model = ENTRAP(min_cluster_size=10, significance=0.2, advanced=FAST).fit(X)
    assert model.labels_.shape == (len(X),)
    r = model.result_
    assert r.probabilities.shape == (len(X),)
    assert np.all((r.probabilities >= 0) & (r.probabilities <= 1))
    assert r.n_clusters >= 1
    assert r.execution_time > 0
    s = model.get_summary()
    assert {"n_clusters", "n_noise", "noise_rescued"} <= set(s)


def test_fit_predict_matches_labels():
    """Verify fit predict matches labels."""
    X, _ = make_blobs(n_per=25)
    m = ENTRAP(10, 0.2, advanced=FAST)
    lab = m.fit_predict(X)
    np.testing.assert_array_equal(lab, m.labels_)


def test_reproducible_with_fixed_seed():
    """Verify reproducible with fixed seed."""
    X, _ = make_blobs(n_per=25)
    a = ENTRAP(10, 0.2, advanced=FAST).fit(X).labels_
    b = ENTRAP(10, 0.2, advanced=FAST).fit(X).labels_
    np.testing.assert_array_equal(a, b)


def test_zero_argument_construction():
    """Verify zero argument construction."""
    m = ENTRAP()
    assert m.min_cluster_size == 10 and m.significance == 0.05


def test_sklearn_clone():
    """Verify sklearn clone."""
    m = ENTRAP(12, 0.1, advanced=FAST)
    c = clone(m)
    assert c.min_cluster_size == 12 and c.significance == 0.1


def test_summary_before_fit_raises():
    """Verify summary before fit raises."""
    with pytest.raises(ValueError):
        ENTRAP().get_summary()


def test_invalid_input_rejected():
    """Verify invalid input rejected."""
    with pytest.raises(ValueError):
        ENTRAP(advanced=FAST).fit(np.random.default_rng(0).normal(size=(2, 3)))
    with pytest.raises(ValueError):
        ENTRAP(advanced=FAST).fit(np.full((10, 3), np.nan))
