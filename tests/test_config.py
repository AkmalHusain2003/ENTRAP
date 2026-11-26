import math
import numpy as np
import pytest
from entrap.config import EntrapConfig


def test_two_public_parameters_only():
    """Verify two public parameters only."""
    cfg = EntrapConfig.from_params(15, 0.05)
    assert cfg.min_cluster_size == 15 and cfg.significance == 0.05


def test_unknown_advanced_option_rejected():
    """Verify unknown advanced option rejected."""
    with pytest.raises(ValueError):
        EntrapConfig.from_params(10, 0.05, {"not_a_real_option": 1})


@pytest.mark.parametrize("mcs,alpha", [(1, 0.05), (10, 0.0), (10, 1.0), (10, 0.05)])
def test_validation_bounds(mcs, alpha):
    """Verify validation bounds."""
    if mcs < 2 or not (0 < alpha < 1):
        with pytest.raises(ValueError):
            EntrapConfig.from_params(mcs, alpha)


def test_auto_k_formula():
    """Verify auto k formula."""
    for N in (50, 1000, 100_000):
        cfg = EntrapConfig.from_params(10, 0.05).resolve(N)
        expected = min(50, max(15, math.ceil(2 * math.log(max(N, 2)))))
        expected = min(expected, max(2, N - 1))
        assert cfg.k_resolved == expected


def test_pi_threshold_linked_to_alpha():
    """Verify pi threshold linked to alpha."""
    cfg = EntrapConfig.from_params(10, 0.07).resolve(500)
    assert cfg.pi_threshold_resolved == pytest.approx(1 - 0.07)


def test_stability_and_pi_runs_default_to_B():
    """Verify stability and pi runs default to B."""
    cfg = EntrapConfig.from_params(10, 0.05, {"B": 17}).resolve(500)
    assert cfg.stability_B == 17 and cfg.pi_runs == 17


def test_full_fidelity_defaults():
    """Verify full fidelity defaults."""
    cfg = EntrapConfig.from_params(10, 0.05)
    assert cfg.pi_full_stability is True
    assert cfg.B == 50 and cfg.P == 199 and cfg.f == 0.8


def test_to_dict_roundtrip():
    """Verify to dict roundtrip."""
    cfg = EntrapConfig.from_params(12, 0.08, {"B": 5}).resolve(300)
    d = cfg.to_dict()
    assert d["min_cluster_size"] == 12 and d["B"] == 5
