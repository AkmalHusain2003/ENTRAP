import numpy as np
import pytest
from entrap.config import EntrapConfig
from entrap.tails import extend_tidal_tail_candidates


def test_tail_membership_requires_explicit_feature_coordinates():
    """Verify tail membership requires explicit feature coordinates."""
    with pytest.raises(ValueError, match="tail_feature_indices"):
        EntrapConfig.from_params(10, 0.05, {"tail_membership": True})


def test_geometric_tail_extension_is_separate_and_opt_in():
    """Verify geometric tail extension is separate and opt in."""
    rng = np.random.default_rng(8)
    core = np.column_stack([np.linspace(-1, 1, 80), rng.normal(0, 0.035, 80)])
    tail = np.column_stack([np.linspace(1.08, 1.65, 12), rng.normal(0, 0.035, 12)])
    field = rng.uniform(-2.0, 2.0, size=(60, 2))
    X = np.vstack([core, tail, field])
    core_idx = np.arange(len(core))
    labels = np.full(len(X), -1, dtype=np.int64)
    labels[core_idx] = 0
    out, mask = extend_tidal_tail_candidates(
        X, labels, {0: core_idx}, (0, 1), k_neighbors=4,
        max_hops=4, radius_factor=3.0, width_factor=3.0,
        min_anisotropy=1.5)
    assert np.all(out[core_idx] == 0)
    assert np.any(mask[len(core):len(core)+len(tail)])
    assert np.all(out[mask] == 0)
    assert np.all(out[~mask & (labels < 0)] == -1)
