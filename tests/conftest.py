import numpy as np
import pytest
from entrap.config import EntrapConfig
from helpers import make_blobs, uniform_field


@pytest.fixture
def rng():
    """Create the seeded random generator used by the tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def blobs():
    """Create the shared Gaussian-blob fixture."""
    return make_blobs()


@pytest.fixture
def field():
    """Create the shared uniform-background fixture."""
    return uniform_field()


@pytest.fixture
def fast_cfg():
    """Create a low-cost configuration for unit tests."""
    return EntrapConfig.from_params(
        10, 0.1,
        {"B": 6, "P": 19, "pi_runs": 4, "stability_B": 6,
         "pi_full_stability": False, "seed": 0, "use_leiden": False,
         "multi_k": False}
    ).resolve(120)
