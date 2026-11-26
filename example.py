"""Run ENTRAP on a synthetic three-cluster catalogue."""

import numpy as np

from entrap import ENTRAP, topology


def main():
    """Fit ENTRAP to three Gaussian clusters with a uniform noise field."""
    print("Cython core:", topology.USING_CYTHON)
    rng = np.random.default_rng(0)
    centers = ([0, 0, 0], [6, 6, 0], [0, 7, 5])
    blobs = [rng.normal(center, 0.35, size=(150, 3)) for center in centers]
    noise = np.random.default_rng(0).uniform(-3, 9, size=(120, 3))
    X = np.vstack(blobs + [noise])

    # Use a sufficient null budget for the multi-scale, two-statistic correction.
    advanced = {
        "B": 6,
        "P": 199,
        "pi_runs": 6,
        "stability_B": 6,
        "pi_full_stability": False,
        "seed": 0,
        "verbose": True,
    }
    model = ENTRAP(min_cluster_size=20, significance=0.1, advanced=advanced)
    model.fit(X)

    print(model.get_summary())
    print("Root selected K:", model.result_.tree.selected_k)
    print("Root K family:", model.result_.tree.k_scales)
    print("Root adjusted p:", model.result_.tree.k_adjusted_p)


if __name__ == "__main__":
    main()
