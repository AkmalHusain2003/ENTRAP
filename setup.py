import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Use the compiler's standard optimized mode without fast-math.  Avoiding
# -ffast-math preserves IEEE floating-point behaviour used by persistence ties.
extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

extensions = [
    Extension(
        f"entrap.{module}",
        [f"entrap/{module}.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    )
    for module in ("_topology", "_density", "_null", "_halo", "_membership")
]

setup(
    name="entrap",
    version="1.1.0.dev1",
    description=(
        "ENTRAP: adaptive topological clustering with modular Cython acceleration for "
        "density, null traversal, halo, and membership kernels"
    ),
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "igraph>=0.11",
        "leidenalg>=0.10",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
