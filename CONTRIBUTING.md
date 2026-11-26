# Contributing to ENTRAP

Thank you for considering a contribution to ENTRAP. The project is research
software, so contributions must preserve traceability between mathematical
methods, implementation, and scientific tests.

## Development setup

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-test.txt
python -m pip install -e .
```

Build the Cython extensions in place when working from a source checkout:

```bash
python setup.py build_ext --inplace
```

Run the complete test suite before submitting a change:

```bash
python -m pytest -ra
```

## Code style

- Use one statement per line; do not combine statements with semicolons.
- Keep function and class docstrings to one concise sentence.
- Add a literature citation to a docstring only when the implementation
  directly follows a published method; distinguish published methods from
  ENTRAP-specific adaptations.
- Use descriptive variable names where they improve mathematical clarity.
- Keep Python modules responsible for validation, orchestration, and public
  interfaces.
- Keep Cython kernels focused on measured numerical or memory bottlenecks.
- Preserve readable Python reference implementations when a native kernel
  replaces an existing numerical path.
- Avoid compiler flags such as `-ffast-math` that can change floating-point
  comparison behaviour in persistence calculations.

## Mathematical and numerical changes

A performance-only change must not alter the mathematical definition of the
algorithm. For every optimized kernel:

1. Compare the optimized implementation with its reference implementation.
2. Test empty inputs, small inputs, duplicate points, ties, and boundary cases.
3. Check floating-point outputs with a justified tolerance.
4. Check discrete decisions exactly, including selected K, mode counts,
   component assignments, core masks, rescued labels, and tail-candidate masks.
5. Test reproducibility with fixed random seeds.
6. Benchmark runtime and peak memory on the same data and configuration.

A change to a formula, null model, threshold, graph-scale family, Leiden
objective, or membership rule is an algorithmic change. It must be documented
separately and supported by mathematical reasoning and dedicated tests.

## References and attribution

Cite the original publication in a function docstring when a function directly
implements a published method. Do not attribute ENTRAP-specific combinations,
heuristics, or parameter-selection rules to an external paper. Add complete
bibliographic details to the References section of `README.md` when a new
method dependency is introduced.

## Pull requests

A pull request should describe its motivation, affected modules, whether it is
an algorithmic or implementation-only change, test results, benchmark setup,
and any known limitations. Do not report a speed-up without stating the input
size, hardware/software environment, repetitions, and reference path.
