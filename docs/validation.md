# Release preparation validation

Validated on October 26, 2026 with Python 3.12 on macOS arm64. Exact dependency versions and machine-readable check results are recorded in [validation.json](validation.json).

- All 12 unit tests passed, including independent Hermite values, model finite differences, exact Gaussian moments, sequential cached-kernel equivalence, historical-gain ordering, stream continuation, projection and failure handling.
- Six cases were compared directly with the untouched upstream research estimator, each for 513 observations. Every parameter iterate, full-history average and Jacobian state matched exactly in this environment (maximum absolute difference 0). The cases include the scalar calibration with degree one/two, quadratic models with three basis choices, and the exact exponential model.
- All five smoke groups completed: 45 configurations, two repetitions each, 512 observations per configuration/repetition, with declared small pilots. There were no numerical failures. The runner generated all 12 expected PNG/SVG figure pairs, and both safe resume and independent replot completed.
- The source distribution and wheel built successfully. In a fresh environment containing only the wheel and NumPy, all six core tests and both examples ran successfully; Numba and Matplotlib were absent.
- Static error checks, formatting checks and relative documentation-link checks passed.

These checks establish runnable packaging and implementation consistency. They do not replace the post's statistical assumptions. The full 512/64-repeat experiments at the post's budgets were not rerun during repository preparation. Historical large result archives are not bundled. CI additionally defines Linux jobs for Python 3.10, 3.12 and 3.13; its status is visible on GitHub.
