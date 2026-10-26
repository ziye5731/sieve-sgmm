# Contributing

Please open an issue with a minimal example, dependency versions, the chosen model and degrees, and the expected versus observed behavior. Do not include private data or credentials.

For development:

```bash
python -m pip install -e '.[study,dev]'
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
python -m experiments.study --smoke --jobs 1 --output results/contribution-check
```

Changes to features, parameterization, update ordering, step sizes, projection, averaging or numerical safeguards must state their statistical implications. Preserve the single-observation historical-Jacobian recursion. Add a meaningful independent check when changing numerical behavior; retain failed repetitions in experiment aggregation. Use a new result directory after changing source or protocol.

The core package must remain usable with NumPy alone. Import optional experiment dependencies only from the experiment modules.
