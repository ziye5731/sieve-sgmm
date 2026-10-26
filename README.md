# sieve-sgmm

[![Tests](https://github.com/ziye5731/sieve-sgmm/actions/workflows/tests.yml/badge.svg)](https://github.com/ziye5731/sieve-sgmm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official reference implementation accompanying **[Streaming Sieve GMM for Nonlinear Instrumental Variable Regression](https://ziye5731.github.io/posts/2026/10/sieve-sgmm/)** by Ziye Zhang.

SSGMM estimates a finite-dimensional structural model from an ordinary stream of instrumental-variable observations. A fixed Hermite sieve supplies instrument functions; a running Jacobian supplies a damped Gauss–Newton direction. Each update consumes one observation, uses the **previous** Jacobian state, projects the parameter, and then updates the Jacobian using that same observation. It requires no first-stage regression or conditional two-sample oracle.

This repository contains an installable NumPy implementation, examples, and the identification and convergence-rate experiments accompanying the post. It was extracted from the broader [IV research repository](https://github.com/ziye5731/iv-regression); exploratory competing algorithms and local experiment outputs are not part of this distribution.

## Installation

Python 3.10 or later:

```bash
git clone https://github.com/ziye5731/sieve-sgmm.git
cd sieve-sgmm
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core package depends only on NumPy. Install the optional experiment and development dependencies with:

```bash
python -m pip install -e '.[study,dev]'
```

The distribution is named `sieve-sgmm`; the Python import is `ssgmm`. Installation from this repository is supported; no PyPI release is implied.

## Quick start

```python
import numpy as np
from ssgmm import LinearModel, SSGMM

rng = np.random.default_rng(42)
z = rng.normal(size=(20_000, 1))
u = rng.normal(size=(20_000, 1))
x = z + u
y = 0.7 * x[:, 0] + 0.5 * u[:, 0] + rng.normal(size=len(z))

estimator = SSGMM(
    model=LinearModel(),
    n_instruments=1,
    theta0=np.zeros(1),
    degrees=(1,),
    lr=3.0,
    lr_decay=1.0,
    jacobian_lr_decay=2 / 3,
    lr_offset=100,
    ridge=1.0,
    radius=2.0,
)
estimator.fit(z, x, y)
print(estimator.coef_)  # average of all updated iterates
print(estimator.theta_)  # last iterate
```

For an observation stream, call `estimator.update(z_t, x_t, y_t)` with one-dimensional instrument and regressor vectors and a scalar outcome. Repeated calls to `fit` continue the existing stream. Both interfaces retain fixed-size state and average **all** updated parameters, excluding the initial value.

Run complete examples:

```bash
python examples/linear_iv.py
python examples/nonlinear_iv.py
```

Custom structural models implement `predict(theta, x)` with output shape `(n, 1)` and `gradient(theta, x)` with output shape `(n, d)`. The parameter `theta` has shape `(d,)`. Built-in linear, symmetric quadratic, and exact exponential models use this interface. See [API and model conventions](docs/api.md).

## Algorithm

For residual `g(theta; x) - y`, define the moment and its Jacobian as

```math
m(D,\theta)=\psi(z)\{g(\theta;x)-y\},\qquad
M(D,\theta)=\psi(z)\nabla_\theta g(\theta;x)^\top.
```

Given the old parameter and historical Jacobian, one observation performs

```math
\begin{aligned}
\theta_t &= \Pi_\Theta\!\left[\theta_{t-1}-\alpha_t
(S_{t-1}^\top S_{t-1}+\lambda I)^{-1}S_{t-1}^\top m(D_t,\theta_{t-1})\right],\\
S_t &= (1-\beta_t)S_{t-1}+\beta_t M(D_t,\theta_{t-1}),\\
\bar\theta_T &= T^{-1}\sum_{t=1}^{T}\theta_t.
\end{aligned}
```

The implementation solves the linear system; it never forms a matrix inverse. `S_0=0`, so the first observation updates the Jacobian but leaves the parameter unchanged. The ridge is fixed and damps the gain without penalizing the moment equation. There is no direction clipping, exponential clipping, restart, or burn-in selection.

`degrees` selects complete normalized Hermite total-degree blocks from 0 through 3. With eight instruments, `(1,)`, `(0, 1)`, `(0, 1, 2)`, and `(0, 1, 2, 3)` give 8, 9, 45, and 165 moments. Gaussian orthonormality requires independent standard Gaussian instrument coordinates; the package performs no automatic standardization.

The [method notes](docs/method.md) explain identification, step sizes, projection, and the limits of the convergence claims. Rank, moment, and local-region conditions must be checked for the application. The software does not certify them, provide standard errors, or establish global nonlinear recovery or statistical efficiency.

## Reproduce the experiments

Install `.[study]`, then run the quick execution check:

```bash
python -m unittest discover -s tests -v
python -m experiments.study --smoke --jobs 1
```

The smoke run uses all five experiment groups with 512 observations, two repetitions, and small pilots. It checks execution and output generation; it does not reproduce the post's estimates.

For the reported budgets:

```bash
# Scalar calibration and Jacobian tracking: 512 repetitions, 2^18 observations.
python -m experiments.study --groups calibration tracking \
  --output results/scalar --jobs 4

# Multivariate baseline: 64 repetitions, 2^18 observations.
python -m experiments.study --groups quadratic-linear quadratic-relu exp-tanh \
  --output results/multivariate-2p18 --jobs 4

# Prespecified extended budgets, in separate output directories.
python -m experiments.study --groups quadratic-linear quadratic-relu exp-tanh \
  --steps 524288 --output results/multivariate-2p19 --jobs 4
python -m experiments.study --groups quadratic-linear quadratic-relu exp-tanh \
  --steps 1048576 --output results/multivariate-2p20 --jobs 4
```

Replot existing results:

```bash
python -m experiments.plot results/scalar
```

Runs save source hashes, configuration, seed lineage, environment information, repeat-level trajectories, diagnostics, tables, and PNG/SVG figures. Completed repetitions resume under the same source and protocol; changed budgets or sources require a new output directory. Large results are generated locally and ignored by Git. Full runs are substantial Monte Carlo computations, particularly the extended budgets; use the smoke run to check the environment first.

Release preparation checks are recorded in [the validation report](docs/validation.md).

See [the full reproducibility protocol](docs/reproducibility.md), including seeds, model settings, output definitions, and numerical equivalence checks. Historical raw result files and a locked historical environment are not bundled; exact cross-version floating-point identity is not promised.

## Repository layout

```text
src/ssgmm/       Public estimator, Hermite features, structural models
experiments/    Fixed study configuration, compiled kernels, runner, plotting
examples/       Small linear and nonlinear IV examples
tests/          Recursion equivalence, exact moments, API and workflow tests
docs/           Method, API, reproducibility, and source provenance
```

## Citation and license

Use the repository's [CITATION.cff](CITATION.cff) for the software citation and link to the [accompanying methods post](https://ziye5731.github.io/posts/2026/10/sieve-sgmm/). No paper DOI is currently specified. The original research implementation is credited in [source provenance](docs/provenance.md).

Released under the [MIT license](LICENSE). Contributions and bug reports are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
