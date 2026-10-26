# Python API

```python
from ssgmm import (
    SSGMM,
    BaseModel,
    LinearModel,
    QuadraticModel,
    ExponentialModel,
    hermite_features,
    feature_dimension,
)
```

## Estimator

Construct `SSGMM(model, n_instruments, theta0, ...)`. `theta0` is a finite one-dimensional parameter vector. Keyword arguments and defaults:

- `degrees=(0, 1, 2)`: total-degree blocks, from 0–3. An integer selects one block; duplicates are removed and degrees sorted.
- `lr=0.05`, `lr_decay=0.8`: parameter step coefficient and exponent.
- `jacobian_lr=1.0`, `jacobian_lr_decay=0.6`: Jacobian step coefficient and exponent.
- `lr_offset=10.0`: common nonnegative offset, with the first observation indexed by 1.
- `ridge=0.01`: fixed positive gain ridge.
- `radius=20.0`, `center=None`: Euclidean projection ball; `None` means zero center.

`update(z, x, y)` consumes one observation: vectors `z` and `x`, scalar `y`. It returns the estimator. Nonfinite inputs, dimension changes, and malformed model outputs raise errors. A failed individual update does not change the estimator state. Earlier successful observations in a `fit` call remain committed if a later observation fails.

`fit(z, x, y)` processes array rows once in their given order. Shapes are `(n, n_instruments)`, `(n, d_x)`, and `(n,)` or `(n, 1)`. It returns the estimator and **continues**, rather than resets, the existing stream. Construct a new estimator to start over. The estimator stores neither observations nor trajectories.

Read-only state accessors return copies:

- `theta_`: last parameter, shape `(d,)`.
- `average_` and `coef_`: arithmetic average of all updated parameters, shape `(d,)`; before any observations they equal the projected initialization.
- `jacobian_`: current historical moment Jacobian, shape `(p, d)`.
- `n_samples_seen_`: number of committed observations.

`predict(x, averaged=True)` evaluates the structural function using the average; `averaged=False` uses the last parameter. Input shape is `(n, d_x)` and output shape is `(n, 1)`. This is a structural prediction, not an estimate of the observational conditional mean under endogeneity.

## Models

A custom object can implement the two required methods directly, or subclass `BaseModel`:

```python
import numpy as np
from ssgmm import BaseModel


class CustomExponential(BaseModel):
    def predict(self, theta, x):
        return np.exp(x @ theta.reshape(-1, 1))

    def gradient(self, theta, x):
        return self.predict(theta, x) * x
```

`theta` is `(d,)`, `x` is `(n, d_x)`. Predictions must be `(n, 1)`, and gradients `(n, d)`. Both quantities must correspond to the same structural function and parameterization.

`LinearModel` implements `x @ theta`. `ExponentialModel` implements the exact exponential of that index. Neither adds an intercept; include it explicitly in the regressor if required.

`QuadraticModel` implements `x @ b + x.T @ A @ x` for symmetric A. Pack theta as the `d_x` entries of b followed by the upper triangle of A in row-major order. For two regressors the order is `[b0, b1, A00, A01, A11]`. The off-diagonal gradient is **2*x0*x1** because A is symmetric. `param_dim(d_x)` returns `d_x + d_x*(d_x+1)//2`.

The linear and quadratic model classes retain research convenience methods `init_params`, `true_params`, and `param_dim`. The estimator does not call these methods or require known data-generating parameters.

## Features

`hermite_features(z, degrees)` maps a two-dimensional instrument array to `(n, p)`. `feature_dimension(n_instruments, degrees)` gives p without constructing data. Neither function estimates a distribution or standardizes instruments. See [method notes](method.md) for normalization assumptions and ordering.
