"""Endogenous linear IV example. Run after installing the package."""

import numpy as np
from ssgmm import LinearModel, SSGMM


def main():
    rng = np.random.default_rng(42)
    n = 20_000
    z = rng.normal(size=(n, 1))
    u = rng.normal(size=(n, 1))
    x = z + u
    truth = np.array([0.7])
    y = x @ truth + 0.5 * u[:, 0] + rng.normal(size=n)
    estimator = SSGMM(
        LinearModel(),
        1,
        np.zeros(1),
        degrees=(1,),
        lr=3.0,
        lr_decay=1.0,
        jacobian_lr_decay=2 / 3,
        lr_offset=100,
        ridge=1.0,
        radius=2.0,
    )
    # Blocks can be supplied incrementally without resetting either time scale.
    for start in range(0, n, 1000):
        estimator.fit(z[start : start + 1000], x[start : start + 1000], y[start : start + 1000])
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    print(
        f"Truth: {truth[0]:.4f}; OLS: {ols[0]:.4f}; "
        f"SSGMM last: {estimator.theta_[0]:.4f}; average: {estimator.coef_[0]:.4f}"
    )


if __name__ == "__main__":
    main()
