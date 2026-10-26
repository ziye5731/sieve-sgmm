"""Nonlinear-in-parameter scalar calibration used in the methods post."""

import numpy as np
from ssgmm import BaseModel, SSGMM


class CurvedQuadratic(BaseModel):
    def predict(self, theta, x):
        return (theta[0] + 0.01 * theta[0] ** 2) * x**2

    def gradient(self, theta, x):
        return (1 + 0.02 * theta[0]) * x**2


def main():
    rng = np.random.default_rng(20261023)
    estimators = [
        SSGMM(
            CurvedQuadratic(),
            1,
            np.array([0.5]),
            degrees=degrees,
            lr=3.0,
            lr_decay=1.0,
            jacobian_lr_decay=2 / 3,
            lr_offset=100,
            ridge=1.0,
            radius=1.0,
        )
        for degrees in [(1,), (2,)]
    ]
    for z, u, nu in rng.normal(size=(20_000, 3)):
        x = z + u
        # theta*=0; E[epsilon|z]=0 while E[x^2 epsilon]=1.
        y = 0.5 * u + 0.5 * (u * u - 1) + nu
        for estimator in estimators:
            estimator.update(np.array([z]), np.array([x]), y)
    print("True parameter: 0. Degree one misses the population identifying signal.")
    for estimator in estimators:
        print(
            f"Degrees {estimator.degrees}: last={estimator.theta_[0]:.5f}; "
            f"average={estimator.coef_[0]:.5f}"
        )
    print("A single stream is illustrative; use the study runner for repeated-stream MSE.")


if __name__ == "__main__":
    main()
