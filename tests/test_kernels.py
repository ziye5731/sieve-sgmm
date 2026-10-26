"""Scientific integrity checks: independent reference algorithm and exact moments."""

import unittest
import numpy as np
from ssgmm import SSGMM
from ssgmm.basis import hermite_features as _hermite_degrees_features
from ssgmm.models import BaseModel, QuadraticModel
from experiments.kernels import (
    basis_code,
    feature_dim,
    hermite_one,
    train_path,
    quadratic_population_g,
    multivariate_data,
    chol_solve,
)


class ScalarModel(BaseModel):
    def predict(self, theta, x):
        return (theta.ravel()[0] + 0.01 * theta.ravel()[0] ** 2) * x * x

    def gradient(self, theta, x):
        return (1 + 0.02 * theta.ravel()[0]) * x * x

    def param_dim(self, d_x):
        return 1

    def init_params(self, rng, d_x):
        return np.array([[0.5]])

    def true_params(self, rng, d_x):
        return np.zeros((1, 1))


class PureExpModel(ScalarModel):
    def predict(self, theta, x):
        return np.exp(x @ theta.reshape(-1, 1))

    def gradient(self, theta, x):
        return self.predict(theta, x) * x


class StudyTests(unittest.TestCase):
    def test_features_and_positive_definite_solve(self):
        rng = np.random.default_rng(7191)
        z = rng.normal(size=8)
        for basis in [(1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]:
            code = basis_code(basis)
            out = np.empty(feature_dim(8, code))
            hermite_one(z, code, out)
            np.testing.assert_allclose(
                out, _hermite_degrees_features(z[None], basis)[0], atol=1e-14
            )
        s = rng.normal(size=(165, 14))
        rhs = rng.normal(size=14)
        np.testing.assert_allclose(
            chol_solve(s.T @ s, rhs, 0.01),
            np.linalg.solve(s.T @ s + 0.01 * np.eye(14), rhs),
            rtol=1e-12,
        )

    def compare_reference(self, model_code, basis):
        rng = np.random.default_rng(75319)
        n = 513
        if model_code == 0:
            gamma = np.ones((1, 1))
            truth = np.zeros(1)
            initial = np.array([0.5])
            draws = rng.normal(size=(n, 3))
            z = draws[:, :1]
            x = z + draws[:, 1:2]
            u = draws[:, 1:2]
            y = 0.5 * u + 0.5 * (u * u - 1) + draws[:, 2:3]
            model = ScalarModel()
            ridge = 1.0
            radius = 1.0
            a = 3.0
        else:
            gamma = rng.normal(size=(8, 4)) * 0.2
            draws = rng.normal(size=(n, 17))
            truth = np.zeros(14 if model_code == 1 else 4)
            initial = rng.normal(0, 0.1, len(truth))
            x, y, _ = multivariate_data(draws, gamma, 0, model_code, truth)
            y = y[:, None]
            z = draws[:, :8]
            model = QuadraticModel() if model_code == 1 else PureExpModel()
            ridge = 0.01
            radius = 20.0
            a = 0.1
        d = len(truth)
        algo = SSGMM(
            model=model,
            n_instruments=len(gamma),
            theta0=initial,
            degrees=basis,
            lr=a,
            lr_decay=1.0,
            jacobian_lr_decay=2 / 3,
            lr_offset=100.0,
            ridge=ridge,
            radius=radius,
        )
        reference = []
        averages = []
        for i in range(n):
            algo.update(z[i], x[i], y[i, 0])
            reference.append(algo.theta_.copy())
            averages.append(algo.average_.copy())
        code = basis_code(basis)
        popg = np.zeros((feature_dim(len(gamma), code), d))
        raw, avg, diag, state, failure = train_path(
            draws,
            gamma,
            truth,
            initial,
            np.zeros(d),
            radius,
            model_code,
            0,
            code,
            a,
            1.0,
            2 / 3,
            100.0,
            ridge,
            0.01,
            False,
            np.arange(1, n + 1),
            popg,
        )
        self.assertEqual(failure, 0)
        np.testing.assert_allclose(raw, reference, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(avg, averages, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(state, algo.jacobian_, atol=2e-10, rtol=2e-9)

    def test_same_stream_as_reference_estimator(self):
        for model, basis in [
            (0, (1,)),
            (0, (2,)),
            (1, (1,)),
            (1, (0, 1, 2)),
            (1, (0, 1, 2, 3)),
            (2, (0, 1, 2)),
        ]:
            with self.subTest(model=model, basis=basis):
                self.compare_reference(model, basis)

    def test_exact_rank_and_basis_order(self):
        gamma = np.random.default_rng(10).normal(size=(8, 4))
        for basis, rank in [((1,), 4), ((0, 1), 5), ((0, 1, 2), 14), ((0, 1, 2, 3), 14)]:
            self.assertEqual(np.linalg.matrix_rank(quadratic_population_g(gamma, basis)), rank)
        full = quadratic_population_g(gamma, (0, 1, 2, 3))
        np.testing.assert_array_equal(full[45:], 0)
        # Degree-two expected moments of a quadratic verified by independent quadrature.
        from numpy.polynomial.hermite_e import hermegauss

        nodes, weights = hermegauss(5)
        g = np.array([[0.7], [0.4]])
        z = np.array([(a, b) for a in nodes for b in nodes])
        w = np.outer(weights, weights).ravel() / (2 * np.pi)
        h = z @ g
        q = np.c_[h, h * h + 1.0]
        numerical = _hermite_degrees_features(z, (0, 1, 2, 3)).T @ (q * w[:, None])
        np.testing.assert_allclose(quadratic_population_g(g, (0, 1, 2, 3)), numerical, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
