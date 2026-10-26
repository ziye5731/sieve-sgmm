"""Public API checks and independent deterministic recursion checks (NumPy only)."""

import unittest
import numpy as np
from ssgmm import (
    SSGMM,
    LinearModel,
    QuadraticModel,
    ExponentialModel,
    hermite_features,
    feature_dimension,
)


class EstimatorTests(unittest.TestCase):
    def make(self, **kwargs):
        return SSGMM(LinearModel(), 1, np.array([0.5]), degrees=(1,), lr=0.3, **kwargs)

    def test_historical_gain_and_same_old_parameter(self):
        estimator = self.make()
        estimator.update([2.0], [3.0], 1.0)
        beta1 = 1 / 11**0.6
        self.assertEqual(estimator.theta_[0], 0.5)  # zero initial gain
        self.assertAlmostEqual(estimator.jacobian_[0, 0], beta1 * 6)
        old = beta1 * 6
        residual = 0.5 * 5 - 2
        expected = 0.5 - (0.3 / 12**0.8) * old * (4 * residual) / (old**2 + 0.01)
        estimator.update([4.0], [5.0], 2.0)
        self.assertAlmostEqual(estimator.theta_[0], expected)
        self.assertAlmostEqual(estimator.average_[0], (0.5 + expected) / 2)
        beta2 = 1 / 12**0.6
        self.assertAlmostEqual(estimator.jacobian_[0, 0], (1 - beta2) * old + beta2 * 20)

    def test_split_stream_matches_single_fit(self):
        rng = np.random.default_rng(212)
        z, x = rng.normal(size=(50, 1)), rng.normal(size=(50, 1))
        y = rng.normal(size=50)
        one, blocks = self.make(), self.make()
        one.fit(z, x, y)
        blocks.fit(z[:17], x[:17], y[:17, None]).fit(z[17:], x[17:], y[17:])
        for name in ["theta_", "average_", "jacobian_"]:
            np.testing.assert_array_equal(getattr(one, name), getattr(blocks, name))
        self.assertEqual(blocks.n_samples_seen_, 50)
        np.testing.assert_allclose(one.predict(x), x @ one.coef_[:, None])
        np.testing.assert_allclose(one.predict(x, averaged=False), x @ one.theta_[:, None])

    def test_projection_and_state_copies(self):
        estimator = SSGMM(LinearModel(), 1, np.array([50.0]), center=[1.0], radius=0.1)
        self.assertAlmostEqual(estimator.theta_[0], 1.1)
        for _ in range(20):
            estimator.update([1.0], [1.0], -1000.0)
            self.assertLessEqual(abs(estimator.theta_[0] - 1), 0.10000000001)
        state = estimator.theta_
        state[:] = 999
        self.assertLess(estimator.theta_[0], 2)

    def test_validation_and_nonfinite_update_are_atomic(self):
        for kwargs in [
            dict(lr=0),
            dict(ridge=0),
            dict(radius=np.inf),
            dict(lr_decay=0.6),
            dict(jacobian_lr_decay=0.5),
            dict(jacobian_lr=100),
            dict(lr_offset=-1),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SSGMM(LinearModel(), 1, np.zeros(1), **kwargs)
        estimator = self.make()
        for z, x, y in [([1, 2], [1], 1), ([1], [np.nan], 1), ([1], [1], [1])]:
            with self.assertRaises(ValueError):
                estimator.update(z, x, y)
            self.assertEqual(estimator.n_samples_seen_, 0)
        exp = SSGMM(ExponentialModel(), 1, np.array([1.0]), radius=2)
        with self.assertRaises(FloatingPointError):
            exp.update([1.0], [1000.0], 0.0)
        np.testing.assert_array_equal(exp.theta_, [1])
        np.testing.assert_array_equal(exp.jacobian_, np.zeros((3, 1)))
        self.assertEqual(exp.n_samples_seen_, 0)
        with self.assertRaises(ValueError):
            estimator.fit(np.zeros((3, 1)), np.zeros((2, 1)), np.zeros(3))

    def test_models_by_finite_differences(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(5, 3))
        for model, dimension in [
            (LinearModel(), 3),
            (QuadraticModel(), 9),
            (ExponentialModel(), 3),
        ]:
            theta = rng.normal(scale=0.1, size=dimension)
            numerical = np.empty((len(x), dimension))
            for i in range(dimension):
                step = np.zeros(dimension)
                step[i] = 1e-6
                numerical[:, i] = (
                    (model.predict(theta + step, x) - model.predict(theta - step, x)) / 2e-6
                ).ravel()
            np.testing.assert_allclose(model.gradient(theta, x), numerical, atol=1e-8, rtol=1e-7)

    def test_dimensions_and_independent_hermite_values(self):
        self.assertEqual(
            [feature_dimension(8, k) for k in [(1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]],
            [8, 9, 45, 165],
        )
        z = np.array([[2.0, 3.0]])
        expected = [
            1,
            2,
            3,
            3 / np.sqrt(2),
            8 / np.sqrt(2),
            6,
            2 / np.sqrt(6),
            18 / np.sqrt(6),
            9 / np.sqrt(2),
            16 / np.sqrt(2),
        ]
        np.testing.assert_allclose(hermite_features(z, (0, 1, 2, 3))[0], expected)
        for bad in [(), (4,), (1.5,), (True,), "12"]:
            with self.assertRaises(ValueError):
                hermite_features(z, bad)
        with self.assertRaises(ValueError):
            feature_dimension(0)


if __name__ == "__main__":
    unittest.main()
