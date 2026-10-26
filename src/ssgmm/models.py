"""Structural models g(theta; x) and their per-observation gradients.

Inputs are theta of shape (d,) and x of shape (n, d_x). Predictions have
shape (n, 1); gradients have shape (n, d). Custom models need only implement
predict and gradient, without any first-stage fit.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Interface for a differentiable structural model."""

    @abstractmethod
    def predict(self, theta, x):
        """Return g(theta; x), shape (n, 1)."""

    @abstractmethod
    def gradient(self, theta, x):
        """Return gradients with respect to theta, shape (n, d)."""


class LinearModel(BaseModel):
    """Linear structural equation: g(theta; x) = theta^T x."""

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta_2d = np.atleast_2d(theta.reshape(-1, 1))
        return x @ theta_2d

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        _ = theta
        return x

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        return rng.normal(0, 0.1, size=(d_x, 1))

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        return rng.normal(0, 1.0, size=(d_x, 1))

    def param_dim(self, d_x: int) -> int:
        return d_x


linear_model = LinearModel()


class QuadraticModel(BaseModel):
    """Quadratic structural model:

    Parameterization (theta flattened):
      - first d_x entries: linear coefficient b (shape d_x)
      - remaining entries: upper-triangular entries of symmetric matrix A

    Prediction: g(theta; x) = x @ b + x^T A x  (returns shape (n,1)).
    """

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta.reshape(-1, 1)).ravel()
        d_x = x.shape[1]
        b = theta[:d_x].reshape(-1, 1)  # (d_x, 1)
        A = self._vec_to_mat(theta[d_x:], d_x)
        linear = x @ b  # (n,1)
        quad = np.sum((x @ A) * x, axis=-1, keepdims=True)  # (n,1)
        return linear + quad

    def gradient(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        d_x = x.shape[1]
        # gradient w.r.t. b is x (n, d_x)
        grad_b = x  # (n, d_x)
        # gradient w.r.t packed upper-triangular A entries
        # build per-sample outer products x_i x_j
        grad_full = np.einsum("ni,nj->nij", x, x)  # (n, d_x, d_x)
        grad_packed_list = []
        for i in range(d_x):
            for j in range(i, d_x):
                if i == j:
                    grad_packed_list.append(grad_full[:, i, j : j + 1])
                else:
                    # off-diagonal contributes twice due to symmetry
                    grad_packed_list.append(2.0 * grad_full[:, i, j : j + 1])
        grad_packed = np.concatenate(grad_packed_list, axis=-1)  # (n, n_packed)
        return np.concatenate([grad_b, grad_packed], axis=-1)

    def init_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        # initialize linear coeffs small, and diagonal of A ~ 1/d_x scale
        b = rng.normal(0, 0.1, size=(d_x, 1))
        # upper-triangular packed entries
        n_packed = d_x * (d_x + 1) // 2
        A_packed = np.zeros((n_packed, 1))
        return np.vstack([b, A_packed])

    def true_params(self, rng: np.random.Generator, d_x: int) -> np.ndarray:
        # Construct a true parameter that corresponds roughly to
        # g(x) = (x + s)^T (x + s) = x^T x + 2 s^T x + const
        # We encode A = I and b = 2*s.
        s = rng.normal(0, 1.0, size=(d_x, 1))
        b = 2.0 * s  # linear term
        A = np.eye(d_x)
        # pack upper-triangular entries
        packed = []
        for i in range(d_x):
            for j in range(i, d_x):
                packed.append(A[i, j])
        packed = np.array(packed).reshape(-1, 1)
        return np.vstack([b, packed])

    def param_dim(self, d_x: int) -> int:
        return d_x + d_x * (d_x + 1) // 2

    @staticmethod
    def _vec_to_mat(vec, d_x):
        vec = np.atleast_1d(vec.ravel())
        A = np.zeros((d_x, d_x))
        idx = 0
        for i in range(d_x):
            for j in range(i, d_x):
                A[i, j] = vec[idx]
                if i != j:
                    A[j, i] = A[i, j]
                idx += 1
        return A


class ExponentialModel(BaseModel):
    """Exact exponential link, with no index or gradient clipping.

    Choose the parameter region and input scale carefully. Numerical overflow
    raises an error in SSGMM.update; the estimator never silently clips it.
    """

    def predict(self, theta, x):
        return np.exp(x @ np.asarray(theta).reshape(-1, 1))

    def gradient(self, theta, x):
        return self.predict(theta, x) * x
