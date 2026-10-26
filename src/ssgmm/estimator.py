"""Single-observation, two-time-scale projected sieve GMM."""

from __future__ import annotations

import numpy as np

from .basis import feature_dimension, hermite_features, normalize_degrees


class SSGMM:
    """Streaming sieve GMM with a historical Jacobian and iterate averaging.

    Parameters
    ----------
    model : object
        Implements ``predict(theta, x) -> (n, 1)`` and
        ``gradient(theta, x) -> (n, d)``. Theta is a one-dimensional vector.
    n_instruments : int
        Number of instrument coordinates.
    theta0 : array-like, shape (d,)
        Deterministic initial parameter; projected into the ball immediately.
    degrees : int or sequence of int
        Complete Hermite total-degree blocks, drawn from 0, 1, 2, 3.
    lr, lr_decay : float
        Parameter schedule ``lr / (t + lr_offset)**lr_decay``, t starts at 1.
    jacobian_lr, jacobian_lr_decay : float
        Jacobian schedule with the same offset. Requires
        ``0.5 < jacobian_lr_decay < lr_decay <= 1`` and initial beta <= 1.
    lr_offset : float
        Nonnegative schedule offset.
    ridge : float
        Fixed positive ridge in the gain, not in the moment equation.
    radius, center : float, array-like or None
        Positive finite radius and center of the Euclidean projection ball.
        The default center is zero. Choosing a region containing the truth
        and satisfying the theory's curvature conditions is the user's task.

    Notes
    -----
    Calls to fit continue the same stream. Persistent state has fixed size;
    no observations or parameter paths are retained. Numerical errors raise
    before the observation is committed, without clipping or dropping it.
    """

    def __init__(
        self,
        model,
        n_instruments,
        theta0,
        *,
        degrees=(0, 1, 2),
        lr=0.05,
        lr_decay=0.8,
        jacobian_lr=1.0,
        jacobian_lr_decay=0.6,
        lr_offset=10.0,
        ridge=0.01,
        radius=20.0,
        center=None,
    ):
        self.degrees = normalize_degrees(degrees)
        self.n_moments = feature_dimension(n_instruments, self.degrees)
        self.n_instruments = int(n_instruments)
        if not callable(getattr(model, "predict", None)) or not callable(
            getattr(model, "gradient", None)
        ):
            raise TypeError("model must implement predict(theta, x) and gradient(theta, x)")
        self.model = model
        names = (
            "lr",
            "lr_decay",
            "jacobian_lr",
            "jacobian_lr_decay",
            "lr_offset",
            "ridge",
            "radius",
        )
        for name, value in zip(
            names, (lr, lr_decay, jacobian_lr, jacobian_lr_decay, lr_offset, ridge, radius)
        ):
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        if min(self.lr, self.jacobian_lr, self.ridge, self.radius) <= 0 or self.lr_offset < 0:
            raise ValueError(
                "lr, jacobian_lr, ridge, radius must be positive; lr_offset must be nonnegative"
            )
        if not 0.5 < self.jacobian_lr_decay < self.lr_decay <= 1:
            raise ValueError("require 0.5 < jacobian_lr_decay < lr_decay <= 1")
        if self.jacobian_lr / (1 + self.lr_offset) ** self.jacobian_lr_decay > 1:
            raise ValueError(
                "initial Jacobian step must be <= 1; increase lr_offset or decrease jacobian_lr"
            )
        theta = np.asarray(theta0, dtype=float)
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("theta0 must be a finite nonempty one-dimensional vector")
        self.center = (
            np.zeros_like(theta) if center is None else np.asarray(center, dtype=float).copy()
        )
        if self.center.shape != theta.shape or not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite and have the same shape as theta0")
        self._theta = self._project(theta.copy())
        self._average = self._theta.copy()
        self._jacobian = np.zeros((self.n_moments, theta.size))
        self.n_samples_seen_ = 0
        self._n_regressors = None

    def _project(self, theta):
        delta = theta - self.center
        norm = np.linalg.norm(delta)
        if not np.isfinite(norm):
            raise FloatingPointError("nonfinite projection distance")
        return self.center + delta * (self.radius / norm) if norm > self.radius else theta

    @property
    def theta_(self):
        """Last parameter iterate, shape (d,); returns a copy."""
        return self._theta.copy()

    @property
    def average_(self):
        """Mean of updated iterates, excluding theta0; returns a copy."""
        return self._average.copy()

    @property
    def coef_(self):
        """Reported estimate: the full-history average (theta0 before fitting)."""
        return self.average_

    @property
    def jacobian_(self):
        """Historical moment Jacobian, shape (n_moments, d); returns a copy."""
        return self._jacobian.copy()

    def update(self, z, x, y):
        """Consume exactly one (z, x, y) observation and return self.

        Both model evaluations use the old parameter; the direction uses the
        old Jacobian. Only after projection is the same observation absorbed
        into the Jacobian state and the updated parameter into the average.
        """
        z, x, y = np.asarray(z, dtype=float), np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if z.shape != (self.n_instruments,) or not np.all(np.isfinite(z)):
            raise ValueError(f"z must be finite with shape ({self.n_instruments},)")
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite nonempty one-dimensional vector")
        if self._n_regressors is not None and x.size != self._n_regressors:
            raise ValueError("regressor dimension changed within the stream")
        if y.ndim != 0 or not np.isfinite(y):
            raise ValueError("y must be a finite scalar")
        t = self.n_samples_seen_ + 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            psi = hermite_features(z[None, :], self.degrees)[0]
            # Copies prevent a custom model from modifying the historical state.
            pred = np.asarray(
                self.model.predict(self._theta.copy(), x[None, :].copy()), dtype=float
            )
            grad = np.asarray(
                self.model.gradient(self._theta.copy(), x[None, :].copy()), dtype=float
            )
            if pred.shape != (1, 1) or grad.shape != (1, self._theta.size):
                raise ValueError("model must return predictions (1, 1) and gradients (1, d)")
            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"nonfinite model output at observation {t}")
            moment = psi * (pred[0, 0] - float(y))
            old = self._jacobian
            direction = np.linalg.solve(
                old.T @ old + self.ridge * np.eye(old.shape[1]), old.T @ moment
            )
            alpha = self.lr / (t + self.lr_offset) ** self.lr_decay
            beta = self.jacobian_lr / (t + self.lr_offset) ** self.jacobian_lr_decay
            theta = self._project(self._theta - alpha * direction)
            jacobian = (1 - beta) * old + beta * np.outer(psi, grad[0])
            average = (1 - 1 / t) * self._average + (1 / t) * theta
            if not all(np.all(np.isfinite(a)) for a in (theta, jacobian, average)):
                raise FloatingPointError(f"nonfinite update at observation {t}")
        self._theta, self._jacobian, self._average = theta, jacobian, average
        self.n_samples_seen_ = t
        self._n_regressors = x.size
        return self

    def fit(self, z, x, y):
        """Consume array rows once in order, continuing any previous updates.

        z and x must have shapes (n, n_instruments) and (n, d_x). y may
        have shape (n,) or (n, 1). The full data are supplied by the caller;
        use update for a stream without storing the data in memory.
        """
        z, x, y = np.asarray(z, dtype=float), np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.n_instruments or x.ndim != 2 or x.shape[1] == 0:
            raise ValueError(
                "z and x must be two-dimensional with the declared instrument dimension"
            )
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1 or len(z) != len(x) or len(z) != len(y):
            raise ValueError(
                "z, x, y must have equal numbers of observations; y must be (n,) or (n, 1)"
            )
        if not all(np.all(np.isfinite(a)) for a in (z, x, y)):
            raise ValueError("training data must be finite")
        if self._n_regressors is not None and x.shape[1] != self._n_regressors:
            raise ValueError("regressor dimension changed within the stream")
        for zi, xi, yi in zip(z, x, y):
            self.update(zi, xi, yi)
        return self

    def predict(self, x, *, averaged=True):
        """Evaluate the structural function at the averaged or last estimate."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite two-dimensional regressor array")
        if self._n_regressors is not None and x.shape[1] != self._n_regressors:
            raise ValueError("prediction regressor dimension differs from the fitted stream")
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(
                self.model.predict(self.coef_ if averaged else self.theta_, x.copy()), dtype=float
            )
        if result.shape != (len(x), 1) or not np.all(np.isfinite(result)):
            raise ValueError("model prediction must be finite with shape (n, 1)")
        return result
