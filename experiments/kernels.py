"""Numerical kernels and diagnostics for the SSGMM theory study.

The recursion is checked against ssgmm.SSGMM in tests/test_kernels.py. A rank-one identity caches
S.T@S, rebuilt every 256 observations to limit floating-point drift. Nothing
is batched in time: one observation drives each parameter/Jacobian update.
Numba is an optional study dependency; the public estimator needs only NumPy.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from ssgmm.basis import hermite_features as _hermite_degrees_features
from ssgmm.models import QuadraticModel


def basis_code(basis):
    return {(1,): 0, (0, 1): 1, (0, 1, 2): 2, (0, 1, 2, 3): 3, (2,): 4}[tuple(basis)]


def feature_dim(dz, code):
    return [dz, dz + 1, (dz + 1) * (dz + 2) // 2, (dz + 1) * (dz + 2) * (dz + 3) // 6, 1][code]


@njit(cache=True)
def hermite_one(z, code, out):
    """Match the existing Hermite feature order, including mixed cubics."""
    n = len(z)
    if code == 4:
        out[0] = (z[0] * z[0] - 1.0) / np.sqrt(2.0)
        return
    pos = 0
    if code != 0:
        out[pos] = 1.0
        pos += 1
    for i in range(n):
        out[pos] = z[i]
        pos += 1
    if code < 2:
        return
    for i in range(n):
        out[pos] = (z[i] * z[i] - 1.0) / np.sqrt(2.0)
        pos += 1
    for i in range(n):
        for j in range(i + 1, n):
            out[pos] = z[i] * z[j]
            pos += 1
    if code < 3:
        return
    for i in range(n):
        out[pos] = (z[i] ** 3 - 3 * z[i]) / np.sqrt(6.0)
        pos += 1
    for i in range(n):
        for j in range(n):
            if i != j:
                out[pos] = (z[i] * z[i] - 1.0) / np.sqrt(2.0) * z[j]
                pos += 1
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                out[pos] = z[i] * z[j] * z[k]
                pos += 1


@njit(cache=True)
def chol_solve(gram, rhs, ridge):
    """Small positive-definite solve; no jitter or regularization changes."""
    d = len(rhs)
    lower = np.zeros((d, d))
    result = rhs.copy()
    for i in range(d):
        for j in range(i + 1):
            v = gram[i, j] + (ridge if i == j else 0.0)
            for k in range(j):
                v -= lower[i, k] * lower[j, k]
            if i == j:
                if v <= 0 or not np.isfinite(v):
                    return np.full(d, np.nan)
                lower[i, j] = np.sqrt(v)
            else:
                lower[i, j] = v / lower[j, j]
    for i in range(d):
        for j in range(i):
            result[i] -= lower[i, j] * result[j]
        result[i] /= lower[i, i]
    for i in range(d - 1, -1, -1):
        for j in range(i + 1, d):
            result[i] -= lower[j, i] * result[j]
        result[i] /= lower[i, i]
    return result


@njit(cache=True)
def gram_from_state(state):
    p, d = state.shape
    gram = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            v = 0.0
            for k in range(p):
                v += state[k, i] * state[k, j]
            gram[i, j] = v
            gram[j, i] = v
    return gram


@njit(cache=True)
def train_path(
    draws,
    gamma,
    truth,
    initial,
    center,
    radius,
    model,
    first_stage,
    code,
    a,
    eta,
    kappa,
    offset,
    ridge,
    curvature,
    oracle,
    checkpoints,
    population_g,
):
    """Process an ordinary stream. models: 0 calibration, 1 quadratic, 2 exp.

    draws columns: calibration (z,u,nu), otherwise (z,c,v,e).
    Returns per-checkpoint states and diagnostics, retaining all repetitions.
    """
    dz, dx = gamma.shape
    d = len(truth)
    p = feature_dim_jit(dz, code)
    theta = initial.copy()
    average = np.zeros(d)
    state = np.zeros((p, d))
    gram = np.zeros((d, d))
    psi = np.empty(p)
    x = np.empty(dx)
    grad = np.empty(d)
    q = np.empty(d)
    stpsi = np.empty(d)
    nc = len(checkpoints)
    raw = np.full((nc, d), np.nan)
    avg = raw.copy()
    diagnostics = np.full(
        (nc, 5), np.nan
    )  # tracking squared, hits, maxdist, maxindex, rebuild error
    hits = 0
    maxdist = np.linalg.norm(initial - truth)
    maxindex = 0.0
    next_cp = 0
    failure_step = 0
    max_gram_error = 0.0
    for t0 in range(len(draws)):
        t = t0 + 1
        z = draws[t0, :dz]
        hermite_one(z, code, psi)
        if model == 0:
            x[0] = gamma[0, 0] * z[0] + draws[t0, 1]
            u = draws[t0, 1]
            noise = 0.5 * u + 0.5 * (u * u - 1.0) + draws[t0, 2]
            y = (truth[0] + curvature * truth[0] ** 2) * x[0] ** 2 + noise
            pred = (theta[0] + curvature * theta[0] ** 2) * x[0] ** 2
            grad[0] = (1 + 2 * curvature * theta[0]) * x[0] ** 2
        else:
            for j in range(dx):
                h = 0.0
                for k in range(dz):
                    h += gamma[k, j] * z[k]
                if first_stage == 1:
                    h = max(0.0, h)
                elif first_stage == 2:
                    h = np.tanh(h)
                x[j] = h + (draws[t0, dz + j] + draws[t0, dz + dx + j]) / np.sqrt(2.0)
            conf = 0.0
            for j in range(dx):
                conf += draws[t0, dz + j] / np.sqrt(2.0)
            if model == 1:
                noise = 0.5 * conf + draws[t0, dz + 2 * dx]
                for j in range(dx):
                    q[j] = x[j]
                pos = dx
                for j in range(dx):
                    for k in range(j, dx):
                        q[pos] = x[j] * x[k] * (1.0 if j == k else 2.0)
                        pos += 1
                pred = 0.0
                y = noise
                for j in range(d):
                    pred += theta[j] * q[j]
                    y += truth[j] * q[j]
                    grad[j] = q[j]
            else:
                noise = 0.25 * conf + 0.1 * draws[t0, dz + 2 * dx]
                index = 0.0
                index_star = 0.0
                for j in range(dx):
                    index += theta[j] * x[j]
                    index_star += truth[j] * x[j]
                maxindex = max(maxindex, abs(index))
                pred = np.exp(index)
                y = np.exp(index_star) + noise
                for j in range(d):
                    grad[j] = pred * x[j]
        residual = pred - y
        for j in range(d):
            v = 0.0
            for k in range(p):
                v += state[k, j] * psi[k]
            stpsi[j] = v
        if oracle:
            # Only the scalar calibration model has an exact moving Jacobian oracle.
            g = (
                np.sqrt(2.0) * gamma[0, 0] ** 2 * (1 + 2 * curvature * theta[0])
                if code == 4
                else 0.0
            )
            direction = np.array([g / (g * g + ridge) * psi[0] * residual])
        else:
            direction = chol_solve(gram, stpsi * residual, ridge)
            if not np.all(np.isfinite(direction)):
                gram = gram_from_state(state)
                direction = chol_solve(gram, stpsi * residual, ridge)
        if not np.all(np.isfinite(direction)):
            failure_step = t
            break
        alpha = a / (t + offset) ** eta
        beta = 1.0 / (t + offset) ** kappa
        proposal = theta - alpha * direction
        length = 0.0
        for j in range(d):
            length += (proposal[j] - center[j]) ** 2
        length = np.sqrt(length)
        if length > radius:
            hits += 1
            proposal = center + (proposal - center) * (radius / length)
        theta = proposal
        average += (theta - average) / t
        dist = 0.0
        for j in range(d):
            dist += (theta[j] - truth[j]) ** 2
        maxdist = max(maxdist, np.sqrt(dist))
        # Exact rank-one identity for ((1-beta)S + beta*psi*grad.T).T @ (...).
        psinorm = 0.0
        for k in range(p):
            psinorm += psi[k] * psi[k]
        for i in range(d):
            for j in range(i + 1):
                v = (
                    (1 - beta) ** 2 * gram[i, j]
                    + beta * (1 - beta) * (stpsi[i] * grad[j] + grad[i] * stpsi[j])
                    + beta * beta * psinorm * grad[i] * grad[j]
                )
                gram[i, j] = v
                gram[j, i] = v
        for k in range(p):
            for j in range(d):
                state[k, j] = (1 - beta) * state[k, j] + beta * psi[k] * grad[j]
        if t % 256 == 0:
            fresh = gram_from_state(state)
            denom = max(1.0, np.max(np.abs(fresh)))
            max_gram_error = max(max_gram_error, np.max(np.abs(gram - fresh)) / denom)
            gram = fresh
        if t == checkpoints[next_cp]:
            raw[next_cp] = theta
            avg[next_cp] = average
            track = np.nan
            if model == 0:
                g = (
                    np.sqrt(2.0) * gamma[0, 0] ** 2 * (1 + 2 * curvature * theta[0])
                    if code == 4
                    else 0.0
                )
                track = (state[0, 0] - g) ** 2
            elif model == 1:
                track = 0.0
                for k in range(p):
                    for j in range(d):
                        track += (state[k, j] - population_g[k, j]) ** 2
            diagnostics[next_cp, 0] = track
            diagnostics[next_cp, 1] = hits
            diagnostics[next_cp, 2] = maxdist
            diagnostics[next_cp, 3] = maxindex
            diagnostics[next_cp, 4] = max_gram_error
            next_cp += 1
            if next_cp == nc:
                break
    return raw, avg, diagnostics, state, failure_step


@njit(cache=True)
def feature_dim_jit(dz, code):
    if code == 0:
        return dz
    if code == 1:
        return dz + 1
    if code == 2:
        return (dz + 1) * (dz + 2) // 2
    if code == 3:
        return (dz + 1) * (dz + 2) * (dz + 3) // 6
    return 1


def quadratic_population_g(gamma, basis):
    """Exact E[psi(z) q(x).T] for x=Gamma.T z+w, Var(w)=I."""
    dz, dx = gamma.shape
    pairs = list(zip(*np.triu_indices(dx)))
    d = dx + len(pairs)
    allg = np.zeros((feature_dim(dz, 3), d))
    for k, (i, j) in enumerate(pairs):
        allg[0, dx + k] = (gamma[:, i] @ gamma[:, j] + (i == j)) * (1 if i == j else 2)
    allg[1 : 1 + dz, :dx] = gamma
    pos = 1 + dz
    for u in range(dz):
        for k, (i, j) in enumerate(pairs):
            allg[pos, dx + k] = np.sqrt(2) * gamma[u, i] * gamma[u, j] * (1 if i == j else 2)
        pos += 1
    for u in range(dz):
        for v in range(u + 1, dz):
            for k, (i, j) in enumerate(pairs):
                allg[pos, dx + k] = (gamma[u, i] * gamma[v, j] + gamma[v, i] * gamma[u, j]) * (
                    1 if i == j else 2
                )
            pos += 1
    return allg[basis_rows(dz, basis)]


def basis_rows(dz, basis):
    result = []
    start = 0
    for k in range(4):
        count = [1, dz, dz * (dz + 1) // 2, dz * (dz + 1) * (dz + 2) // 6][k]
        if k in basis:
            result.extend(range(start, start + count))
        start += count
    return result


def multivariate_data(draws, gamma, first_stage, model, truth):
    dz, dx = gamma.shape
    h = draws[:, :dz] @ gamma
    if first_stage == 1:
        h = np.maximum(h, 0)
    elif first_stage == 2:
        h = np.tanh(h)
    c = draws[:, dz : dz + dx] / np.sqrt(2)
    x = h + c + draws[:, dz + dx : dz + 2 * dx] / np.sqrt(2)
    eps = (0.5 if model == 1 else 0.25) * c.sum(axis=1) + (1.0 if model == 1 else 0.1) * draws[
        :, -1
    ]
    if model == 1:
        pred = QuadraticModel().predict(truth, x).ravel()
    else:
        pred = np.exp(x @ truth)
    return x, pred + eps, h


def pilot_jacobian(gamma, first_stage, model, truth, seed, n, batch):
    """Independent batch means, integrating out Gaussian first-stage noise.

    Quadratic: E[q(x)|z] is exact. Exponential: E[e^(theta'x)x|z]
    = exp(theta'h + ||theta||^2/2)*(h+theta).
    """
    rng = np.random.default_rng(seed)
    means = []
    dz, dx = gamma.shape
    for start in range(0, n, batch):
        size = min(batch, n - start)
        z = rng.standard_normal((size, dz))
        h = z @ gamma
        if first_stage == 1:
            h = np.maximum(h, 0)
        elif first_stage == 2:
            h = np.tanh(h)
        if model == 1:
            q = QuadraticModel().gradient(truth, h)
            pos = dx
            for i in range(dx):
                for j in range(i, dx):
                    if i == j:
                        q[:, pos] += 1
                    pos += 1
        else:
            q = np.exp(h @ truth + 0.5 * (truth @ truth))[:, None] * (h + truth)
        psi = _hermite_degrees_features(z, (0, 1, 2, 3))
        means.append(psi.T @ q / size)
    return np.stack(means)


def aggregate_paths(theta, average, truth, steps):
    """Existing results.npz naming, plus mean-square and tail diagnostics."""
    out = {"steps": steps}
    for prefix, values in [("", theta), ("average_", average)]:
        err = np.linalg.norm(values - truth, axis=-1)  # repeat, checkpoint
        for name, arr in [
            ("mean", err.mean(0)),
            ("std", err.std(0)),
            ("median", np.median(err, axis=0)),
            ("q25", np.quantile(err, 0.25, axis=0)),
            ("q75", np.quantile(err, 0.75, axis=0)),
            ("q90", np.quantile(err, 0.9, axis=0)),
        ]:
            out[prefix + "param_error_" + name] = arr
        sq = err**2
        out[prefix + "param_mse"] = sq.mean(0)
        out[prefix + "param_mse_mcse"] = sq.std(0, ddof=1) / np.sqrt(len(sq))
    return out
