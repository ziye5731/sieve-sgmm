"""Normalized probabilists' Hermite features in the original research ordering.

Orthonormality requires independent standard Gaussian instrument coordinates.
"""

from __future__ import annotations
import math
import numpy as np

_INDEX_CACHE: dict = {}  # cached row-major index arrays for the sieve blocks


def _upper_pairs(n: int, offset: int = 1) -> tuple:
    """Row-major (i, j) index arrays with j >= i + offset (cached)."""
    key = ("triu", n, offset)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = np.triu_indices(n, offset)
        _INDEX_CACHE[key] = idx
    return idx


def _offdiag_pairs(n: int) -> tuple:
    """Row-major (i, j) index arrays with j != i (cached)."""
    key = ("offdiag", n)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = ii != jj
        idx = (ii[mask], jj[mask])
        _INDEX_CACHE[key] = idx
    return idx


def normalize_degrees(degrees):
    """Return sorted, distinct integer total degrees, each between zero and three."""
    if isinstance(degrees, (int, np.integer)) and not isinstance(degrees, (bool, np.bool_)):
        degrees = (degrees,)
    try:
        values = tuple(degrees)
    except TypeError as exc:
        raise ValueError("degrees must be an integer or a nonempty sequence of integers") from exc
    if not values or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k <= 3
        for k in values
    ):
        raise ValueError("Hermite degrees must be integers from 0 through 3")
    return tuple(sorted(set(int(k) for k in values)))


def feature_dimension(n_instruments, degrees=(0, 1, 2)):
    """Number of scalar moments in complete selected total-degree blocks."""
    if (
        isinstance(n_instruments, (bool, np.bool_))
        or not isinstance(n_instruments, (int, np.integer))
        or n_instruments < 1
    ):
        raise ValueError("n_instruments must be a positive integer")
    return sum(math.comb(n_instruments + k - 1, k) for k in normalize_degrees(degrees))


def _hermite_block(z: np.ndarray, degree: int) -> np.ndarray:
    """One orthonormal (probabilists') Hermite block of exactly `degree`.

        degree 0: H_0 = 1
        degree 1: H_1(z_i) = z_i
        degree 2: H_2(z_i) = (z_i^2 - 1)/sqrt(2),
                  H_1(z_i) H_1(z_j) = z_i z_j              (i < j)
        degree 3: H_3(z_i) = (z_i^3 - 3 z_i)/sqrt(6),
                  H_2(z_i) H_1(z_j)                        (i != j),
                  H_1(z_i) H_1(z_j) H_1(z_k) = z_i z_j z_k (i < j < k)

    Built with whole-array operations (cached index arrays); the column order
    matches the obvious nested loops.
    """
    d_z = z.shape[1]
    if degree == 0:
        return np.ones((z.shape[0], 1))
    if degree == 1:
        return z
    if degree == 2:
        h2 = (z**2 - 1.0) / np.sqrt(2.0)  # (B, d_z)
        outer = z[:, :, None] * z[:, None, :]  # (B, d_z, d_z)
        iu, ju = _upper_pairs(d_z, 1)
        return np.concatenate([h2, outer[:, iu, ju]], axis=1)
    if degree == 3:
        h3 = (z**3 - 3.0 * z) / np.sqrt(6.0)  # (B, d_z)
        h2 = (z**2 - 1.0) / np.sqrt(2.0)  # (B, d_z)
        cross23 = h2[:, :, None] * z[:, None, :]  # (B, d_z, d_z) [i, j]
        ii, jj = _offdiag_pairs(d_z)
        blocks = [h3, cross23[:, ii, jj]]  # H3, then H2*H1
        # z_i z_j z_k with i < j < k, in i-major then (j, k) row-major order
        for i in range(d_z - 2):
            sub = z[:, i + 1 :]
            outer = sub[:, :, None] * sub[:, None, :]
            su, sv = _upper_pairs(sub.shape[1], 1)
            blocks.append(z[:, i : i + 1] * outer[:, su, sv])
        return np.concatenate(blocks, axis=1)
    raise ValueError(f"Hermite sieve only supports degrees 0-3, got {degree}")


def hermite_features(z, degrees=(0, 1, 2)):
    """Map an (n, n_instruments) array to (n, n_moments) Hermite features.

    Blocks: constant; linear; H2 then pair interactions; H3 then H2*H1
    in ordered distinct pairs, then triple interactions. Degree blocks are
    sorted; selecting a block includes every interaction of that total degree.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] < 1 or not np.all(np.isfinite(z)):
        raise ValueError("z must be a finite two-dimensional array with at least one column")
    with np.errstate(over="raise", invalid="raise"):
        return np.concatenate([_hermite_block(z, k) for k in normalize_degrees(degrees)], axis=1)
