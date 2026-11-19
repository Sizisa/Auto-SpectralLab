"""Fluorescence baseline estimation for evenly sampled Raman spectra."""

from warnings import warn

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import expit


def arpls(intensity, lam=1e5, *, pad_width=None, tol=1e-3, max_iter=100):
    """Return (corrected_intensity, baseline) using sparse arPLS.

    The samples should be evenly spaced in Raman shift. Increasing lam
    favors smoother baselines. pad_width sets the number of samples
    reflected at each end and defaults to 10% of the spectrum length.
    """
    y = np.asarray(intensity, dtype=float)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("Supply a finite 1-D spectrum with at least three samples.")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Require positive finite lam and tol between 0 and 1.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if pad_width is None:
        pad_width = max(1, y.size // 10)
    if not isinstance(pad_width, (int, np.integer)) or not 0 <= pad_width < y.size:
        raise ValueError("pad_width must be an integer from 0 through len(intensity) - 1.")

    padded = np.pad(y, pad_width, mode="reflect")
    size = padded.size
    differences = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(size - 2, size), format="csc")
    penalty = lam * (differences.T @ differences)
    weights = np.ones(size)
    epsilon = np.finfo(float).eps
    residual_floor = epsilon * np.max(np.abs(padded)) * (1 + 16 * lam)

    for _ in range(max_iter):
        baseline = spsolve(penalty + diags(weights, format="csc"), weights * padded)
        residual = padded - baseline
        negative = residual[residual < 0]
        if negative.size < 2:
            break
        spread = negative.std()
        if spread <= residual_floor:
            break
        next_weights = expit(-2 * (residual - (2 * spread - negative.mean())) / spread)
        next_weights = np.maximum(next_weights, epsilon)
        if np.linalg.norm(next_weights - weights) <= tol * np.linalg.norm(weights):
            break
        weights = next_weights
    else:
        warn("arPLS reached max_iter before the weights converged.", RuntimeWarning, stacklevel=2)

    baseline = baseline[pad_width:pad_width + y.size]
    return y - baseline, baseline
