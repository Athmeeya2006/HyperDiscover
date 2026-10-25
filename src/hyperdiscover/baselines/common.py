"""Shared helpers for regression baselines."""

from __future__ import annotations

import numpy as np


def stlsq(
    Theta: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.05,
    alpha: float = 1e-6,
    max_iter: int = 20,
    normalize: bool = True,
) -> np.ndarray:
    """Sequentially thresholded least squares (Brunton et al. 2016), ridge-stabilised.

    Thresholding is applied to coefficients of *standardised* columns when ``normalize``
    (so ``threshold`` is in units of target std per column std). Returns raw-scale coefficients.
    """
    std = Theta.std(0) if normalize else np.ones(Theta.shape[1])
    std = np.where(std > 1e-12, std, 1.0)
    A = Theta / std
    ys = y.std() if y.std() > 0 else 1.0
    yn = y / ys
    active = np.ones(A.shape[1], dtype=bool)
    w = np.zeros(A.shape[1])
    for _ in range(max_iter):
        if not active.any():
            break
        Aa = A[:, active]
        G = Aa.T @ Aa + alpha * len(A) * np.eye(Aa.shape[1])
        w_a = np.linalg.solve(G, Aa.T @ yn)
        w = np.zeros(A.shape[1])
        w[active] = w_a
        new = np.abs(w) >= threshold
        if np.array_equal(new, active):
            break
        active = new
    w[~active] = 0.0
    return w * ys / std


def scores_from_blocks(w: np.ndarray, Theta: np.ndarray, blocks: list[tuple]) -> dict:
    """Score per block key = max |coef × column std| (importance on the data scale)."""
    imp = np.abs(w) * Theta.std(0)
    out: dict = {}
    for col, key in enumerate(blocks):
        if key is None:
            continue
        out[key] = max(out.get(key, 0.0), float(imp[col]))
    return out
