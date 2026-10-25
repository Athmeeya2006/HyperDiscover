"""Derivative targets (Stage 0, option A): finite differences, Savitzky–Golay, Tikhonov.

All functions take X [M, T, N, d] (time on axis 1) and return dX/dt with the same shape.
Total-variation regularised differentiation (``tv``) follows the small-data lagged-diffusivity
iteration of Chartrand (2011, "Numerical differentiation of noisy, nonsmooth data"; verify);
Tikhonov (second-difference penalty) is the smooth alternative (DECISIONS.md D-010, D-025).

References (references/references.bib): savitzky1964smoothing, chartrand2011numerical.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _uniform_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12):
        raise ValueError("this derivative method needs a uniform sampling interval")
    return float(dt[0])


def finite_difference(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Second-order central differences (one-sided at the ends); non-uniform t allowed."""
    return np.gradient(X, t, axis=1, edge_order=2)


def savgol_derivative(
    X: np.ndarray, t: np.ndarray, window: int = 11, polyorder: int = 3
) -> np.ndarray:
    """Savitzky–Golay smoothed first derivative (uniform t)."""
    dt = _uniform_dt(t)
    T = X.shape[1]
    window = min(window if window % 2 else window + 1, T if T % 2 else T - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(X, window, polyorder, deriv=1, delta=dt, axis=1, mode="interp")


def tikhonov_derivative(X: np.ndarray, t: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Tikhonov-regularised differentiation.

    Solves min_u ||C u − (x − x_0)||² + α ||D2 u||² per series, where C is cumulative
    trapezoidal integration and D2 the second-difference operator; α is relative to
    ||CᵀC||. One factorisation is shared by all M·N·d series (uniform t).
    """
    _uniform_dt(t)
    T = len(t)
    dt = np.diff(t)
    C = np.zeros((T, T))
    for n in range(1, T):  # trapezoid: ∫_0^{t_n} u = Σ_{m<n} dt_m (u_m + u_{m+1}) / 2
        C[n] = C[n - 1]
        C[n, n - 1] += 0.5 * dt[n - 1]
        C[n, n] += 0.5 * dt[n - 1]
    D2 = np.diff(np.eye(T), n=2, axis=0)
    G = C.T @ C
    lam = alpha * np.trace(G) / T
    Kinv = np.linalg.inv(G + lam * (D2.T @ D2))
    Y = np.moveaxis(X, 1, -1)  # [M, N, d, T]
    rhs = (Y - Y[..., :1]) @ C  # (x - x0)^T C  -> [..., T]
    U = rhs @ Kinv.T
    return np.moveaxis(U, -1, 1)


def _trapezoid_operator(t: np.ndarray) -> np.ndarray:
    """C with (C u)_n = ∫_0^{t_n} u (trapezoid rule)."""
    T = len(t)
    dt = np.diff(t)
    C = np.zeros((T, T))
    for n in range(1, T):
        C[n] = C[n - 1]
        C[n, n - 1] += 0.5 * dt[n - 1]
        C[n, n] += 0.5 * dt[n - 1]
    return C


def tv_derivative(
    X: np.ndarray, t: np.ndarray, alpha: float = 1e-3, iters: int = 20, eps: float = 1e-8
) -> np.ndarray:
    """Total-variation regularised derivative.

    Minimises ½‖C u − (x − x_0)‖² + α_eff Σ sqrt((D u)² + ε) per series by lagged diffusivity:
    u ← u − H⁻¹ g with H = Cᵀ C + α_eff Dᵀ E D, E = diag(1/sqrt((D u)² + ε)). α is relative to
    ‖CᵀC‖ (trace/T) times the series' derivative scale, so one α works across systems.
    Initialised with the Tikhonov solution. Cost: ``iters`` dense T×T solves per series.
    """
    _uniform_dt(t)
    T = len(t)
    C = _trapezoid_operator(t)
    D = np.diff(np.eye(T), axis=0)
    G = C.T @ C
    scale = np.trace(G) / T
    u0 = tikhonov_derivative(X, t, alpha=1e-4)
    Y = np.moveaxis(X, 1, -1).reshape(-1, T)  # [S, T]
    U = np.moveaxis(u0, 1, -1).reshape(-1, T).copy()
    F = Y - Y[:, :1]
    for s in range(len(Y)):
        u = U[s]
        f = F[s]
        a = alpha * scale * (np.abs(u).max() + 1e-12)
        CtF = C.T @ f
        for _ in range(iters):
            w = 1.0 / np.sqrt((D @ u) ** 2 + eps)
            L = D.T @ (w[:, None] * D)
            g = G @ u - CtF + a * (L @ u)
            u = u - np.linalg.solve(G + a * L, g)
        U[s] = u
    return np.moveaxis(U.reshape(np.moveaxis(X, 1, -1).shape), -1, 1)


def estimate_derivatives(
    X: np.ndarray, t: np.ndarray, method: str = "savgol", segment_starts=None, **kw
) -> np.ndarray:
    """Dispatch to a derivative estimator, applied separately to each continuous segment."""
    fn = {
        "fd": finite_difference,
        "savgol": savgol_derivative,
        "tikhonov": tikhonov_derivative,
        "tv": tv_derivative,
    }[method]
    starts = list(segment_starts or [0]) + [X.shape[1]]
    out = np.empty_like(X)
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 3:  # too short for edge_order=2 / filters: first-order slope (or 0 for 1 point)
            out[:, a:b] = (
                np.gradient(X[:, a:b], t[a:b], axis=1, edge_order=1) if b - a == 2 else 0.0
            )
        else:
            out[:, a:b] = fn(X[:, a:b], t[a:b], **kw)
    return out
