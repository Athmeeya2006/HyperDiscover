"""Weak-form targets (Stage 0, option B), after weak SINDy (Messenger & Bortz 2021, verify).

For a compactly supported test function φ on a window [a, b] (φ(a) = φ(b) = 0),

    ∫ φ(t) ẋ(t) dt = −∫ φ'(t) x(t) dt,

so the model's RHS F(x) can be matched to data without differentiating x:

    target_w = −Σ_l φ'_l x(t_{s_w + l}) Δt_l,      pred_w = Σ_l φ_l F(x(t_{s_w + l})) Δt_l.

Test functions: φ(t) = ((t − a)(b − t))^p normalised so that Σ_l φ_l Δt_l = 1 (the weak
target is then a weighted average of ẋ over the window).

References (references/references.bib): messenger2021weak, reinbold2020using.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WeakForm:
    """Window layout and quadrature weights (uniform sampling).

    Attributes:
        phi: [L] quadrature-weighted test function values (φ_l Δt_l).
        dphi: [L] quadrature-weighted derivative values (φ'_l Δt_l).
        starts: [W] window start indices into the time axis.
    """

    phi: np.ndarray
    dphi: np.ndarray
    starts: np.ndarray

    @property
    def length(self) -> int:
        return len(self.phi)


def make_weak_form(
    t: np.ndarray, window: int = 21, stride: int = 5, power: int = 4, segment_starts=None
) -> WeakForm:
    """Build test functions on sliding windows of ``window`` points (uniform t).

    Windows never straddle a segment boundary (e.g. a kick).
    """
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6):
        raise ValueError("weak form currently needs a uniform sampling interval")
    dt = float(dt[0])
    T = len(t)
    bounds = list(segment_starts or [0]) + [T]
    longest = max(hi - lo for lo, hi in zip(bounds[:-1], bounds[1:]))
    # short kick segments: shrink the window to fit (the requested size is an upper bound)
    window = min(window, T, longest)
    if window < 5:
        raise ValueError(f"segments too short for a weak-form window (longest {longest} points)")
    s = np.arange(window) * dt
    a, b = s[0], s[-1]
    base = (s - a) * (b - s)
    phi = base**power
    dphi = power * base ** (power - 1) * ((b - s) - (s - a))
    w = np.full(window, dt)  # trapezoid weights
    w[0] = w[-1] = 0.5 * dt
    norm = np.sum(phi * w)
    phi_q, dphi_q = phi * w / norm, dphi * w / norm
    starts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        starts.extend(range(lo, hi - window + 1, stride))
    if not starts:
        seg = min(hi - lo for lo, hi in zip(bounds[:-1], bounds[1:]))
        raise ValueError(
            f"no weak-form window of {window} points fits a segment (shortest segment {seg} points); "
            "reduce weak_kw.window"
        )
    return WeakForm(phi_q, dphi_q, np.asarray(starts, dtype=np.int64))


def window_states(X: np.ndarray, wf: WeakForm) -> np.ndarray:
    """Gather window states: X [M, T, N, d] -> [M, W, L, N, d]."""
    idx = wf.starts[:, None] + np.arange(wf.length)[None, :]
    return X[:, idx]


def weak_targets(X: np.ndarray, wf: WeakForm) -> np.ndarray:
    """−Σ_l φ'_l x_l for every window: [M, W, N, d]."""
    return -np.einsum("l,mwlnd->mwnd", wf.dphi, window_states(X, wf))


def weak_project(F: np.ndarray, wf: WeakForm) -> np.ndarray:
    """Σ_l φ_l F_l for RHS values F [M, T, N, d] (or windowed [M, W, L, N, d])."""
    if F.ndim == 4:
        F = window_states(F, wf)
    return np.einsum("l,mwlnd->mwnd", wf.phi, F)
