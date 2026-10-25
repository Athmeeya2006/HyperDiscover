"""CLAUDE.md §12 test 2: derivative estimates converge as noise → 0; weak form exact for the true model."""

import numpy as np
import pytest

from hyperdiscover import benchmarks
from hyperdiscover.data.derivatives import estimate_derivatives
from hyperdiscover.data.weak_form import make_weak_form, weak_project, weak_targets


def _sin_data(noise, rng, T=400):
    t = np.linspace(0, 4 * np.pi, T)
    x = np.sin(t)[None, :, None, None]
    return x + noise * rng.standard_normal(x.shape), t, np.cos(t)[None, :, None, None]


@pytest.mark.parametrize(
    "method,kw",
    [
        ("fd", {}),
        ("savgol", {"window": 15, "polyorder": 3}),
        ("tikhonov", {"alpha": 1e-6}),
        ("tv", {"alpha": 1e-6}),
    ],
)
def test_sin_convergence(method, kw, rng):
    errs = []
    for noise in (1e-2, 1e-3, 1e-4, 0.0):
        X, t, dx = _sin_data(noise, np.random.default_rng(0))
        est = estimate_derivatives(X, t, method, **kw)
        errs.append(np.sqrt(np.mean((est - dx)[:, 10:-10] ** 2)))
    assert errs[-1] < 1e-3, errs
    assert errs[0] > errs[-1]
    assert all(a >= b * 0.9 for a, b in zip(errs[:-1], errs[1:])), errs  # (near-)monotone


def test_known_ode_convergence():
    b = benchmarks.load("S4", N=10, M=5, T=200, seed=0)
    errs = []
    for noise in (1e-2, 1e-3, 1e-4, 0.0):
        X = b.data.X_clean + noise * np.random.default_rng(1).standard_normal(b.data.X.shape)
        est = estimate_derivatives(X, b.data.t, "savgol", window=9, polyorder=3)
        errs.append(np.sqrt(np.mean((est - b.data.dX_clean)[:, 5:-5] ** 2)) / b.data.dX_clean.std())
    assert all(a > b for a, b in zip(errs[:-1], errs[1:])), errs
    assert errs[-1] < 2e-2, errs


def test_weak_form_zero_for_true_model():
    b = benchmarks.load("S2", N=10, M=5, T=200, seed=0)
    wf = make_weak_form(b.data.t, window=21, stride=4)
    lhs = weak_targets(b.data.X_clean, wf)
    rhs = weak_project(b.system.rhs(b.data.X_clean), wf)
    rel = np.sqrt(np.mean((lhs - rhs) ** 2)) / (np.sqrt(np.mean(lhs**2)) + 1e-12)
    assert rel < 1e-3, rel  # trapezoid quadrature error only


def test_weak_form_respects_segments():
    t = np.arange(100) * 0.1
    wf = make_weak_form(t, window=11, stride=3, segment_starts=[0, 50])
    for s in wf.starts:
        assert not (s < 50 < s + wf.length), s


def test_tv_handles_kinks_better_than_fd_under_noise():
    """|t|-shaped signal: the derivative is a step; TV should beat raw finite differences."""
    t = np.linspace(-1, 1, 201)
    x = np.abs(t)[None, :, None, None]
    rng = np.random.default_rng(0)
    X = x + 1e-3 * rng.standard_normal(x.shape)
    true = np.sign(t)[None, :, None, None]
    e_fd = np.sqrt(np.mean((estimate_derivatives(X, t, "fd") - true)[:, 5:-5] ** 2))
    e_tv = np.sqrt(np.mean((estimate_derivatives(X, t, "tv", alpha=1e-3) - true)[:, 5:-5] ** 2))
    assert e_tv < 0.5 * e_fd, (e_tv, e_fd)
