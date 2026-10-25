"""CLAUDE.md §12 test 5: symbolic regression recovers sin(x_j − x_i) and x_i x_j x_k.

The PySR tests need Julia (marked `pysr`; skipped when PySR is not importable). The library
fallback is tested unconditionally.
"""

import numpy as np
import pytest
import sympy as sp

from hyperdiscover.distill import pysr_available, run_library, run_pysr
from hyperdiscover.distill.refit import parametrize, refit
from hyperdiscover.distill.simplify import symmetrize_h
from hyperdiscover.eval.equation_metrics import parametric_match
from hyperdiscover.systems import XI, XJ, XK


def _data(rng, n_in, fn, lo=-3, hi=3, n=500):
    X = rng.uniform(lo, hi, (n, n_in))
    return X, fn(X)[:, None]


@pytest.mark.parametrize(
    "target,kind,n_in,fn,lo,hi",
    [
        (sp.sin(XJ - XI), "g", 2, lambda X: np.sin(X[:, 1] - X[:, 0]), -3, 3),
        (XI * XJ * XK, "h", 3, lambda X: X[:, 0] * X[:, 1] * X[:, 2], 0, 2),
    ],
)
def test_library_recovers(target, kind, n_in, fn, lo, hi, rng):
    X, y = _data(rng, n_in, fn, lo, hi)
    res = run_library(X, y, kind)
    best = min(res.candidates, key=lambda c: (c.val_loss > 1e-10, c.complexity))
    assert parametric_match(best.expr, target, kind, X)["exact"], best.expr


@pytest.mark.pysr
@pytest.mark.skipif(not pysr_available(), reason="PySR/Julia not installed")
@pytest.mark.parametrize(
    "target,kind,n_in,fn,lo,hi",
    [
        (sp.sin(XJ - XI), "g", 2, lambda X: np.sin(X[:, 1] - X[:, 0]), -3, 3),
        (XI * XJ * XK, "h", 3, lambda X: X[:, 0] * X[:, 1] * X[:, 2], 0, 2),
    ],
)
def test_pysr_recovers(target, kind, n_in, fn, lo, hi, rng, tmp_path):
    X, y = _data(rng, n_in, fn, lo, hi)
    opts = {
        "niterations": 60,
        "timeout_in_seconds": 90,
        "restarts": 1,
        "maxsize": 12,
        "unary_operators": ["sin", "cos"],
        "binary_operators": ["+", "-", "*"],
    }
    res = run_pysr(X, y, kind, opts, seed=0, out_dir=tmp_path)
    ok = [c for c in res.candidates if parametric_match(c.expr, target, kind, X)["exact"]]
    assert ok, [str(c.expr) for c in res.pareto()]


def test_symmetrize():
    s = symmetrize_h(XI * XJ**2 * XK)
    assert sp.simplify(s - s.xreplace({XJ: XK, XK: XJ})) == 0  # xreplace swaps simultaneously
    sym = XI * XJ * XK
    assert symmetrize_h(sym) == sym


def test_parametric_match_tolerances(rng):
    X = rng.uniform(0, 1, (300, 2))
    truth = sp.Float(0.4) * (1 - XI) * XJ
    assert parametric_match(sp.Float(0.41) * XJ - sp.Float(0.41) * XI * XJ, truth, "g", X)["exact"]
    assert not parametric_match(sp.Float(0.5) * XJ - sp.Float(0.5) * XI * XJ, truth, "g", X)[
        "exact"
    ]  # 25% off
    assert not parametric_match(sp.Float(0.4) * XJ, truth, "g", X)["exact"]  # missing term


def test_parametrize_roundtrip():
    tmpl, syms, vals = parametrize({"f": sp.Float(-1.5) * XI, "g": sp.Float(0.3) * XI * XJ})
    assert len(syms) == 2 and sorted(vals.tolist()) == [-1.5, 0.3]


def test_refit_recovers_coefficients():
    from hyperdiscover import benchmarks

    b = benchmarks.load("S4", N=10, M=10, T=100, n_hyper=2, seed=0)
    guess = {
        "f": sp.Float(-0.8) * XI,
        "g": sp.Float(0.3) * (1 - XI) * XJ,
        "h": sp.Float(1.0) * (1 - XI) * XJ * XK,
    }
    r = refit(guess, b.structure, b.data.X_clean, b.data.dX_clean)
    X = np.random.default_rng(0).uniform(0, 1, (300, 3))
    truth = b.system.sympy_exprs()
    for k, n in (("f", 1), ("g", 2), ("h", 3)):
        assert parametric_match(r.exprs[k], truth[k], k, X[:, :n])["exact"], (k, r.exprs[k])
