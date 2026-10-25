"""Sympy post-processing of candidate expressions."""

from __future__ import annotations

import sympy as sp

from ..systems.base import XI, XJ, XK

VARS = {1: (XI,), 2: (XI, XJ), 3: (XI, XJ, XK)}


def rename_inputs(expr: sp.Expr, n_inputs: int) -> sp.Expr:
    """Map PySR's x0, x1, x2 to x_i, x_j, x_k."""
    subs = {sp.Symbol(f"x{q}"): VARS[n_inputs][q] for q in range(n_inputs)}
    subs.update({sp.Symbol(f"x{q}", real=True): VARS[n_inputs][q] for q in range(n_inputs)})
    return expr.xreplace(subs)


def round_floats(expr: sp.Expr, digits: int = 4, zero_tol: float = 1e-8) -> sp.Expr:
    """Round numeric constants to ``digits`` significant digits; drop |c| < zero_tol."""

    def _r(f):
        v = float(f)
        if abs(v) < zero_tol:
            return sp.Integer(0)
        return sp.Float(float(f"{v:.{digits}g}"))

    return expr.xreplace({f: _r(f) for f in expr.atoms(sp.Float)})


def symmetrize_h(expr: sp.Expr) -> sp.Expr:
    """Enforce h(x_i, x_j, x_k) = h(x_i, x_k, x_j) by averaging over the swap."""
    swapped = expr.xreplace({XJ: XK, XK: XJ})
    if sp.simplify(expr - swapped) == 0:
        return expr
    return sp.Rational(1, 2) * (expr + swapped)


def clean(expr: sp.Expr, kind: str) -> sp.Expr:
    e = sp.sympify(expr)
    if kind == "h":
        e = symmetrize_h(e)
    try:
        e = sp.simplify(e)
    except Exception:  # sympy can fail on odd expressions; keep the unsimplified form
        pass
    return round_floats(e)


def complexity(expr: sp.Expr) -> int:
    """Node count of the expression tree (comparable to PySR's complexity)."""
    return int(sum(1 for _ in sp.preorder_traversal(expr)))
