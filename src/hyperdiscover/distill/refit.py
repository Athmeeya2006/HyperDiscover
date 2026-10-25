"""Stage 5: refit all numeric constants of discovered f, g, h on the original data (float64, CPU).

Every Float in the expressions becomes a free parameter, initialised at its distilled value;
per-node constants c_i (optional) are extra linear parameters. The objective is the
derivative-target MSE (or the weak-form residual) summed over the *inferred* structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from ..data.weak_form import WeakForm, weak_project, weak_targets
from ..systems.base import XI, XJ, XK, HigherOrderSystem, Structure


def parametrize(
    exprs: dict[str, sp.Expr],
) -> tuple[dict[str, sp.Expr], list[sp.Symbol], np.ndarray]:
    """Replace Floats by parameter symbols θ_q. Returns (templates, symbols, init values).

    f, g: every Float OCCURRENCE gets its own parameter (two constants that merely happen to be
    equal must not be tied). h: equal Floats share one parameter, which keeps the (j, k)-mirrored
    pairs created by ``symmetrize_h`` tied, so the refit h stays symmetric.
    """
    syms, vals, templates = [], [], {}

    def new(fl):
        th = sp.Symbol(f"theta_{len(syms)}")
        syms.append(th)
        vals.append(float(fl))
        return th

    def untie(e):
        if isinstance(e, sp.Float):
            return new(e)
        if not e.args:
            return e
        return e.func(*[untie(a) for a in e.args])

    for kind in ("f", "g", "h"):
        e = sp.sympify(exprs.get(kind, 0))
        if kind == "h":
            rep = {fl: new(fl) for fl in sorted(e.atoms(sp.Float), key=lambda a: float(a))}
            templates[kind] = e.xreplace(rep)
        else:
            templates[kind] = untie(e)
    return templates, syms, np.asarray(vals, dtype=np.float64)


def _lambdify(templates, syms):
    args = {"f": [XI], "g": [XI, XJ], "h": [XI, XJ, XK]}
    return {k: sp.lambdify(args[k] + list(syms), templates[k], "numpy") for k in templates}


class SymbolicModel(HigherOrderSystem):
    """A discovered model as a simulatable system (d = 1)."""

    system_id = "discovered"
    name = "discovered"

    def __init__(
        self, structure: Structure, exprs: dict[str, sp.Expr], node_constants=None, periodic=False
    ):
        self.exprs = {k: sp.sympify(exprs.get(k, 0)) for k in ("f", "g", "h")}
        self.periodic = periodic
        super().__init__(structure, {}, node_constants)
        self._fns = {
            "f": sp.lambdify([XI], self.exprs["f"], "numpy"),
            "g": sp.lambdify([XI, XJ], self.exprs["g"], "numpy"),
            "h": sp.lambdify([XI, XJ, XK], self.exprs["h"], "numpy"),
        }

    def _call(self, k, *xs):
        out = np.asarray(self._fns[k](*[x[..., 0] for x in xs]), dtype=np.float64)
        return np.broadcast_to(out, xs[0].shape[:-1]).copy()[..., None]

    def f(self, xi):
        return self._call("f", xi)

    def g(self, xi, xj):
        return self._call("g", xi, xj)

    def h(self, xi, xj, xk):
        return self._call("h", xi, xj, xk)

    def sympy_exprs(self):
        return dict(self.exprs)

    def pairwise_only(self):
        return SymbolicModel(self.structure.pairwise_only(), self.exprs, self.c, self.periodic)


@dataclass
class RefitResult:
    exprs: dict[str, sp.Expr]
    node_constants: np.ndarray | None
    mse: float
    n_params: int
    n_samples: int
    success: bool
    info: dict = field(default_factory=dict)

    def bic(self) -> float:
        return float(
            self.n_samples * np.log(max(self.mse, 1e-300)) + self.n_params * np.log(self.n_samples)
        )

    def aic(self) -> float:
        return float(self.n_samples * np.log(max(self.mse, 1e-300)) + 2 * self.n_params)


def refit(
    exprs: dict[str, sp.Expr],
    structure: Structure,
    X: np.ndarray,
    dX: np.ndarray | None = None,
    node_constants: bool = False,
    weak: WeakForm | None = None,
    max_samples: int = 3000,
    rng: np.random.Generator | None = None,
    max_nfev: int = 200,
) -> RefitResult:
    """Least-squares refit of all constants.

    Args:
        X: states [M, T, N, 1]; dX: derivative targets (same shape) for the derivative loss.
        weak: if given, fit weak-form targets instead (dX ignored).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    templates, syms, theta0 = parametrize(exprs)
    fns = _lambdify(templates, syms)
    pairs, tris = structure.pair_index, structure.tri_index
    M, T, N, d = X.shape

    if weak is None:
        S = X.reshape(-1, N, d)
        Y = dX.reshape(-1, N, d)
        if len(S) > max_samples:
            sel = rng.choice(len(S), max_samples, replace=False)
            S, Y = S[sel], Y[sel]
    else:
        S, Y = X, weak_targets(X, weak)

    def rhs(theta, states):
        x = states[..., 0]
        out = np.zeros_like(x)
        out += np.broadcast_to(fns["f"](x, *theta), x.shape)
        if len(pairs) and templates["g"] != 0:
            m = np.broadcast_to(
                fns["g"](x[..., pairs[:, 0]], x[..., pairs[:, 1]], *theta),
                x[..., pairs[:, 0]].shape,
            )
            o = np.moveaxis(out, -1, 0)
            np.add.at(o, pairs[:, 0], np.moveaxis(m, -1, 0))
        if len(tris) and templates["h"] != 0:
            xi, xj, xk = (x[..., tris[:, q]] for q in range(3))
            m = np.broadcast_to(fns["h"](xi, xj, xk, *theta), xi.shape)
            o = np.moveaxis(out, -1, 0)
            np.add.at(o, tris[:, 0], np.moveaxis(m, -1, 0))
        return out[..., None]

    n_th = len(theta0)
    scale = float(Y.std()) or 1.0

    def resid(p):
        theta, c = p[:n_th], (p[n_th:].reshape(N, 1) if node_constants else 0.0)
        with np.errstate(all="ignore"):
            if weak is None:
                pred = rhs(theta, S) + c
            else:
                pred = weak_project(rhs(theta, S) + c, weak)
        r = ((pred - Y) / scale).ravel()
        return np.where(np.isfinite(r), r, 1e6)

    p0 = np.concatenate([theta0, np.zeros(N if node_constants else 0)])
    if len(p0) == 0:
        r = resid(p0)
        return RefitResult(dict(exprs), None, float(np.mean(r**2) * scale**2), 0, r.size, True)
    sol = least_squares(resid, p0, method="trf", max_nfev=max_nfev, x_scale="jac")
    theta = sol.x[:n_th]
    sub = dict(zip(syms, [sp.Float(float(v)) for v in theta]))
    new = {k: sp.sympify(templates[k]).xreplace(sub) for k in templates}
    c = sol.x[n_th:].reshape(N, 1) if node_constants else None
    mse = float(np.mean(sol.fun**2) * scale**2)
    return RefitResult(
        new,
        c,
        mse,
        len(sol.x),
        sol.fun.size,
        bool(sol.success),
        {"nfev": int(sol.nfev), "cost": float(sol.cost)},
    )


__all__ = ["parametrize", "refit", "RefitResult", "SymbolicModel"]
