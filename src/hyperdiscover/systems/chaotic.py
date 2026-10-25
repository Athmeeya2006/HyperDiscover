"""S6 (stretch): coupled Rössler oscillators, d = 3, coupling through the x-variable.

    ẋ = −y − z + Σ_j A_ij σ1 (x_j − x_i) + Σ_{j<k} B_ijk σ2 (x_j x_k − x_i²)
    ẏ = x + a y
    ż = b + z (x − c)

Note: the triadic term σ2 (x_j x_k − x_i^2) in the x-equation is a placeholder choice for
this stretch goal; it is not taken from a specific paper. Sympy ground truth is given for the
x-component only (``sympy_exprs``); full d = 3 support in the learning pipeline is a stretch.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem


class RosslerHO(HigherOrderSystem):
    system_id = "S6"
    name = "coupled_rossler"
    state_dim = 3
    default_params = {"a": 0.2, "b": 0.2, "c": 5.7, "s1": 0.1, "s2": 0.05}
    default_dt = 0.05
    ic_box = (-5.0, 5.0)

    def f(self, x):
        p = self.params
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        return np.stack([-Y - Z, X + p["a"] * Y, p["b"] + Z * (X - p["c"])], axis=-1)

    def g(self, xi, xj):
        out = np.zeros_like(xi)
        out[..., 0] = self.params["s1"] * (xj[..., 0] - xi[..., 0])
        return out

    def h(self, xi, xj, xk):
        out = np.zeros_like(xi)
        out[..., 0] = self.params["s2"] * (xj[..., 0] * xk[..., 0] - xi[..., 0] ** 2)
        return out

    def sympy_exprs(self):
        p = self.params
        return {
            "f": sp.Symbol("(vector field; see docstring)"),
            "g": sp.Float(p["s1"]) * (XJ - XI),
            "h": sp.Float(p["s2"]) * (XJ * XK - XI**2),
        }
