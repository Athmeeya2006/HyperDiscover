"""S2: higher-order (competitive) Lotka–Volterra.

    dx_i/dt = x_i (r − x_i) + a Σ_j A_ij x_i x_j + b Σ_{j<k} B_ijk x_i x_j x_k

Defaults use competition (a, b < 0) so that trajectories stay bounded for x ≥ 0.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem


class LotkaVolterraHO(HigherOrderSystem):
    system_id = "S2"
    name = "higher_order_lotka_volterra"
    default_params = {"r": 1.0, "a": -0.15, "b": -0.3}
    default_dt = 0.05
    ic_box = (0.05, 1.5)

    def f(self, xi):
        return xi * (self.params["r"] - xi)

    def g(self, xi, xj):
        return self.params["a"] * xi * xj

    def h(self, xi, xj, xk):
        return self.params["b"] * xi * xj * xk

    def sympy_exprs(self):
        p = self.params
        return {
            "f": XI * (sp.Float(p["r"]) - XI),
            "g": sp.Float(p["a"]) * XI * XJ,
            "h": sp.Float(p["b"]) * XI * XJ * XK,
        }

    def postprocess(self, x):
        return np.maximum(x, 0.0)
