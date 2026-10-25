"""S3: nonlinear higher-order consensus (after Neuhäuser, Mellor & Lambiotte 2020 (verify)).

dx_i/dt = κ1 Σ_j A_ij (x_j − x_i) + κ2 Σ_{j<k} B_ijk exp(−λ |x_j − x_k|) ((x_j + x_k)/2 − x_i)
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem


class ConsensusHO(HigherOrderSystem):
    system_id = "S3"
    name = "nonlinear_higher_order_consensus"
    default_params = {"k1": 0.2, "k2": 0.4, "lam": 1.0}
    default_dt = 0.05
    ic_box = (-2.0, 2.0)

    def f(self, xi):
        return np.zeros_like(xi)

    def g(self, xi, xj):
        return self.params["k1"] * (xj - xi)

    def h(self, xi, xj, xk):
        p = self.params
        return p["k2"] * np.exp(-p["lam"] * np.abs(xj - xk)) * (0.5 * (xj + xk) - xi)

    def sympy_exprs(self):
        p = self.params
        return {
            "f": sp.Integer(0),
            "g": sp.Float(p["k1"]) * (XJ - XI),
            "h": sp.Float(p["k2"])
            * sp.exp(-sp.Float(p["lam"]) * sp.Abs(XJ - XK))
            * ((XJ + XK) / 2 - XI),
        }
