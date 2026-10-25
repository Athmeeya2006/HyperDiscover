"""S5: mutualistic dynamics with a triadic term. SYNTHETIC VARIANT designed for this benchmark.

The pairwise part follows the standard mutualistic model (Gao, Barzel & Barabási 2016);
the triadic term γ x_i x_j x_k / (D + x_j x_k) is our own construction.

    dx_i/dt = B + x_i (1 − x_i/K)(x_i/C − 1) + Σ_j A_ij x_i x_j / (D + E x_i + H x_j)
              + Σ_{j<k} B_ijk γ x_i x_j x_k / (D + x_j x_k)
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem


class MutualisticHO(HigherOrderSystem):
    system_id = "S5"
    name = "mutualistic_triadic_synthetic"
    default_params = {
        "B": 0.1,
        "K": 5.0,
        "C": 1.0,
        "D": 5.0,
        "E": 0.9,
        "H": 0.1,
        "gamma": 1.0,
    }
    default_dt = 0.05
    ic_box = (0.0, 5.0)

    def f(self, xi):
        p = self.params
        return p["B"] + xi * (1.0 - xi / p["K"]) * (xi / p["C"] - 1.0)

    def g(self, xi, xj):
        p = self.params
        return xi * xj / (p["D"] + p["E"] * xi + p["H"] * xj)

    def h(self, xi, xj, xk):
        p = self.params
        return p["gamma"] * xi * xj * xk / (p["D"] + xj * xk)

    def sympy_exprs(self):
        p = {k: sp.Float(v) for k, v in self.params.items()}
        return {
            "f": p["B"] + XI * (1 - XI / p["K"]) * (XI / p["C"] - 1),
            "g": XI * XJ / (p["D"] + p["E"] * XI + p["H"] * XJ),
            "h": p["gamma"] * XI * XJ * XK / (p["D"] + XJ * XK),
        }

    def postprocess(self, x):
        return np.maximum(x, 0.0)
