"""S4: higher-order SIS, quenched mean field (simplicial contagion, Iacopini et al. 2019).

dx_i/dt = −μ x_i + β (1 − x_i) Σ_j A_ij x_j + β_Δ (1 − x_i) Σ_{j<k} B_ijk x_j x_k
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem


class SISHO(HigherOrderSystem):
    system_id = "S4"
    name = "higher_order_sis"
    default_params = {"mu": 1.0, "beta": 0.4, "beta_tri": 1.2}
    default_dt = 0.05
    ic_box = (0.0, 1.0)

    def f(self, xi):
        return -self.params["mu"] * xi

    def g(self, xi, xj):
        return self.params["beta"] * (1.0 - xi) * xj

    def h(self, xi, xj, xk):
        return self.params["beta_tri"] * (1.0 - xi) * xj * xk

    def sympy_exprs(self):
        p = self.params
        return {
            "f": -sp.Float(p["mu"]) * XI,
            "g": sp.Float(p["beta"]) * (1 - XI) * XJ,
            "h": sp.Float(p["beta_tri"]) * (1 - XI) * XJ * XK,
        }

    def postprocess(self, x):
        return np.clip(x, 0.0, 1.0)
