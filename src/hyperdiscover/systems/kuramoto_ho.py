"""S1: higher-order Kuramoto model on unwrapped phases (Skardal & Arenas style).

dθ_i/dt = ω_i + K1 Σ_j A_ij sin(θ_j − θ_i) + K2 Σ_{j<k} B_ijk sin(θ_j + θ_k − 2θ_i)
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .base import XI, XJ, XK, HigherOrderSystem, Structure


class KuramotoHO(HigherOrderSystem):
    system_id = "S1"
    name = "higher_order_kuramoto"
    periodic = True
    default_params = {"K1": 0.5, "K2": 0.5, "omega_scale": 1.0}
    default_dt = 0.05
    ic_box = (0.0, 2.0 * np.pi)

    def __init__(self, structure: Structure, params=None, node_constants=None, rng=None):
        params = {**self.default_params, **(params or {})}
        if node_constants is None:
            rng = rng if rng is not None else np.random.default_rng(0)
            node_constants = (
                rng.uniform(-1.0, 1.0, size=(structure.n_nodes, 1)) * params["omega_scale"]
            )
        super().__init__(structure, params, node_constants)

    def f(self, xi):
        return np.zeros_like(xi)

    def g(self, xi, xj):
        return self.params["K1"] * np.sin(xj - xi)

    def h(self, xi, xj, xk):
        return self.params["K2"] * np.sin(xj + xk - 2.0 * xi)

    def sympy_exprs(self):
        p = self.params
        return {
            "f": sp.Integer(0),
            "g": sp.Float(p["K1"]) * sp.sin(XJ - XI),
            "h": sp.Float(p["K2"]) * sp.sin(XJ + XK - 2 * XI),
        }

    def pairwise_only(self):
        return type(self)(self.structure.pairwise_only(), self.params, self.c)
