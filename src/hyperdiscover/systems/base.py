"""Base class for ground-truth higher-order network dynamical systems.

Every benchmark system has the form

    dx_i/dt = f(x_i) + c_i + sum_j A_ij g(x_i, x_j) + sum_{j<k} B_ijk h(x_i, x_j, x_k)

with f, g, h shared across nodes. Systems store the ground truth both as vectorised
numpy callables (for simulation) and as sympy expressions (for evaluation).

Structure conventions (used throughout the package):

* ``edges``: int array [E_u, 2] of *undirected* pairwise edges (a < b).
* ``hyperedges``: int array [H_u, 3] of *undirected* 3-node hyperedges (a < b < c).
* ``pair_index``: int array [E, 2] of *directed* (target i, source j) pairs, both directions
  of every undirected edge.
* ``tri_index``: int array [H, 3] of *focal* triples (i, j, k) with j < k: every node of
  every hyperedge receives one triadic message from the other two.

All state arrays have a trailing node and feature axis: ``x[..., N, d]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

# Canonical sympy symbols for message-function inputs (d = 1).
XI, XJ, XK = sp.symbols("x_i x_j x_k", real=True)


def directed_pairs(edges: np.ndarray) -> np.ndarray:
    """Expand undirected edges [E_u, 2] into directed (target, source) pairs [2 E_u, 2]."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.concatenate([edges, edges[:, ::-1]], axis=0)


def focal_triples(hyperedges: np.ndarray) -> np.ndarray:
    """Expand undirected hyperedges [H_u, 3] into focal triples (i, j, k), j < k, [3 H_u, 3]."""
    hyp = np.sort(np.asarray(hyperedges, dtype=np.int64).reshape(-1, 3), axis=1)
    a, b, c = hyp[:, 0], hyp[:, 1], hyp[:, 2]
    return np.concatenate(
        [np.stack([a, b, c], 1), np.stack([b, a, c], 1), np.stack([c, a, b], 1)], axis=0
    )


@dataclass
class Structure:
    """Ground-truth (or inferred) higher-order structure on N nodes."""

    n_nodes: int
    edges: np.ndarray  # [E_u, 2], a < b
    hyperedges: np.ndarray  # [H_u, 3], a < b < c
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        e = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        e = np.sort(e, axis=1)
        self.edges = np.unique(e, axis=0) if len(e) else e
        h = np.asarray(self.hyperedges, dtype=np.int64).reshape(-1, 3)
        h = np.sort(h, axis=1)
        self.hyperedges = np.unique(h, axis=0) if len(h) else h

    @property
    def pair_index(self) -> np.ndarray:
        return directed_pairs(self.edges)

    @property
    def tri_index(self) -> np.ndarray:
        return focal_triples(self.hyperedges)

    def adjacency(self) -> np.ndarray:
        """Dense symmetric adjacency [N, N] (only for small N)."""
        A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        if len(self.edges):
            A[self.edges[:, 0], self.edges[:, 1]] = 1
            A[self.edges[:, 1], self.edges[:, 0]] = 1
        return A

    def closure_class(self, triples: np.ndarray | None = None) -> np.ndarray:
        """Number of pairwise edges (0..3) among the nodes of each undirected triple.

        0 = open hyperedge, 3 = closed hyperedge, 1-2 = partially closed.
        """
        triples = self.hyperedges if triples is None else np.asarray(triples).reshape(-1, 3)
        edge_set = {(int(a), int(b)) for a, b in self.edges}
        out = np.zeros(len(triples), dtype=np.int64)
        for n, (a, b, c) in enumerate(np.sort(triples, axis=1)):
            out[n] = ((a, b) in edge_set) + ((a, c) in edge_set) + ((b, c) in edge_set)
        return out

    def pairwise_only(self) -> Structure:
        return Structure(
            self.n_nodes, self.edges.copy(), np.zeros((0, 3), np.int64), dict(self.meta)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": int(self.n_nodes),
            "edges": self.edges.tolist(),
            "hyperedges": self.hyperedges.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Structure:
        return cls(
            int(d["n_nodes"]), np.array(d["edges"]), np.array(d["hyperedges"]), d.get("meta", {})
        )


class HigherOrderSystem:
    """Abstract ground-truth system. Subclasses define f, g, h (numpy + sympy) and params.

    Subclasses must set ``system_id``, ``name``, ``state_dim`` and ``default_params`` and
    implement ``f``, ``g``, ``h`` (vectorised numpy, inputs/outputs ``[..., d]``) and
    ``sympy_exprs``. Optional per-node constants come from ``node_constants``.
    """

    system_id: str = "S0"
    name: str = "base"
    state_dim: int = 1
    periodic: bool = False  # True for phase variables (inputs are angles)
    default_params: dict[str, float] = {}
    default_dt: float = 0.05
    ic_box: tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        structure: Structure,
        params: dict[str, float] | None = None,
        node_constants: np.ndarray | None = None,
    ) -> None:
        self.structure = structure
        self.params = {**self.default_params, **(params or {})}
        self.N = structure.n_nodes
        self._pairs = structure.pair_index
        self._tris = structure.tri_index
        if node_constants is None:
            node_constants = np.zeros((self.N, self.state_dim))
        self.c = np.asarray(node_constants, dtype=np.float64).reshape(self.N, self.state_dim)
        # Scale factors on the couplings (used by bifurcation sweeps).
        self.g_scale = 1.0
        self.h_scale = 1.0

    # ---- message functions: inputs [..., d] -> [..., d] -----------------------------------
    def f(self, xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def g(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def h(self, xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sympy_exprs(self) -> dict[str, sp.Expr]:
        """Ground truth {"f": f(x_i), "g": g(x_i, x_j), "h": h(x_i, x_j, x_k)} for d = 1."""
        raise NotImplementedError

    @property
    def has_node_constants(self) -> bool:
        return bool(np.any(self.c != 0))

    # ---- full right-hand side ---------------------------------------------------------------
    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Vector field. x: [..., N, d] -> dx/dt: [..., N, d]."""
        out = self.f(x) + self.c
        if len(self._pairs) and self.g_scale != 0:
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            msg = self.g(x[..., i, :], x[..., j, :]) * self.g_scale  # [..., E, d]
            _scatter_add(out, i, msg)
        if len(self._tris) and self.h_scale != 0:
            i, j, k = self._tris[:, 0], self._tris[:, 1], self._tris[:, 2]
            msg = self.h(x[..., i, :], x[..., j, :], x[..., k, :]) * self.h_scale
            _scatter_add(out, i, msg)
        return out

    def coupling_parts(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Separate contributions {"self", "pair", "tri"}, each [..., N, d]."""
        parts = {"self": self.f(x) + self.c, "pair": np.zeros_like(x), "tri": np.zeros_like(x)}
        if len(self._pairs):
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            msg = self.g(x[..., i, :], x[..., j, :]) * self.g_scale  # same scaling as rhs
            _scatter_add(parts["pair"], i, msg)
        if len(self._tris):
            i, j, k = self._tris[:, 0], self._tris[:, 1], self._tris[:, 2]
            msg = self.h(x[..., i, :], x[..., j, :], x[..., k, :]) * self.h_scale
            _scatter_add(parts["tri"], i, msg)
        return parts

    def sample_ic(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Random initial conditions [n, N, d] uniform in ``ic_box``."""
        lo, hi = self.ic_box
        return rng.uniform(lo, hi, size=(n, self.N, self.state_dim))

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        """Hook applied after each integration step (e.g. clipping to a valid domain)."""
        return x

    def ground_truth(self) -> dict[str, Any]:
        exprs = self.sympy_exprs()
        return {
            "system_id": self.system_id,
            "name": self.name,
            "params": {k: float(v) for k, v in self.params.items()},
            "equations": {k: str(v) for k, v in exprs.items()},
            "latex": {k: sp.latex(v) for k, v in exprs.items()},
            "node_constants": self.c.tolist(),
            "structure": self.structure.to_dict(),
            "periodic": self.periodic,
        }

    def pairwise_only(self) -> HigherOrderSystem:
        """Same system with B = 0."""
        return type(self)(self.structure.pairwise_only(), self.params, self.c)


def _scatter_add(out: np.ndarray, index: np.ndarray, msg: np.ndarray) -> None:
    """out[..., index[e], :] += msg[..., e, :] (numpy, handles repeated indices)."""
    # move node axis first for np.add.at
    o = np.moveaxis(out, -2, 0)
    m = np.moveaxis(msg, -2, 0)
    np.add.at(o, index, m)
