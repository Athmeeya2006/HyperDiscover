"""Baseline 1: per-node SINDy over all pairwise and triadic candidate terms.

* ``library="oracle"``: terms built from the *true* functional families (the ground-truth
  additive terms of f, g, h with coefficients stripped; inner constants kept);
* ``library="generic"``: polynomial (≤ cubic total degree in (x_i, x_j, x_k)) or trig library.

Cost per node grows as O(N²) columns, so it stops scaling quickly; the runner records the
column count and wall-clock so the paper can report where it fails (CLAUDE.md §8.1).
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import sympy as sp

from ..systems.base import XI, XJ, XK
from .common import scores_from_blocks, stlsq


def basis_terms(expr: sp.Expr) -> list[sp.Expr]:
    """Additive terms of ``expr`` with leading numeric coefficients removed."""
    expr = sp.expand(sp.sympify(expr))
    if expr == 0:
        return []
    out = []
    for t in sp.Add.make_args(expr):
        _, rest = t.as_coeff_Mul()
        if rest != 1 and rest not in out:
            out.append(rest)
    return out


def term_coefficients(expr: sp.Expr) -> dict[sp.Expr, float]:
    """{basis term: numeric coefficient} of the expanded expression (inverse of ``basis_terms``)."""
    expr = sp.expand(sp.sympify(expr))
    out: dict[sp.Expr, float] = {}
    if expr == 0:
        return out
    for t in sp.Add.make_args(expr):
        c, rest = t.as_coeff_Mul()
        if rest != 1:
            out[rest] = out.get(rest, 0.0) + float(c)
    return out


def generic_terms(periodic: bool) -> dict[str, list[sp.Expr]]:
    if periodic:
        return {
            "f": [sp.sin(XI), sp.cos(XI)],
            "g": [sp.sin(XJ - XI), sp.cos(XJ - XI)],
            "h": [sp.sin(XJ + XK - 2 * XI), sp.cos(XJ + XK - 2 * XI)],
        }
    return {
        "f": [XI, XI**2, XI**3],
        "g": [XJ, XI * XJ, XJ**2, XI**2 * XJ, XI * XJ**2],
        "h": [XJ * XK, XI * XJ * XK],
    }


def _lamb(terms, args):
    return [sp.lambdify(args, t, "numpy") for t in terms]


def build_node_library(
    X: np.ndarray, i: int, terms: dict, node_constant: bool = True, max_cols: int = 200_000
):
    """X [S, N] -> (Theta [S, C], block keys per column: None | ('p', i, j) | ('t', i, j, k))."""
    S, N = X.shape
    others = [v for v in range(N) if v != i]
    n_cols = (
        len(terms["f"])
        + len(others) * len(terms["g"])
        + (len(others) * (len(others) - 1) // 2) * len(terms["h"])
    )
    if n_cols > max_cols:
        raise MemoryError(f"SINDy library would have {n_cols} columns for node {i}")
    cols, keys = [], []
    xi = X[:, i]
    if node_constant:
        cols.append(np.ones(S))
        keys.append(None)  # noqa: E702
    for fn in _lamb(terms["f"], [XI]):
        cols.append(np.broadcast_to(fn(xi), (S,)))
        keys.append(None)  # noqa: E702
    gf = _lamb(terms["g"], [XI, XJ])
    for j in others:
        for fn in gf:
            cols.append(np.broadcast_to(fn(xi, X[:, j]), (S,)))
            keys.append(("p", i, j))  # noqa: E702
    hf = _lamb(terms["h"], [XI, XJ, XK])
    for j, k in combinations(others, 2):
        for fn in hf:
            cols.append(np.broadcast_to(fn(xi, X[:, j], X[:, k]), (S,)))
            keys.append(("t", i, j, k))  # noqa: E702
    return np.stack(cols, 1), keys


def sindy_full(
    X: np.ndarray,
    dX: np.ndarray,
    library: str = "generic",
    truth_exprs: dict | None = None,
    periodic: bool = False,
    threshold: float = 0.05,
    max_samples: int = 5000,
    node_constant: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run per-node SINDy. X, dX: [M, T, N, 1]. Returns pair/triple scores and coefficients."""
    rng = rng if rng is not None else np.random.default_rng(0)
    if library == "oracle":
        terms = {k: basis_terms(truth_exprs[k]) for k in ("f", "g", "h")}
    else:
        terms = generic_terms(periodic)
    N = X.shape[2]
    Xs, Ys = X[..., 0].reshape(-1, N), dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    t0 = time.time()
    pair_scores, tri_scores, coefs = {}, {}, {}
    n_cols = 0
    for i in range(N):
        Theta, keys = build_node_library(Xs, i, terms, node_constant)
        n_cols = Theta.shape[1]
        w = stlsq(Theta, Ys[:, i], threshold)
        sc = scores_from_blocks(w, Theta, keys)
        for key, s in sc.items():
            if key[0] == "p":
                pair_scores[(key[1], key[2])] = s
            else:
                tri_scores[(key[1], key[2], key[3])] = s
        coefs[i] = {"w": w.tolist(), "keys": [None if k is None else list(k) for k in keys]}
    return {
        "pair_scores": pair_scores,
        "tri_scores": tri_scores,
        "coefs": coefs,
        "terms": {k: [str(t) for t in v] for k, v in terms.items()},
        "n_cols_per_node": n_cols,
        "seconds": time.time() - t0,
        "library": library,
    }


def directed_to_undirected(pair_scores: dict, tri_scores: dict, binary: bool = False):
    """Max over directions/focal nodes; with ``binary`` scores become 0/1 (nonzero coefficient)."""
    pu, tu = {}, {}
    for (i, j), s in pair_scores.items():
        k = (min(i, j), max(i, j))
        pu[k] = max(pu.get(k, 0.0), float(s > 0) if binary else s)
    for (i, j, k), s in tri_scores.items():
        key = tuple(sorted((i, j, k)))
        tu[key] = max(tu.get(key, 0.0), float(s > 0) if binary else s)
    return pu, tu


def oracle_coefficient_report(
    result: dict, truth_exprs: dict, structure, node_constant: bool = True, tol: float = 0.05
) -> dict:
    """Phase-2 gate check for ``library="oracle"``: exact structure AND every true coefficient
    within ``tol`` relative error (CLAUDE.md §13).

    Column layout per node (see ``build_node_library``): [1] + f-terms + (g-terms per source j)
    + (h-terms per pair (j, k)). True columns are the f-terms of every node, the g-terms of true
    directed pairs and the h-terms of true focal triples; every other non-constant column should be
    exactly zero.
    """
    coef = {k: term_coefficients(truth_exprs[k]) for k in ("f", "g", "h")}
    terms = {k: basis_terms(truth_exprs[k]) for k in ("f", "g", "h")}
    true_pairs = {tuple(map(int, p)) for p in structure.pair_index}
    true_tris = {tuple(map(int, t)) for t in structure.tri_index}
    rel_errs, n_spurious, n_missing = [], 0, 0
    for i, rec in result["coefs"].items():
        i = int(i)
        w = np.asarray(rec["w"])
        keys = rec["keys"]
        col = 1 if node_constant else 0
        for q, tm in enumerate(terms["f"]):
            rel_errs.append(abs(w[col + q] - coef["f"][tm]) / max(abs(coef["f"][tm]), 1e-12))
        col += len(terms["f"])
        for c in range(col, len(w)):
            key = keys[c]
            kind = "g" if key[0] == "p" else "h"
            node_key = tuple(key[1:])
            # position of this column within its block of terms
            q = (c - col) % len(terms["g"]) if kind == "g" else None
            if kind == "h":
                n_g_cols = (structure.n_nodes - 1) * len(terms["g"])
                q = (c - col - n_g_cols) % len(terms["h"])
            is_true = node_key in (true_pairs if kind == "g" else true_tris)
            if is_true:
                tc = coef[kind][terms[kind][q]]
                if w[c] == 0:
                    n_missing += 1
                rel_errs.append(abs(w[c] - tc) / max(abs(tc), 1e-12))
            elif w[c] != 0:
                n_spurious += 1
    max_err = float(max(rel_errs)) if rel_errs else 0.0
    return {
        "max_coef_rel_err": max_err,
        "n_spurious_terms": n_spurious,
        "n_missing_terms": n_missing,
        "exact": bool(n_spurious == 0 and n_missing == 0 and max_err <= tol),
    }
