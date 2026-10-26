# SPDX-License-Identifier: GPL-3.0-or-later
# This file is a port of the official THIS code (GPL-3.0, LICENSE file at tag v2.0 of
# github.com/TaylorBasedHypergraphInference/THIS) and is therefore licensed under GPL-3.0, NOT under the
# MIT licence of the rest of HyperDiscover. It is not imported by the hyperdiscover package; only the
# baseline scripts import it explicitly (DECISIONS.md D-040).
"""Baseline 2: THIS, Taylor-based Hypergraph Inference using SINDy (Delabays, De Pasquale,
Dörfler & Zhang, Nature Communications 16, 2025; doi:10.1038/s41467-025-57664-2).

Official code: https://github.com/TaylorBasedHypergraphInference/THIS (v2.0, Julia; LICENSE file GPL-3.0 —
the Zenodo record doi:10.5281/zenodo.10530470 lists CC-BY-4.0 in its metadata; we follow the stricter one). This module is a line-by-line Python port of its core
functions ``get_θd``, ``mySINDy`` and ``this`` (file ``this.jl``), validated against the
official Julia code in ``tests/test_this_port.py`` / ``scripts/validate_this.py``
(DECISIONS.md D-022).

Algorithm (as in the official code):
  * library θ: all monomials of the (shifted) states x − x0 up to degree ``dmax`` (constant
    included), enumerated in the same order as ``get_θd``;
  * SINDy (``mySINDy``): Tikhonov-regularised least squares (ρ), then repeated hard
    thresholding of |Ξ| < λ on the *raw* coefficients and refitting, at most ``niter`` times;
  * readout: a monomial whose variables are all distinct, with support m, yields an order-|m|+1
    interaction (i, m) for every node i ∉ m with |Ξ_{i,m}| > 1e-8. Monomials containing i itself
    or a repeated variable are in the library but are never read as edges.
Scores for ROC curves are |Ξ_{i,m}| (the paper sweeps an edge threshold ε over them).
"""

from __future__ import annotations

import time

import numpy as np


def get_theta_d(X: np.ndarray, dmax: int, i0: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``get_θd``. X: [n, T] (rows = agents). Returns θ [m, T] and d [m, dmax]
    (1-based agent indices of each monomial's factors, 0 = unused slot)."""
    n, T = X.shape
    theta = np.ones((1, T))
    d = np.zeros((1, dmax), dtype=np.int64)
    if dmax == 0:
        return theta, d
    for i in range(1, n + 1):
        th0, d0 = get_theta_d(X[i - 1 :, :], dmax - 1, i)
        theta = np.vstack([theta, X[i - 1 : i, :] * th0])
        d = np.vstack([d, np.hstack([d0, np.full((d0.shape[0], 1), i, dtype=np.int64)])])
    d = d + (i0 - 1) * (d > 0)
    return theta, d


def my_sindy(theta: np.ndarray, Y: np.ndarray, lam: float = 0.1, rho: float = 1.0, niter: int = 10):
    """Port of ``mySINDy``. theta [m, T], Y [n, T] -> (Ξ [n, m], 1-norm error)."""
    n, _ = Y.shape
    m = theta.shape[0]
    Xi = Y @ theta.T @ np.linalg.inv(theta @ theta.T + rho * np.eye(m))
    nz = 10**6
    k = 1
    while k < niter and int(np.sum(np.abs(Xi) > 1e-6)) != nz:
        k += 1
        nz = int(np.sum(np.abs(Xi) > 1e-6))
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for i in range(n):
            big = ~small[i]
            tb = theta[big]
            Xi[i, big] = (
                Y[i : i + 1] @ tb.T @ np.linalg.inv(tb @ tb.T + rho * np.eye(int(big.sum())))
            )
    err = float(np.abs(Y - Xi @ theta).sum())
    return Xi, err


def this_official(
    X: np.ndarray, Y: np.ndarray, dmax: int = 2, lam: float = 0.1, rho: float = 1.0, niter: int = 10
):
    """Port of ``this``. X, Y: [n, T] states and derivatives (rows = agents).

    Returns (Ainf, Xi, relerr) with Ainf = {order: array of rows [i, j(, k), coeff]} using
    1-based node indices, exactly as the Julia function.
    """
    n, T = X.shape
    theta, d = get_theta_d(X, dmax)
    idx_mon = {}
    for r in range(d.shape[0]):
        mon = d[r][d[r] != 0]
        if len(mon) == len(set(mon.tolist())):
            idx_mon[r] = sorted(mon.tolist())
    Xi, err = my_sindy(theta, Y, lam, rho, niter)
    relerr = err / float(np.abs(Y).sum())
    Ainf: dict[int, list] = {o: [] for o in range(1, dmax + 2)}
    for r, mon in idx_mon.items():
        nodes = [i for i in range(1, n + 1) if abs(Xi[i - 1, r]) > 1e-8 and i not in mon]
        o = len(mon) + 1
        for i in nodes:
            Ainf[o].append([i, *mon, Xi[i - 1, r]])
    return {o: np.array(v).reshape(-1, o + 1) for o, v in Ainf.items()}, Xi, relerr


def this_reconstruct(
    X: np.ndarray,
    dX: np.ndarray,
    dmax: int = 2,
    lam: float = 0.1,
    rho: float = 1.0,
    niter: int = 10,
    x0: float | np.ndarray | None = 0.0,
    max_samples: int = 5000,
    max_cols: int = 50_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """THIS on our data layout. X, dX: [M, T, N, 1]. Scores: |Ξ| (0-based node indices).

    ``x0`` is the Taylor base point (the paper samples a box around x0 = 0 for Kuramoto);
    ``x0=None`` uses the data mean. ``dmax = 2`` gives pairwise + triadic interactions.
    """
    from math import comb

    rng = rng if rng is not None else np.random.default_rng(0)
    N = X.shape[2]
    n_cols = comb(N + dmax, dmax)
    if n_cols > max_cols:
        raise MemoryError(f"THIS library would have {n_cols} monomials (N={N}, dmax={dmax})")
    Xs, Ys = X[..., 0].reshape(-1, N), dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    base = Xs.mean(0) if x0 is None else np.broadcast_to(np.asarray(x0, float), (N,))
    t0 = time.time()
    Ainf, Xi, relerr = this_official((Xs - base).T, Ys.T, dmax, lam, rho, niter)
    pair, tri = {}, {}
    for row in Ainf.get(2, []):
        i, j, c = int(row[0]) - 1, int(row[1]) - 1, abs(row[2])
        pair[(i, j)] = max(pair.get((i, j), 0.0), float(c))
    for row in Ainf.get(3, []):
        i, j, k, c = int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1, abs(row[3])
        key = (i, min(j, k), max(j, k))
        tri[key] = max(tri.get(key, 0.0), float(c))
    return {
        "pair_scores": pair,
        "tri_scores": tri,
        "n_cols": int(n_cols),
        "relerr": float(relerr),
        "seconds": time.time() - t0,
        "label": "THIS (Python port of official v2.0)",
    }
