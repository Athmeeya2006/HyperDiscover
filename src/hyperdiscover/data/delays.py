"""Delayed coupling (CLAUDE.md §18.12 G4b): simulate and build lagged message columns.

Model: dx_i/dt = f(x_i) + c_i + Σ_j A_ij g(x_i(t), x_j(t − τ_ij)) + Σ_{j<k} B_ijk h(x_i(t), x_j(t − τ_ijk), x_k(t − τ_ijk)),
with delays that are integer multiples of the saved interval dt. Integration: method of steps on a fine
grid (dt / substeps); within one RK4 substep the delayed arguments are frozen at the substep start (first
order in the delayed argument, error O(dt/substeps); documented limitation of this benchmark variant).

Inference with known f, g, h (train/windowed.py logic): candidate lags L = {0, 1, ..., max_lag} saved
steps; one column per (edge, lag) and (triple, lag); the non-negative BIC LASSO picks edge and lag jointly.
"""

from __future__ import annotations

import numpy as np


def simulate_delayed(
    system,
    pair_lag: np.ndarray,
    tri_lag: np.ndarray,
    x_hist: np.ndarray,
    n_steps: int,
    substeps: int = 10,
) -> np.ndarray:
    """Integrate with per-edge lags (saved steps).

    Args:
        system: HigherOrderSystem (its directed pairs ``_pairs`` and focal triples ``_tris`` define the
            coupling; ``pair_lag`` [E] / ``tri_lag`` [Ht] align with them).
        x_hist: initial history [M, L + 1, N, 1] on the SAVED grid (constant history is fine).
        n_steps: saved steps to produce after the history.
    Returns: states [M, n_steps + 1, N, 1] starting at the last history point.
    """
    dt = system.default_dt
    h = dt / substeps
    M, L1, N, _ = x_hist.shape
    # fine-grid history buffer
    fine = np.repeat(x_hist, substeps, axis=1)[:, : (L1 - 1) * substeps + 1]
    buf = [fine[:, q] for q in range(fine.shape[1])]
    pi, pj = system._pairs[:, 0], system._pairs[:, 1]
    ti, tj, tk = (system._tris[:, q] for q in range(3)) if len(system._tris) else (np.zeros(0, int),) * 3
    plag = np.asarray(pair_lag, int) * substeps
    tlag = np.asarray(tri_lag, int) * substeps
    post = getattr(system, "postprocess", None)

    def rhs(x, n):
        out = system.f(x) + system.c
        if len(pi):
            xj = np.stack([(x if lg == 0 else buf[n - lg])[:, j] for j, lg in zip(pj, plag)], 1)
            msg = system.g(x[:, pi], xj) * system.g_scale
            np.add.at(out, (slice(None), pi), msg)
        if len(ti):
            xj = np.stack([(x if lg == 0 else buf[n - lg])[:, j] for j, lg in zip(tj, tlag)], 1)
            xk = np.stack([(x if lg == 0 else buf[n - lg])[:, k] for k, lg in zip(tk, tlag)], 1)
            msg = system.h(x[:, ti], xj, xk) * system.h_scale
            np.add.at(out, (slice(None), ti), msg)
        return out

    saved = [buf[-1]]
    for _ in range(n_steps):
        for _ in range(substeps):
            n = len(buf) - 1
            x = buf[-1]
            k1 = rhs(x, n)
            k2 = rhs(x + 0.5 * h * k1, n)
            k3 = rhs(x + 0.5 * h * k2, n)
            k4 = rhs(x + h * k3, n)
            xn = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            buf.append(post(xn) if post is not None else xn)
        saved.append(buf[-1])
    return np.stack(saved, 1)


def lagged_design(model, X: np.ndarray, pairs, tris, max_lag: int):
    """Columns for every (candidate, lag). X [M, T, N, 1] -> base [n, N], G [n, E·(L+1)], H [n, Ht·(L+1)],
    and the (candidate index, lag) of every column. Rows: times t ≥ max_lag of every trajectory."""
    M, T, N, _ = X.shape
    cur = X[:, max_lag:]  # [M, T', N, 1]
    x = cur.reshape(-1, N, 1)
    base = (model.f(x) + model.c)[..., 0]
    Gs, Hs, gcol, hcol = [], [], [], []
    for lag in range(max_lag + 1):
        past = X[:, max_lag - lag : T - lag].reshape(-1, N, 1)
        if len(pairs):
            Gs.append(model.g(x[:, pairs[:, 0]], past[:, pairs[:, 1]])[..., 0])
            gcol += [(e, lag) for e in range(len(pairs))]
        if len(tris):
            Hs.append(model.h(x[:, tris[:, 0]], past[:, tris[:, 1]], past[:, tris[:, 2]])[..., 0])
            hcol += [(q, lag) for q in range(len(tris))]
    G = np.concatenate(Gs, 1) if Gs else np.zeros((len(x), 0))
    H = np.concatenate(Hs, 1) if Hs else np.zeros((len(x), 0))
    return base, G, H, np.array(gcol).reshape(-1, 2), np.array(hcol).reshape(-1, 2)
