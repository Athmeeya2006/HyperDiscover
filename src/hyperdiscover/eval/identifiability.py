"""A-priori identifiability of triadic terms: how much of h is *pure interaction* on the data.

For a focal triple (i, j, k) the triadic contribution h(x_i, x_j, x_k) can be partly mimicked by
two pairwise couplings u(x_i, x_j) + u(x_i, x_k) with one shared function u (pairwise couplings
are shared across edges in our model class). We fit the best such symmetric additive surrogate
on the empirical data distribution of the true hyperedges by backfitting with a flexible
regressor, and report

    pure_fraction = Var(h − û_ij − û_ik) / Var(h)          (0 = fully mimicable, 1 = fully joint)

Low values predict that observational data cannot tell a hyperedge from two pairwise edges
(the identifiability problem in CLAUDE.md §2), whatever the inference method. This uses the
true h on real trajectories, so it characterises the benchmark, not a fitted model.

References (references/references.bib): hooker2007generalized, lengerich2020purifying, buja1989linear, hastie1990generalized.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def triadic_samples(system, X: np.ndarray, n: int = 20000, rng=None, return_groups: bool = False):
    """Inputs [S, 3] = (x_i, x_j, x_k) at true focal triples on data states, and h values [S]
    (+ the trajectory index of every sample [S] with ``return_groups``)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tris = system.structure.tri_index
    if len(tris) == 0:
        raise ValueError("system has no hyperedges")
    S = X.reshape(-1, X.shape[2], X.shape[3])[..., 0]
    s = rng.integers(0, len(S), n)
    t = tris[rng.integers(0, len(tris), n)]
    Z = np.stack([S[s, t[:, 0]], S[s, t[:, 1]], S[s, t[:, 2]]], 1)
    hv = system.h(Z[:, :1], Z[:, 1:2], Z[:, 2:3])[:, 0]
    if return_groups:
        return Z, hv, s // X.shape[1]  # X: [M, T, N, d] -> trajectory of each sampled state
    return Z, hv


def _features(a: np.ndarray, b: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:  # phase inputs: let the regressor see the circle
        return np.stack(
            [np.sin(a), np.cos(a), np.sin(b), np.cos(b), np.sin(b - a), np.cos(b - a)], 1
        )
    return np.stack([a, b], 1)


def pure_interaction_fraction(
    Z: np.ndarray,
    hv: np.ndarray,
    periodic: bool = False,
    n_iter: int = 12,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict:
    """Backfit h ≈ u(x_i, x_j) + u(x_i, x_k) with ONE shared u (symmetric additive surrogate).

    The surrogate is fitted on one half of the samples and the leftover variance is measured on the
    other half; measuring in-sample would let the regressor memorise part of the joint term and
    understate the pure-interaction fraction. Updates are damped (u ← ½(u_new + u_old)): undamped
    simultaneous updates make the part of u that depends on x_i alone flip sign every iteration.
    """
    rng = np.random.default_rng(seed)
    if groups is not None and len(np.unique(groups)) >= 4:
        # split by TRAJECTORY: states of one trajectory are strongly correlated, so a per-sample split
        # puts near-duplicates on both sides and flatters the surrogate (audit E14)
        ug = rng.permutation(np.unique(groups))
        in_fit = np.isin(groups, ug[: len(ug) // 2])
        fit, ev = np.where(in_fit)[0], np.where(~in_fit)[0]
    else:
        perm = rng.permutation(len(hv))
        fit, ev = perm[: len(hv) // 2], perm[len(hv) // 2 :]
    xi, xj, xk = Z[:, 0], Z[:, 1], Z[:, 2]
    F1, F2 = _features(xi, xj, periodic), _features(xi, xk, periodic)
    var_ev = float(np.var(hv[ev])) or 1.0
    u1 = np.zeros(len(fit))
    u2 = np.zeros(len(fit))
    history = []
    for _ in range(n_iter):
        # shared u: fit on both slots at once, each slot's target = h − other slot's current value
        Xs = np.concatenate([F1[fit], F2[fit]])
        ys = np.concatenate([hv[fit] - u2, hv[fit] - u1])
        # regularised (early stopping, larger leaves): an overfit surrogate adds variance on the
        # held-out half and pushes the fraction above 1
        reg = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=50,
            early_stopping=True,
            random_state=seed,
        ).fit(Xs, ys)
        u1 = 0.5 * (u1 + reg.predict(F1[fit]))
        u2 = 0.5 * (u2 + reg.predict(F2[fit]))
        e1 = reg.predict(F1[ev]) if not history else 0.5 * (e1 + reg.predict(F1[ev]))  # noqa: F821
        e2 = reg.predict(F2[ev]) if not history else 0.5 * (e2 + reg.predict(F2[ev]))  # noqa: F821
        resid_ev = hv[ev] - e1 - e2
        history.append(float(np.var(resid_ev) / var_ev))
    return {
        "pure_fraction": history[-1],
        "history": history,
        "h_std": float(np.sqrt(np.var(hv))),
        "n_samples": int(len(hv)),
        "evaluation": "held-out half (by trajectory)" if groups is not None else "held-out half",
    }


def identifiability_report(system, X: np.ndarray, n: int = 20000, seed: int = 0) -> dict:
    Z, hv, grp = triadic_samples(system, X, n, np.random.default_rng(seed), return_groups=True)
    out = pure_interaction_fraction(Z, hv, system.periodic, seed=seed, groups=grp)
    # Var(triadic part) / Var(full vector field) on random data states. Can exceed 1 when the self
    # and pairwise parts partly cancel the triadic part (it is a ratio, not a share).
    S = X.reshape(-1, X.shape[2], X.shape[3])
    sel = np.random.default_rng(seed).choice(len(S), min(2000, len(S)), replace=False)
    parts = system.coupling_parts(S[sel])
    full = parts["self"] + parts["pair"] + parts["tri"]
    out["tri_var_ratio"] = float(np.var(parts["tri"]) / (np.var(full) + 1e-300))
    return out
