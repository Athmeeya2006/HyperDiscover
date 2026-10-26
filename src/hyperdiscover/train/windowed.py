"""Structure inference with KNOWN (or already discovered) f, g, h; per time window (CLAUDE.md §18.12 G4a, §18.14 E12b).

With f, g, h fixed the model is linear in the edge weights:

    ẋ_i − f(x_i) − c_i = Σ_j w_ij g(x_i, x_j) + Σ_{(j,k)} w_ijk h(x_i, x_j, x_k),

so structure selection is the per-node non-negative BIC LASSO of train/alternating.py (``select_weights``,
relaxed readout, Occam rescaling of triadic columns) on message columns evaluated with the given
functions. Uses:
  * E12(b): infer the structure of a NEW network from its time series, with f, g, h discovered elsewhere;
  * G4a: time-varying structure — select per time window and flag windows whose support changes
    (a switch / rewiring); consecutive windows with identical support are merged into one segment.

``model`` is any object with ``f(x) / g(xi, xj) / h(xi, xj, xk)`` on [..., 1] arrays and ``c`` (a
HigherOrderSystem: a ground-truth system for the oracle, or distill.refit.SymbolicModel).
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from .alternating import AlternatingConfig, select_weights


def all_candidates(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Every directed pair (i, j) and every focal triple (i, j, k), j < k."""
    pairs = np.array(list(permutations(range(N), 2)), dtype=np.int64)
    tris = np.array(
        [(i, j, k) for i in range(N) for j, k in combinations([v for v in range(N) if v != i], 2)],
        dtype=np.int64,
    )
    return pairs, tris


def message_design(model, S: np.ndarray, pairs: np.ndarray, tris: np.ndarray):
    """S: states [n, N] -> base f + c [n, N], G [n, E], H [n, Ht] (d = 1)."""
    x = S[..., None]
    base = (model.f(x) + model.c)[..., 0]
    G = model.g(x[:, pairs[:, 0]], x[:, pairs[:, 1]])[..., 0] if len(pairs) else np.zeros((len(S), 0))
    H = (
        model.h(x[:, tris[:, 0]], x[:, tris[:, 1]], x[:, tris[:, 2]])[..., 0]
        if len(tris)
        else np.zeros((len(S), 0))
    )
    return base, G, H


def select_structure(
    S: np.ndarray,
    dS: np.ndarray,
    model,
    pairs: np.ndarray,
    tris: np.ndarray,
    cfg: AlternatingConfig | None = None,
    max_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge weights (w_pair [E], w_tri [Ht]) from states S [n, N] and derivative targets dS [n, N]."""
    cfg = cfg or AlternatingConfig()
    rng = rng if rng is not None else np.random.default_rng(0)
    if len(S) > max_samples:
        sel = rng.choice(len(S), max_samples, replace=False)
        S, dS = S[sel], dS[sel]
    base, G, H = message_design(model, S, pairs, tris)
    return select_weights(dS - base, G, H, pairs[:, 0], tris[:, 0], S.shape[1], cfg, len(tris) > 0)


def windowed_structure(
    X: np.ndarray,
    dX: np.ndarray,
    model,
    window: int,
    pairs: np.ndarray,
    tris: np.ndarray,
    cfg: AlternatingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-window selection on X, dX [M, T, N, 1] (windows over the time axis, pooled over trajectories).

    Returns {"starts": [W], "w_pair": [W, E], "w_tri": [W, Ht], "changes": [W − 1] (# support flips between
    consecutive windows), "segments": list of (start, end) time indices with constant support}.
    """
    M, T, N, _ = X.shape
    starts = list(range(0, T - window + 1, window))
    wp, wt = [], []
    for s in starts:
        S = X[:, s : s + window, :, 0].reshape(-1, N)
        dS = dX[:, s : s + window, :, 0].reshape(-1, N)
        a, b = select_structure(S, dS, model, pairs, tris, cfg, rng=rng)
        wp.append(a)
        wt.append(b)
    wp, wt = np.array(wp), np.array(wt)
    sup = np.concatenate([wp > 0, wt > 0], 1)
    changes = (sup[1:] != sup[:-1]).sum(1)
    segs, a = [], 0
    for q, c in enumerate(changes):
        if c > 0:
            segs.append((starts[a], starts[q] + window))
            a = q + 1
    segs.append((starts[a], starts[-1] + window))
    return {"starts": np.array(starts), "w_pair": wp, "w_tri": wt, "changes": changes, "segments": segs}
