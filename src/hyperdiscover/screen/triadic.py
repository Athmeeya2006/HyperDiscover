"""Stage 1, triadic candidate screening.

For each node i:
  1. fit a pairwise ridge model of ẋ_i on self features of x_i and per-pair features of its
     top-K_p screened sources (x_j, x_i x_j, x_j², and sin/cos(x_j − x_i) for phases);
  2. take the residual r_i;
  3. build the partner pool: i's top-L pairwise neighbours ∪ the top-L_r nodes whose own features
     (``_pair_feats``) best correlate with r_i — open-hyperedge partners are not pairwise neighbours
     and often rank low on the pairwise score (S2 N=50: 31 % of true partners in the top 15;
     RESEARCH_LOG 2026-10-26), but they do explain the pairwise residual;
  4. score every pair (j, k) from the pool (and all pairs when N is small) by the
     best |corr(r_i, φ(x_i, x_j, x_k))| over generic joint features φ: products
     (x_j x_k, x_i x_j x_k, x_i² x_j x_k) and symmetric functions of u = x_j − x_i,
     v = x_k − x_i (u v, |u − v|(u + v), (u − v)²(u + v)); sin/cos(x_j + x_k − 2x_i) and
     sin/cos(x_j + x_k) for phases;
  5. keep the top-K_t pairs per node.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _self_feats(xi: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return np.stack([np.ones_like(xi), np.sin(xi), np.cos(xi)], 1)
    return np.stack([np.ones_like(xi), xi, xi**2, xi**3], 1)


def _pair_feats(xi: np.ndarray, xj: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return np.stack([np.sin(xj - xi), np.cos(xj - xi), np.sin(xj), np.cos(xj)], 1)
    return np.stack([xj, xi * xj, xj**2, xi**2 * xj], 1)


def _tri_feats(xi, xj, xk, periodic: bool) -> np.ndarray:
    """[S, P, F] product features for many (j, k) pairs at once (xj, xk: [S, P])."""
    xi = xi[:, None]
    if periodic:
        a = xj + xk - 2 * xi
        return np.stack([np.sin(a), np.cos(a), np.sin(xj + xk), np.cos(xj + xk)], -1)
    u, v = xj - xi, xk - xi  # generic symmetric functions of the two differences
    return np.stack(
        [
            xj * xk,
            xi * xj * xk,
            xi**2 * xj * xk,
            u * v,
            np.abs(u - v) * (u + v),
            (u - v) ** 2 * (u + v),
        ],
        -1,
    )


def _ridge_residual(D: np.ndarray, y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    mu, sd = D.mean(0), D.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Dn = (D - mu) / sd
    Dn[:, sd == 1.0] = D[:, sd == 1.0]  # leave constant columns
    G = Dn.T @ Dn
    w = np.linalg.solve(G + alpha * np.trace(G) / len(G) * np.eye(len(G)), Dn.T @ y)
    return y - Dn @ w


def _residual_partners(
    Xs: np.ndarray, r: np.ndarray, i: int, n: int, periodic: bool
) -> np.ndarray:
    """The n nodes v ≠ i whose features φ(x_i, x_v) best correlate with the standardised residual r [S]."""
    N = Xs.shape[1]
    others = np.array([v for v in range(N) if v != i])
    xi = Xs[:, i : i + 1]
    xv = Xs[:, others]
    if periodic:
        F = np.stack([np.sin(xv - xi), np.cos(xv - xi), np.sin(xv), np.cos(xv)], -1)
    else:
        F = np.stack([xv, xi * xv, xv**2, xi**2 * xv], -1)  # [S, N-1, 4]
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    score = np.abs(np.einsum("s,svf->vf", r, F) / len(r)).max(-1)
    return others[np.argsort(-score)[: min(n, len(others))]]


def screen_triples(
    X: np.ndarray,
    dX: np.ndarray,
    pair_scores: np.ndarray,
    k_p: int = 30,
    k_t: int = 50,
    n_neighbours: int = 15,
    all_triples_max_n: int = 12,
    periodic: bool = False,
    max_samples: int = 20000,
    rng: np.random.Generator | None = None,
    n_resid: int = 15,
    resid_k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate focal triples.

    Args:
        X, dX: [M, T, N, 1]; pair_scores: [N, N] from ``screen_pairs``.
    Returns:
        tri_index [H, 3] (i, j, k) with j < k, and their scores [H].
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    N = X.shape[2]
    Xs = X[..., 0].reshape(-1, N)
    Ys = dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    out_idx, out_score = [], []
    for i in range(N):
        sc = pair_scores[i].copy()
        sc[i] = -np.inf
        order = np.argsort(-sc)
        srcs = order[: min(k_p, N - 1)]
        xi, y = Xs[:, i], Ys[:, i]
        D = np.concatenate(
            [_self_feats(xi, periodic)] + [_pair_feats(xi, Xs[:, j], periodic) for j in srcs], 1
        )
        r = _ridge_residual(D, y)
        r = (r - r.mean()) / (r.std() + 1e-12)
        if N <= all_triples_max_n:
            pool = [v for v in range(N) if v != i]
        else:
            pool = set(order[: min(n_neighbours, N - 1)].tolist())
            if n_resid > 0:
                # partner ranking uses a residual after only the top resid_k sources: with all k_p
                # sources, the pair features of hyperedge partners absorb their marginal signal
                Dp = np.concatenate(
                    [_self_feats(xi, periodic)]
                    + [_pair_feats(xi, Xs[:, j], periodic) for j in order[:resid_k]],
                    1,
                )
                rp = _ridge_residual(Dp, y)
                rp = (rp - rp.mean()) / (rp.std() + 1e-12)
                pool |= set(_residual_partners(Xs, rp, i, n_resid, periodic).tolist())
            pool = sorted(pool)
        pairs = np.array(list(combinations(sorted(pool), 2)), dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            continue
        Phi = _tri_feats(xi, Xs[:, pairs[:, 0]], Xs[:, pairs[:, 1]], periodic)  # [S, P, F]
        Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
        corr = np.abs(np.einsum("s,spf->pf", r, Phi) / len(r)).max(-1)  # [P]
        k = min(k_t, len(pairs))
        top = np.argsort(-corr)[:k]
        out_idx.append(np.column_stack([np.full(k, i), pairs[top]]))
        out_score.append(corr[top])
    if not out_idx:
        return np.zeros((0, 3), np.int64), np.zeros(0)
    return np.concatenate(out_idx).astype(np.int64), np.concatenate(out_score)


def triadic_screening_recall(tri_index: np.ndarray, true_tris: np.ndarray) -> float:
    """Fraction of true focal triples present among the candidates."""
    if len(true_tris) == 0:
        return 1.0
    cand = {(int(a), int(b), int(c)) for a, b, c in tri_index}
    return float(np.mean([(int(a), int(b), int(c)) in cand for a, b, c in true_tris]))
