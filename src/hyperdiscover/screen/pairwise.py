"""Stage 1, pairwise candidate screening.

If N ≤ ``all_pairs_max_n`` every ordered pair is a candidate. Otherwise each node keeps its
top-K_p sources by a cheap score combining
  (1) a joint ridge regression of ẋ_i on simple features of every x_j (x_j, x_j², and
      sin/cos for phases); the Gram matrix is shared by all targets so one solve serves all
      nodes; score_ij = ‖w_ij‖ on standardised features;
  (2) |partial correlation| between ẋ_i and x_j given the other states (rank-equivalent
      t-statistic form, one N×N inverse for all targets).
Scores are combined by averaging per-target ranks.
"""

from __future__ import annotations

import numpy as np


def node_features(X: np.ndarray, periodic: bool) -> np.ndarray:
    """X [S, N] -> standardised features [S, N, F]."""
    feats = [np.sin(X), np.cos(X)] if periodic else [X, X**2]
    F = np.stack(feats, axis=-1)
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1.0)
    return F


def _modulated_targets(X: np.ndarray, Y: np.ndarray, periodic: bool) -> list[np.ndarray]:
    """Targets ẋ_i and ẋ_i modulated by functions of x_i.

    Couplings like a x_i x_j or K sin(x_j − x_i) = K (sin x_j cos x_i − cos x_j sin x_i) are
    invisible to a regression of ẋ_i on features of x_j alone when the x_i factor averages
    out; multiplying the target by x̃_i (or cos x_i, sin x_i) exposes them while keeping
    the Gram matrix shared across targets.
    """
    Yc = Y - Y.mean(0)
    if periodic:
        return [Yc, Yc * np.cos(X), Yc * np.sin(X)]
    Xt = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return [Yc, Yc * Xt]


def ridge_pair_scores(
    X: np.ndarray, Y: np.ndarray, periodic: bool, alpha: float = 1e-2
) -> np.ndarray:
    """X, Y: [S, N] states and derivatives (d = 1). Returns scores [N_target, N_source]."""
    S, N = X.shape
    F = node_features(X, periodic).reshape(S, -1)  # [S, N*F]
    nf = F.shape[1] // N
    G = F.T @ F / S
    lam = alpha * np.trace(G) / G.shape[0]
    Ginv = np.linalg.inv(G + lam * np.eye(G.shape[0]))
    total = np.zeros((N, N))
    for Yt in _modulated_targets(X, Y, periodic):
        Yt = (Yt - Yt.mean(0)) / np.where(Yt.std(0) > 0, Yt.std(0), 1.0)
        W = (Ginv @ (F.T @ Yt / S)).reshape(N, nf, N)  # [source, feat, target]
        total += (W**2).sum(1).T
    return np.sqrt(total)  # [target, source]


def partial_corr_scores(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Partial-correlation scores |β_j| / sqrt((C⁻¹)_jj) for regressing ẋ_i on all x.

    This is proportional to the OLS t-statistic of x_j, and |partial corr(ẋ_i, x_j | x_-j)|
    = |t| / sqrt(t² + dof) is monotone in |t|, so per-target rankings are identical. One
    inverse of the N×N state covariance serves all targets.
    """
    S, N = X.shape
    Xc = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    C = Xc.T @ Xc / S
    Cinv = np.linalg.inv(C + ridge * np.eye(N))
    Yc = (Y - Y.mean(0)) / np.where(Y.std(0) > 0, Y.std(0), 1.0)
    B = Cinv @ (Xc.T @ Yc / S)  # [N_source, N_target]
    return np.abs(B / np.sqrt(np.diag(Cinv))[:, None]).T


def _rank(scores: np.ndarray) -> np.ndarray:
    """Per-row ranks in [0, 1] (1 = best)."""
    r = scores.argsort(axis=1).argsort(axis=1).astype(float)
    return r / max(1, scores.shape[1] - 1)


def screen_pairs(
    X: np.ndarray,
    dX: np.ndarray,
    k_p: int = 30,
    all_pairs_max_n: int = 200,
    periodic: bool = False,
    max_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate directed pairs.

    Args:
        X, dX: [M, T, N, 1] states and derivative estimates.
    Returns:
        pair_index [E, 2] (target, source) and the score matrix [N, N] (diag = −inf).
    """
    M, T, N, d = X.shape
    rng = rng if rng is not None else np.random.default_rng(0)
    Xs = X[..., 0].reshape(-1, N)
    Ys = dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    s1 = ridge_pair_scores(Xs, Ys, periodic)
    s2 = partial_corr_scores(np.sin(Xs) if periodic else Xs, Ys)
    score = 0.5 * (_rank(s1) + _rank(s2))
    np.fill_diagonal(score, -np.inf)
    if N <= all_pairs_max_n:
        ii, jj = np.where(~np.eye(N, dtype=bool))
        return np.stack([ii, jj], 1).astype(np.int64), score
    k = min(k_p, N - 1)
    top = np.argpartition(-score, k, axis=1)[:, :k]
    ii = np.repeat(np.arange(N), k)
    return np.stack([ii, top.ravel()], 1).astype(np.int64), score


def screening_recall(pair_index: np.ndarray, true_pairs: np.ndarray) -> float:
    """Fraction of true directed pairs present in the candidate list."""
    if len(true_pairs) == 0:
        return 1.0
    cand = {(int(a), int(b)) for a, b in pair_index}
    return float(np.mean([(int(a), int(b)) in cand for a, b in true_pairs]))
