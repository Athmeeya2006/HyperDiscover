"""Baseline 7: statistical pairwise scores (no hyperedges).

* correlation: |Pearson(x_i, x_j)| over all samples;
* partial correlation: from the precision matrix of the states;
* Granger: log-ratio of residual variances of x_i(t+1) regressed on x_i(t..t−p+1) with and
  without x_j(t..t−p+1), pooled over trajectories;
* transfer entropy: binned plug-in estimator TE_{j→i} with history 1.
"""

from __future__ import annotations

import numpy as np


def correlation_scores(X: np.ndarray) -> dict:
    N = X.shape[2]
    C = np.corrcoef(X[..., 0].reshape(-1, N).T)
    return {(i, j): float(abs(C[i, j])) for i in range(N) for j in range(N) if i != j}


def partial_correlation_scores(X: np.ndarray, ridge: float = 1e-6) -> dict:
    N = X.shape[2]
    C = np.cov(X[..., 0].reshape(-1, N).T)
    P = np.linalg.inv(C + ridge * np.eye(N))
    d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d)
    return {(i, j): float(abs(R[i, j])) for i in range(N) for j in range(N) if i != j}


def _lagged(X, p):
    """X [M, T, N] -> targets [S, N] and lag matrix [S, N, p]."""
    M, T, N = X.shape
    Y = X[:, p:].reshape(-1, N)
    L = np.stack([X[:, p - 1 - q : T - 1 - q] for q in range(p)], -1).reshape(-1, N, p)
    return Y, L


def granger_scores(X: np.ndarray, p: int = 2) -> dict:
    Xn = X[..., 0]
    N = Xn.shape[2]
    Y, L = _lagged(Xn, p)
    out = {}
    one = np.ones((len(Y), 1))
    for i in range(N):
        A0 = np.concatenate([one, L[:, i]], 1)
        r0 = Y[:, i] - A0 @ np.linalg.lstsq(A0, Y[:, i], rcond=None)[0]
        v0 = float(r0 @ r0) + 1e-300
        for j in range(N):
            if j == i:
                continue
            A1 = np.concatenate([A0, L[:, j]], 1)
            r1 = Y[:, i] - A1 @ np.linalg.lstsq(A1, Y[:, i], rcond=None)[0]
            out[(i, j)] = float(np.log(v0 / (float(r1 @ r1) + 1e-300)))
    return out


def transfer_entropy_scores(X: np.ndarray, bins: int = 8) -> dict:
    Xn = X[..., 0]
    M, T, N = Xn.shape
    q = np.quantile(Xn.reshape(-1, N), np.linspace(0, 1, bins + 1)[1:-1], axis=0)  # [bins-1, N]
    D = np.stack([np.searchsorted(q[:, n], Xn[..., n]) for n in range(N)], -1)  # [M, T, N]
    fut, now = D[:, 1:].reshape(-1, N), D[:, :-1].reshape(-1, N)

    def H(*cols):
        key = np.zeros(len(cols[0]), dtype=np.int64)
        for c in cols:
            key = key * bins + c
        _, cnt = np.unique(key, return_counts=True)
        p = cnt / cnt.sum()
        return float(-(p * np.log(p)).sum())

    out = {}
    for i in range(N):
        h_fi = H(fut[:, i], now[:, i])
        h_i = H(now[:, i])
        for j in range(N):
            if j == i:
                continue
            # TE = H(f_i | i) − H(f_i | i, j)
            out[(i, j)] = (h_fi - h_i) - (
                H(fut[:, i], now[:, i], now[:, j]) - H(now[:, i], now[:, j])
            )
    return out
