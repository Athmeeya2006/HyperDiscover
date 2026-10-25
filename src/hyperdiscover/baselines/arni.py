"""Baseline 3: ARNI-style pairwise reconstruction (after Casadiego et al. 2017; simplified).

For each target i, every candidate source j owns a block of basis functions of (x_i, x_j);
blocks are added greedily (largest reduction of the residual sum of squares, OLS refit over
all selected blocks) until the relative improvement falls below ``tol``. The score of j is
the residual reduction it achieved when added (0 if never selected). This is a simplified
re-implementation, not the authors' code (no block-orthogonalisation details).
"""

from __future__ import annotations

import time

import numpy as np


def _block(xi, xj, periodic):
    if periodic:
        d = xj - xi
        return np.stack([np.sin(d), np.cos(d), np.sin(xj), np.cos(xj)], 1)
    return np.stack([xj, xj**2, xj**3, xi * xj, xi**2 * xj], 1)


def _self(xi, periodic):
    if periodic:
        return np.stack([np.ones_like(xi), np.sin(xi), np.cos(xi)], 1)
    return np.stack([np.ones_like(xi), xi, xi**2, xi**3], 1)


def arni(
    X: np.ndarray,
    dX: np.ndarray,
    periodic: bool = False,
    tol: float = 1e-3,
    max_blocks: int | None = None,
    max_samples: int = 5000,
    rng=None,
) -> dict:
    rng = rng if rng is not None else np.random.default_rng(0)
    N = X.shape[2]
    Xs, Ys = X[..., 0].reshape(-1, N), dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    max_blocks = max_blocks or N - 1
    t0 = time.time()
    scores = {}
    for i in range(N):
        y = Ys[:, i]
        base = _self(Xs[:, i], periodic)
        blocks = {j: _block(Xs[:, i], Xs[:, j], periodic) for j in range(N) if j != i}
        chosen: list[int] = []
        A = base
        rss = float(np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2)) or 1.0
        while len(chosen) < max_blocks:
            best = None
            for j, B in blocks.items():
                if j in chosen:
                    continue
                Aj = np.concatenate([A, B], 1)
                r = y - Aj @ np.linalg.lstsq(Aj, y, rcond=None)[0]
                v = float(r @ r)
                if best is None or v < best[0]:
                    best = (v, j)
            if best is None:
                break
            gain = (rss - best[0]) / tss
            if gain < tol:
                break
            chosen.append(best[1])
            A = np.concatenate([A, blocks[best[1]]], 1)
            rss = best[0]
            scores[(i, best[1])] = gain
    return {
        "pair_scores": scores,
        "tri_scores": {},
        "seconds": time.time() - t0,
        "label": "ARNI-style (simplified)",
    }
