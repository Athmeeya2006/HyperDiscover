"""Amortized (learned) triadic screening (CLAUDE.md §18.12 G2).

The hand-made triadic score (screen/triadic.py) is the max |corr| between the pairwise residual r_i and a
few generic joint features. Here a small network, trained once on many simulated systems with known
hyperedges, maps cheap per-candidate statistics to a hyperedge probability, so a new system is scored in
one forward pass (no per-system training). It is used as a drop-in screening score: recall at the same
K_t is the success measure (§18.12), on held-out systems and on a held-out system FAMILY.

Per candidate focal triple (i, j, k) the statistics are (all scale-free):
  * |corr(r_i, φ_f)| for every joint feature φ_f of ``_tri_feats`` (6 generic or 4 periodic, zero-padded);
  * |corr(r^small_i, ψ(x_i, x_v))| (residual-partner score) of v = j and v = k, sorted (min, max);
  * pairwise screening ranks of (i←j), (i←k) and the (j, k) pair-score symmetrised, as rank/N;
  * how many other candidate triples of i contain j or k (normalised), a crude "hub" statistic.
One message-passing round over the candidates of the same focal node adds context (mean embedding).

References (references/references.bib): lowe2022amortized, kipf2018neural, gilmer2017neural.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import nn

from ..screen.triadic import _pair_feats, _ridge_residual, _self_feats, _tri_feats

N_TRI_FEATS = 6
N_FEATS = N_TRI_FEATS + 2 + 3 + 2


def triple_candidates_and_features(
    X: np.ndarray,
    dX: np.ndarray,
    pair_scores: np.ndarray,
    periodic: bool = False,
    pool: int = 30,
    k_p: int = 30,
    resid_k: int = 8,
    max_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate focal triples from a wide pool and their statistics.

    Pool per focal node i: top-``pool`` pairwise sources ∪ top-``pool`` residual partners (N ≤ pool + 1:
    all nodes). Returns tri_index [H, 3] (j < k) and features [H, N_FEATS] (float32).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    N = X.shape[2]
    Xs = X[..., 0].reshape(-1, N)
    Ys = dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    S = len(Xs)
    sc = np.where(np.eye(N, dtype=bool), -np.inf, pair_scores)
    order_all = np.argsort(-sc, 1)
    rank = np.argsort(order_all, 1) / max(N - 1, 1)  # rank[i, j] in [0, 1)
    sym = np.minimum(rank, rank.T)
    idx_out, feat_out = [], []
    for i in range(N):
        order = order_all[i][order_all[i] != i]
        xi, y = Xs[:, i], Ys[:, i]
        D = np.concatenate([_self_feats(xi, periodic)] + [_pair_feats(xi, Xs[:, j], periodic) for j in order[:k_p]], 1)
        r = _ridge_residual(D, y)
        r = (r - r.mean()) / (r.std() + 1e-12)
        Dp = np.concatenate([_self_feats(xi, periodic)] + [_pair_feats(xi, Xs[:, j], periodic) for j in order[:resid_k]], 1)
        rp = _ridge_residual(Dp, y)
        rp = (rp - rp.mean()) / (rp.std() + 1e-12)
        others = np.array([v for v in range(N) if v != i])
        xv = Xs[:, others]
        xi1 = xi[:, None]
        if periodic:
            F = np.stack([np.sin(xv - xi1), np.cos(xv - xi1), np.sin(xv), np.cos(xv)], -1)
        else:
            F = np.stack([xv, xi1 * xv, xv**2, xi1**2 * xv], -1)
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        partner = np.zeros(N)
        partner[others] = np.abs(np.einsum("s,svf->vf", rp, F) / S).max(-1)
        if N <= pool + 1:
            cand = list(others)
        else:
            cand = set(order[:pool].tolist()) | set(others[np.argsort(-partner[others])[:pool]].tolist())
            cand = sorted(cand)
        pairs = np.array(list(combinations(cand, 2)), dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            continue
        Phi = _tri_feats(xi, Xs[:, pairs[:, 0]], Xs[:, pairs[:, 1]], periodic)
        Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
        corr = np.abs(np.einsum("s,spf->pf", r, Phi) / S)  # [P, F]
        corr = np.pad(corr, ((0, 0), (0, N_TRI_FEATS - corr.shape[1])))
        j, k = pairs[:, 0], pairs[:, 1]
        pj, pk = partner[j], partner[k]
        cnt = np.bincount(pairs.ravel(), minlength=N) / max(len(pairs), 1)
        feats = np.column_stack(
            [corr, np.minimum(pj, pk), np.maximum(pj, pk),
             np.minimum(rank[i, j], rank[i, k]), np.maximum(rank[i, j], rank[i, k]), sym[j, k],
             np.minimum(cnt[j], cnt[k]), np.maximum(cnt[j], cnt[k])]
        )
        idx_out.append(np.column_stack([np.full(len(pairs), i), pairs]))
        feat_out.append(feats)
    if not idx_out:
        return np.zeros((0, 3), np.int64), np.zeros((0, N_FEATS), np.float32)
    return np.concatenate(idx_out).astype(np.int64), np.concatenate(feat_out).astype(np.float32)


class AmortizedTripleScorer(nn.Module):
    """Candidate statistics [H, F] + focal ids [H] -> hyperedge logits [H] (one context round per focal node)."""

    def __init__(self, n_feats: int = N_FEATS, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_feats, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, feats: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
        e = self.enc(feats)
        n = int(focal.max()) + 1 if len(focal) else 0
        ctx = torch.zeros(n, e.shape[1], dtype=e.dtype).index_add_(0, focal, e)
        cnt = torch.zeros(n, dtype=e.dtype).index_add_(0, focal, torch.ones_like(focal, dtype=e.dtype))
        ctx = ctx / cnt.clamp(min=1)[:, None]
        return self.out(torch.cat([e, ctx[focal]], 1))[:, 0]


def standardize(feats: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu, sd = feats.mean(0), feats.std(0) + 1e-6
    return (feats - mu) / sd, mu, sd
