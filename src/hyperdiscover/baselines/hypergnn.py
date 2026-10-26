"""Baseline: SPHINX-style latent-incidence hypergraph network (structure only; CLAUDE.md §18.12 G3).

Idea after SPHINX (Duta & Liò, ICML 2025: a latent incidence matrix inferred from trajectories and trained
through a downstream prediction loss). Our simplified re-implementation, NOT the authors' code:
github.com/IuliaDuta/SPHINX has no licence file (references/LITERATURE.md, topic survey 3), so nothing is
vendored. Differences: free per-system incidence logits instead of their amortized slot encoder; the
downstream task is our derivative regression; a pairwise channel is added so the comparison with
pairwise+triadic methods is fair.

Model (d = 1): K hyperedge slots with soft memberships m_sv = σ(Z_sv) and a soft adjacency a_ij = σ(W_ij);

    ẋ_i = f(x_i) + Σ_j a_ij g(x_i, x_j) + Σ_s m_si · ψ(x_i, (Σ_v m_sv φ(x_v) − m_si φ(x_i)) / (Σ_v m_sv − m_si + ε)),

f, g, φ, ψ small MLPs. Loss: derivative MSE + λ_a mean(a) + λ_m mean(m) + λ_3 Σ_s (Σ_v m_sv − 3)² (slots of
size 3) + a binarisation term mean(m(1 − m)). Scores: pair (i, j) = a_ij; triple {u, v, w} = max_s min(m_su,
m_sv, m_sw) (every focal version gets the same score). No closed-form equations.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn


def _mlp(n_in, n_out, hidden=32):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, n_out))


class LatentIncidenceNet(nn.Module):
    def __init__(self, N: int, K: int, hidden: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Z = nn.Parameter(torch.randn(K, N, generator=g) * 0.5 - 1.0)
        self.W = nn.Parameter(torch.zeros(N, N) - 0.5)
        self.f, self.g = _mlp(1, 1, hidden), _mlp(2, 1, hidden)
        self.phi, self.psi = _mlp(1, hidden, hidden), _mlp(1 + hidden, 1, hidden)
        self.register_buffer("offdiag", 1.0 - torch.eye(N))

    def memberships(self):
        return torch.sigmoid(self.Z), torch.sigmoid(self.W) * self.offdiag

    def forward(self, x):  # x [B, N]
        B, N = x.shape
        m, a = self.memberships()
        xi = x[..., None]
        out = self.f(xi)[..., 0]
        pair = self.g(torch.stack([xi.expand(B, N, N), x[:, None, :].expand(B, N, N)], -1))[..., 0]
        out = out + (a[None] * pair).sum(-1)
        ph = self.phi(xi)  # [B, N, H]
        tot = torch.einsum("kn,bnh->bkh", m, ph)  # [B, K, H]
        own = m.T[None, :, :, None] * ph[:, :, None, :]  # [B, N, K, H]
        cnt = (m.sum(1)[None, None, :] - m.T[None]).clamp(min=1e-3)  # [1, N, K]
        ctx = (tot[:, None] - own) / cnt[..., None]
        msg = self.psi(torch.cat([xi[:, :, None, :].expand(B, N, m.shape[0], 1), ctx], -1))[..., 0]
        return out + (m.T[None] * msg).sum(-1)


def latent_incidence_baseline(
    X: np.ndarray,
    dX: np.ndarray,
    K: int | None = None,
    steps: int = 3000,
    batch: int = 256,
    lam_a: float = 1e-2,
    lam_m: float = 1e-2,
    lam_size: float = 1e-2,
    lam_bin: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train on states X / derivatives dX [M, T, N, 1]; return directed pair and focal triple scores."""
    M, T, N, _ = X.shape
    K = K or max(2, N // 2)
    S = torch.tensor(X[..., 0].reshape(-1, N), dtype=torch.float32, device=device)
    Y = torch.tensor(dX[..., 0].reshape(-1, N), dtype=torch.float32, device=device)
    Y = Y / (Y.std() + 1e-8)
    torch.manual_seed(seed)
    net = LatentIncidenceNet(N, K, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (batch,), generator=gen).to(device)
        pred = net(S[idx])
        m, a = net.memberships()
        loss = (
            ((pred - Y[idx]) ** 2).mean()
            + lam_a * a.mean() + lam_m * m.mean()
            + lam_size * ((m.sum(1) - 3.0) ** 2).mean() + lam_bin * (m * (1 - m)).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m, a = (t.cpu().numpy() for t in net.memberships())
    pair_scores = {(i, j): float(a[i, j]) for i in range(N) for j in range(N) if i != j}
    tri_scores = {}
    for u, v, w in itertools.combinations(range(N), 3):
        sc = float(np.max(np.minimum(np.minimum(m[:, u], m[:, v]), m[:, w])))
        for i, (j, k) in ((u, (v, w)), (v, (u, w)), (w, (u, v))):
            tri_scores[(i, j, k)] = sc
    return {"pair_scores": pair_scores, "tri_scores": tri_scores, "label": "SPHINX-style latent incidence"}
