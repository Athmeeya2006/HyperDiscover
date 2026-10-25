"""Baseline 4: NRI-style relational inference (after Kipf, Fetaya, Wang, Welling & Zemel, ICML 2018).

Simplified re-implementation, NOT the authors' code (github.com/ethanfetaya/NRI; verify
licence before vendoring it, DECISIONS.md D-026):
  * encoder: MLP over each node's trajectory window → node → edge → node → edge message
    passing (the "MLP encoder" of the paper) → logits for 2 edge types {no edge, edge};
  * edge types sampled with Gumbel-softmax (temperature ``tau``);
  * decoder: Markovian MLP message passing, x_{t+1} = x_t + dt · (f(x_t) + Σ_j z_ij g(x_i, x_j));
  * loss: Gaussian NLL of the one-step prediction + KL(q(z) ‖ prior) with a sparse prior.
Edge scores = encoder probability of "edge", averaged over windows. Pairwise only.

This is a learned baseline: it must be trained like our model (GPU for large N). On CPU it is
only for smoke tests.

References (references/references.bib): kipf2018neural, gilmer2017neural.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _mlp(n_in, n_hid, n_out):
    return nn.Sequential(
        nn.Linear(n_in, n_hid), nn.ELU(), nn.Linear(n_hid, n_hid), nn.ELU(), nn.Linear(n_hid, n_out)
    )


class NRIEncoder(nn.Module):
    def __init__(self, window: int, d: int, hidden: int = 64):
        super().__init__()
        self.node1 = _mlp(window * d, hidden, hidden)
        self.edge1 = _mlp(2 * hidden, hidden, hidden)
        self.node2 = _mlp(hidden, hidden, hidden)
        self.edge2 = _mlp(3 * hidden, hidden, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, send, recv, N):
        """x [B, N, window·d] -> edge logits [B, E, 2]."""
        h = self.node1(x)
        e = self.edge1(torch.cat([h[:, send], h[:, recv]], -1))
        skip = e
        agg = torch.zeros(h.shape[0], N, e.shape[-1], device=x.device).index_add(1, recv, e) / max(
            1, N - 1
        )
        h2 = self.node2(agg)
        e2 = self.edge2(torch.cat([h2[:, send], h2[:, recv], skip], -1))
        return self.out(e2)


class NRIDecoder(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.msg = _mlp(2 * d, hidden, d)
        self.self_ = _mlp(d, hidden, d)

    def forward(self, x, z, send, recv, dt):
        """x [B, N, d], z [B, E] edge weights -> x_next [B, N, d]."""
        m = self.msg(torch.cat([x[:, recv], x[:, send]], -1)) * z[..., None]
        agg = torch.zeros_like(x).index_add(1, recv, m)
        return x + dt * (self.self_(x) + agg)


def nri(
    X: np.ndarray,
    t: np.ndarray,
    window: int = 20,
    steps: int = 2000,
    batch: int = 32,
    hidden: int = 64,
    lr: float = 5e-4,
    tau: float = 0.5,
    prior_edge: float = 0.1,
    var: float = 5e-5,
    device: str = "cpu",
    seed: int = 0,
    logger=None,
) -> dict:
    """Train NRI on X [M, T, N, d] and return {"pair_scores": {(i, j): p}} for directed (target, source)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    M, T, N, d = X.shape
    window = min(window, T - 1)
    dt = float(t[1] - t[0])
    send, recv = np.where(~np.eye(N, dtype=bool))  # all ordered pairs (source=send -> target=recv)
    send_t = torch.as_tensor(send, device=device)
    recv_t = torch.as_tensor(recv, device=device)
    mu, sd = X.mean(), X.std() + 1e-12
    Xn = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=device)
    enc, dec = NRIEncoder(window, d, hidden).to(device), NRIDecoder(d, hidden).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    log_prior = torch.log(torch.tensor([1 - prior_edge, prior_edge], device=device))
    t0 = time.time()
    for step in range(steps):
        m = torch.as_tensor(rng.integers(0, M, batch), device=device)
        s = torch.as_tensor(rng.integers(0, T - window, batch), device=device)
        idx = s[:, None] + torch.arange(window + 1, device=device)[None]
        seg = Xn[m[:, None], idx]  # [B, W+1, N, d]
        inp = seg[:, :window].permute(0, 2, 1, 3).reshape(batch, N, window * d)
        logits = enc(inp, send_t, recv_t, N)
        z = F.gumbel_softmax(logits, tau=tau, hard=False)[..., 1]
        pred = dec(
            seg[:, :-1].reshape(-1, N, d), z.repeat_interleave(window, 0), send_t, recv_t, dt
        )  # rhs learned in normalised units
        nll = ((pred - seg[:, 1:].reshape(-1, N, d)) ** 2).mean() / (2 * var)
        q = F.softmax(logits, -1)
        kl = (q * (torch.log(q + 1e-16) - log_prior)).sum(-1).mean()
        loss = nll + kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if logger and step % 500 == 0:
            logger(f"[nri] step {step} nll {float(nll):.3e} kl {float(kl):.3e}")
    enc.eval()
    probs = np.zeros(len(send))
    with torch.no_grad():
        n_eval = 0
        for mm in range(M):
            for s0 in range(0, T - window, max(1, window)):
                inp = Xn[mm, s0 : s0 + window].permute(1, 0, 2).reshape(1, N, window * d)
                probs += F.softmax(enc(inp, send_t, recv_t, N), -1)[0, :, 1].cpu().numpy()
                n_eval += 1
    probs /= max(1, n_eval)
    return {
        "pair_scores": {(int(r), int(s_)): float(p) for s_, r, p in zip(send, recv, probs)},
        "tri_scores": {},
        "seconds": time.time() - t0,
        "label": "NRI-style (simplified reimplementation)",
    }


def gao_yan_two_phase(*_args, **_kw):
    """Gao & Yan 2022 two-phase method / ND2: code availability to be verified (CLAUDE.md §8.5)."""
    raise NotImplementedError(
        "Gao & Yan / ND2 baselines pending code-availability check (DECISIONS.md D-015)"
    )
