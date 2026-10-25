"""Amortized (GNN-encoder) hard-concrete gates: an NRI-style alternative to free per-candidate logits.

Instead of one free logit log α_e per candidate edge / triple (gates.HardConcreteGate), the logits
are predicted by a shared message-passing encoder over the candidate graph (after Kipf et al., NRI,
ICML 2018), then fed to the same hard-concrete relaxation (Louizos et al. 2018):

    u_i   = [x_i(s_1..s_K), ẋ_i(s_1..s_K)]             node features at K shared snapshots
    e_i   = φ_node(u_i)                                 node embedding
    e'_i  = φ_upd([e_i, mean_{j ∈ cand(i)} φ_msg([e_i, e_j])])   one message-passing round
    log α_ij  = b_p + φ_pair([e'_i, e'_j, e'_i ⊙ e'_j])
    log α_ijk = b_t + φ_tri([e'_i, e'_j + e'_k, e'_j ⊙ e'_k])     (symmetric in j, k)

Rationale: the shared encoder ties the gates of all candidates together (a learned, data-driven
prior over structure), so a false edge cannot be switched on independently of the evidence in the
two nodes' signals; with free logits every candidate is its own parameter. Gating config:
``gating: gnn_encoder`` (pipeline.py), ``model.gnn: {snapshots, hidden}``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(n_in: int, n_hid: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hid), nn.SiLU(), nn.Linear(n_hid, n_out))


class NodeEncoder(nn.Module):
    """Node features [N, F] + candidate pairs -> node embeddings [N, H] (one message-passing round)."""

    def __init__(self, node_feats: torch.Tensor, pair_index: torch.Tensor, hidden: int = 32):
        super().__init__()
        self.register_buffer("feats", node_feats)
        self.register_buffer("pair_index", pair_index.clone())
        F = node_feats.shape[1]
        self.node = _mlp(F, hidden, hidden)
        self.msg = _mlp(2 * hidden, hidden, hidden)
        self.upd = _mlp(2 * hidden, hidden, hidden)

    def forward(self) -> torch.Tensor:
        e = self.node(self.feats)
        N = e.shape[0]
        i, j = self.pair_index[:, 0], self.pair_index[:, 1]
        m = self.msg(torch.cat([e[i], e[j]], -1))
        agg = torch.zeros_like(e).index_add(0, i, m)
        deg = torch.zeros(N, device=e.device).index_add(0, i, torch.ones_like(i, dtype=e.dtype))
        agg = agg / deg.clamp(min=1)[:, None]
        return self.upd(torch.cat([e, agg], -1))


class GNNGate(nn.Module):
    """Drop-in for HardConcreteGate whose logits come from a shared NodeEncoder.

    ``kind`` = "pair" (index [n, 2], target i, source j) or "tri" (index [n, 3], focal i, j < k).
    """

    def __init__(
        self,
        encoder: NodeEncoder,
        index: torch.Tensor,
        kind: str,
        init_prob: float = 0.8,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.gamma, self.zeta = gamma, zeta
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("index", index.clone())
        self._enc = [encoder]  # shared, owned by the model (list: not registered twice)
        H = encoder.upd[-1].out_features
        self.head = _mlp(3 * H, H, 1)
        nn.init.zeros_(self.head[-1].weight)  # start at the prior: every gate at init_prob
        nn.init.zeros_(self.head[-1].bias)
        la0 = math.log(init_prob / (1 - init_prob)) + beta * math.log(-gamma / zeta)
        self.bias = nn.Parameter(torch.tensor(la0))

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def log_alpha(self) -> torch.Tensor:
        if self.n == 0:
            return self.bias.new_zeros(0)
        e = self._enc[0]()
        if self.kind == "pair":
            a, b = e[self.index[:, 0]], e[self.index[:, 1]]
            z = torch.cat([a, b, a * b], -1)
        else:
            a, b, c = (e[self.index[:, q]] for q in range(3))
            z = torch.cat([a, b + c, b * c], -1)
        return self.bias + self.head(z)[:, 0]

    def gate_parameters(self) -> list[nn.Parameter]:
        """Parameters that define this gate's logits (head, bias and the shared encoder)."""
        return [*self.head.parameters(), self.bias, *self._enc[0].parameters()]

    def prob_nonzero(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))

    def l0(self) -> torch.Tensor:
        return self.prob_nonzero().sum()

    def forward(self, mode: str = "sample", threshold: float = 0.5) -> torch.Tensor:
        la = self.log_alpha
        if mode == "sample":
            u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / self.beta)
        elif mode == "deterministic":
            s = torch.sigmoid(la)
        elif mode == "hard":
            return (self.prob_nonzero() > threshold).to(la.dtype)
        elif mode == "ones":
            return torch.ones_like(la)
        else:
            raise ValueError(mode)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def set_temperature(self, beta: float) -> None:
        self.beta.fill_(float(beta))

    def keep(self, mask: torch.Tensor) -> None:
        self.index = self.index[mask.to(self.index.device)]


def node_features(S: torch.Tensor, Y: torch.Tensor, snapshots: int, periodic: bool, seed: int = 0):
    """[n, N, d] normalised states / targets -> per-node features [N, F] at K shared snapshot rows."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(S.shape[0], generator=g)[: min(snapshots, S.shape[0])].to(S.device)
    s, y = S[idx, :, 0].T, Y[idx, :, 0].T  # [N, K]
    if periodic:
        s = torch.cat([torch.sin(s), torch.cos(s)], 1)
    else:
        s = (s - s.mean()) / (s.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return torch.cat([s, y], 1)
