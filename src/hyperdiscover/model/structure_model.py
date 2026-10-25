"""Differentiable structure model (Stage 2).

    ẋ̂_i = f(x_i) + c_i + Σ_j z_ij g(x_i, x_j) + Σ_{(j,k)} z_ijk h(x_i, x_j, x_k)

Message passing over explicit candidate lists with ``index_add_`` (no compiled extensions).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .gates import HardConcreteGate
from .messages import Messages


class StructureModel(nn.Module):
    """Args:
    n_nodes, d: number of nodes and state dimension.
    pair_index: [E, 2] candidate directed pairs (target i, source j).
    tri_index: [H, 3] candidate focal triples (i, j, k), j < k.
    node_constants: learn a per-node constant c_i.
    periodic: phase inputs (sin/cos embedding).
    norm: optional Normalizer with x_mean, x_std, y_std.
    """

    def __init__(
        self,
        n_nodes: int,
        d: int,
        pair_index: np.ndarray,
        tri_index: np.ndarray,
        hidden: int = 64,
        layers: int = 3,
        node_constants: bool = False,
        periodic: bool = False,
        norm=None,
        init_prob_pair: float = 0.8,
        init_prob_tri: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.N, self.d = n_nodes, d
        self.register_buffer(
            "pair_index", torch.as_tensor(np.asarray(pair_index).reshape(-1, 2), dtype=torch.long)
        )
        self.register_buffer(
            "tri_index", torch.as_tensor(np.asarray(tri_index).reshape(-1, 3), dtype=torch.long)
        )
        kw = {}
        if norm is not None:
            kw = {"x_mean": norm.x_mean, "x_std": norm.x_std, "y_std": norm.y_std}
        self.msg = Messages(d, hidden, layers, periodic, **kw)
        self.pair_gate = HardConcreteGate(len(self.pair_index), init_prob_pair, generator=gen)
        self.tri_gate = HardConcreteGate(len(self.tri_index), init_prob_tri, generator=gen)
        self.use_c = node_constants
        self.c = nn.Parameter(torch.zeros(n_nodes, d)) if node_constants else None
        self.use_pairs = True
        self.use_tris = True
        # alternating convex gating (train/alternating.py): explicit non-negative edge weights that
        # replace the stochastic gates in every gate_mode
        self.weight_mode = False
        self.register_buffer("pair_w", torch.ones(len(self.pair_index)))
        self.register_buffer("tri_w", torch.ones(len(self.tri_index)))

    # ------------------------------------------------------------------ forward
    def parts(self, x: torch.Tensor, gate_mode: str = "sample") -> dict[str, torch.Tensor]:
        """x: [B, N, d] -> {"self", "pair", "tri"} each [B, N, d]."""
        out_self = self.msg.f(x)
        if self.use_c:
            out_self = out_self + self.c
        pair = torch.zeros_like(x)
        tri = torch.zeros_like(x)
        if self.use_pairs and len(self.pair_index):
            i, j = self.pair_index[:, 0], self.pair_index[:, 1]
            z = self._weights("pair", gate_mode) if self.weight_mode else self.pair_gate(gate_mode)
            m = self.msg.g(x[:, i], x[:, j]) * z[None, :, None]
            pair = pair.index_add(1, i, m)
        if self.use_tris and len(self.tri_index):
            i, j, k = self.tri_index[:, 0], self.tri_index[:, 1], self.tri_index[:, 2]
            z = self._weights("tri", gate_mode) if self.weight_mode else self.tri_gate(gate_mode)
            active = z > 0 if (gate_mode != "sample" or self.weight_mode) else None
            if active is not None and not bool(active.all()):
                i, j, k, z = i[active], j[active], k[active], z[active]
            if len(i):
                m = self.msg.h(x[:, i], x[:, j], x[:, k]) * z[None, :, None]
                tri = tri.index_add(1, i, m)
        return {"self": out_self, "pair": pair, "tri": tri}

    def _weights(self, kind: str, gate_mode: str) -> torch.Tensor:
        w = self.pair_w if kind == "pair" else self.tri_w
        if gate_mode == "hard":
            return (self.normalized_weights(kind) > 0.5).to(w.dtype)
        if gate_mode == "ones":
            return torch.ones_like(w)
        return w

    def normalized_weights(self, kind: str) -> torch.Tensor:
        """Weights divided by the median non-zero weight of their kind (true edges ≈ 1), clipped to [0, 1]."""
        w = self.pair_w if kind == "pair" else self.tri_w
        nz = w[w > 0]
        if len(nz) == 0:
            return torch.zeros_like(w)
        return (w / nz.median()).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, gate_mode: str = "sample") -> torch.Tensor:
        p = self.parts(x, gate_mode)
        return p["self"] + p["pair"] + p["tri"]

    # ------------------------------------------------------------------ penalties
    def l0_pair(self) -> torch.Tensor:
        return self.pair_gate.l0() if self.pair_gate.n else torch.zeros(())

    def l0_tri(self) -> torch.Tensor:
        return self.tri_gate.l0() if self.tri_gate.n else torch.zeros(())

    def anova_penalty(self, x: torch.Tensor, n_perm: int = 4, max_tris: int = 512) -> torch.Tensor:
        """Functional-ANOVA 'pure interaction' penalty on h (identifiability option (b)).

        Estimates the main effects m_j(x_i, x_j) = E_{x_k}[h(x_i, x_j, x_k)] by replacing x_k
        with x_k from permuted batch rows (and likewise for x_j), and penalises their squared
        magnitude weighted by each hyperedge's gate probability. Normalised by y_std².
        """
        H = len(self.tri_index)
        if H == 0 or x.shape[0] < 2:
            return torch.zeros((), device=x.device)
        sel = torch.randperm(H, device=x.device)[: min(H, max_tris)]
        i, j, k = (self.tri_index[sel, c] for c in range(3))
        w = self.tri_gate.prob_nonzero()[sel]
        xi, xj, xk = x[:, i], x[:, j], x[:, k]
        B = x.shape[0]
        mj = torch.zeros_like(xi)
        mk = torch.zeros_like(xi)
        for _ in range(n_perm):
            p = torch.randperm(B, device=x.device)
            mj = mj + self.msg.h(xi, xj, xk[p]) / n_perm  # averages over x_k
            mk = mk + self.msg.h(xi, xj[p], xk) / n_perm  # averages over x_j
        pen = ((mj**2 + mk**2) / self.msg.y_std**2).mean(dim=(0, 2))  # [H_sel]
        return (pen * w).sum() / max(1, len(sel))

    # ------------------------------------------------------------------ structure
    def gate_probs(self) -> tuple[np.ndarray, np.ndarray]:
        if self.weight_mode:
            with torch.no_grad():
                return (
                    self.normalized_weights("pair").cpu().numpy(),
                    self.normalized_weights("tri").cpu().numpy(),
                )
        with torch.no_grad():
            pp = self.pair_gate.prob_nonzero().cpu().numpy() if self.pair_gate.n else np.zeros(0)
            pt = self.tri_gate.prob_nonzero().cpu().numpy() if self.tri_gate.n else np.zeros(0)
        return pp, pt

    def prune(
        self, pair_keep: torch.Tensor | None = None, tri_keep: torch.Tensor | None = None
    ) -> None:
        """Remove candidates (boolean masks) to save compute in later stages."""
        if pair_keep is not None:
            self.pair_index = self.pair_index[pair_keep.to(self.pair_index.device)]
            self.pair_gate.keep(pair_keep.to(self.pair_gate.log_alpha.device))
            self.pair_w = self.pair_w[pair_keep.to(self.pair_w.device)]
        if tri_keep is not None:
            self.tri_index = self.tri_index[tri_keep.to(self.tri_index.device)]
            self.tri_gate.keep(tri_keep.to(self.tri_gate.log_alpha.device))
            self.tri_w = self.tri_w[tri_keep.to(self.tri_w.device)]

    def set_trainable(self, **flags: bool) -> None:
        """Flags: f, g, h, pair_gates, tri_gates, c."""
        groups = {
            "f": self.msg.f_net.parameters(),
            "g": self.msg.g_net.parameters(),
            "h": self.msg.h_net.parameters(),
            "pair_gates": [self.pair_gate.log_alpha],
            "tri_gates": [self.tri_gate.log_alpha],
            "c": [self.c] if self.c is not None else [],
        }
        for name, on in flags.items():
            for p in groups[name]:
                p.requires_grad_(bool(on))

    def param_groups(self, lr_net: float, lr_gate: float, weight_decay: float) -> list[dict]:
        nets = [p for p in self.msg.parameters() if p.requires_grad]
        gates = [p for p in (self.pair_gate.log_alpha, self.tri_gate.log_alpha) if p.requires_grad]
        consts = [self.c] if (self.c is not None and self.c.requires_grad) else []
        groups = []
        if nets:
            groups.append({"params": nets, "lr": lr_net, "weight_decay": weight_decay})
        if gates:
            groups.append({"params": gates, "lr": lr_gate, "weight_decay": 0.0})
        if consts:
            groups.append({"params": consts, "lr": lr_net, "weight_decay": 0.0})
        return groups
