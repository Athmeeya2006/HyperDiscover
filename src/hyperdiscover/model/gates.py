"""Hard-concrete (L0) gates, Louizos, Welling & Kingma (ICLR 2018).

z = min(1, max(0, s (ζ − γ) + γ)),  s = σ((log u − log(1 − u) + log α) / β),  u ~ U(0, 1).

* ``mode="sample"``: stochastic gates (training).
* ``mode="deterministic"``: test-time estimator min(1, max(0, σ(log α)(ζ − γ) + γ)); reaches
  exact 0 and 1.
* ``mode="hard"``: indicator P(z ≠ 0) > threshold (structure readout).

References (references/references.bib): louizos2018learning, maddison2017concrete, jang2017categorical.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    def __init__(
        self,
        n: int,
        init_prob: float = 0.8,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        init_std: float = 0.01,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.gamma, self.zeta = gamma, zeta
        self.register_buffer("beta", torch.tensor(float(beta)))
        # choose log α so that P(z ≠ 0) = init_prob at the initial temperature
        la0 = math.log(init_prob / (1 - init_prob)) + beta * math.log(-gamma / zeta)
        noise = torch.randn(n, generator=generator) * init_std if n else torch.zeros(0)
        self.log_alpha = nn.Parameter(la0 + noise)

    @property
    def n(self) -> int:
        return self.log_alpha.numel()

    def prob_nonzero(self) -> torch.Tensor:
        """P(z ≠ 0) = σ(log α − β log(−γ/ζ)): the expected L0 norm per gate."""
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))

    def l0(self) -> torch.Tensor:
        return self.prob_nonzero().sum()

    def forward(self, mode: str = "sample", threshold: float = 0.5) -> torch.Tensor:
        if mode == "sample":
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta)
        elif mode == "deterministic":
            s = torch.sigmoid(self.log_alpha)
        elif mode == "hard":
            return (self.prob_nonzero() > threshold).to(self.log_alpha.dtype)
        elif mode == "ones":
            return torch.ones_like(self.log_alpha)
        else:
            raise ValueError(mode)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def set_temperature(self, beta: float) -> None:
        self.beta.fill_(float(beta))

    def keep(self, mask: torch.Tensor) -> None:
        """Drop gates in place (used to prune candidates after a stage)."""
        self.log_alpha = nn.Parameter(self.log_alpha.data[mask].clone())
