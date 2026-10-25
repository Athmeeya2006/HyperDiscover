"""Shared message functions f (self), g (pairwise) and h (triadic) as small MLPs.

Inputs are raw states; each network embeds them internally:
* non-periodic: (x − x_mean) / x_std per feature;
* periodic (phases): [sin x, cos x] per feature, so every message is 2π-periodic in each
  argument (CLAUDE.md §15, phase wrapping).
Outputs are multiplied by y_std so the networks work at unit scale.
h is symmetrised in (j, k): h(x_i, x_j, x_k) = ½ [ĥ(x_i, x_j, x_k) + ĥ(x_i, x_k, x_j)].
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(
    n_in: int, n_out: int, hidden: int = 64, layers: int = 3, zero_last: bool = False
) -> nn.Sequential:
    mods: list[nn.Module] = []
    w = n_in
    for _ in range(layers):
        mods += [nn.Linear(w, hidden), nn.SiLU()]
        w = hidden
    last = nn.Linear(w, n_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


class Embed(nn.Module):
    def __init__(self, d: int, periodic: bool, x_mean=None, x_std=None) -> None:
        super().__init__()
        self.periodic = periodic
        self.register_buffer(
            "x_mean",
            torch.as_tensor(x_mean if x_mean is not None else [0.0] * d, dtype=torch.float32),
        )
        self.register_buffer(
            "x_std", torch.as_tensor(x_std if x_std is not None else [1.0] * d, dtype=torch.float32)
        )
        self.out_dim = 2 * d if periodic else d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.periodic:
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return (x - self.x_mean) / self.x_std


class Messages(nn.Module):
    """f: R^d → R^d, g: R^{2d} → R^d, h: R^{3d} → R^d (all shared across nodes)."""

    def __init__(
        self,
        d: int = 1,
        hidden: int = 64,
        layers: int = 3,
        periodic: bool = False,
        x_mean=None,
        x_std=None,
        y_std=None,
    ) -> None:
        super().__init__()
        self.d = d
        self.embed = Embed(d, periodic, x_mean, x_std)
        e = self.embed.out_dim
        self.f_net = mlp(e, d, hidden, layers)
        self.g_net = mlp(2 * e, d, hidden, layers, zero_last=True)
        self.h_net = mlp(3 * e, d, hidden, layers, zero_last=True)
        self.register_buffer(
            "y_std", torch.as_tensor(y_std if y_std is not None else [1.0] * d, dtype=torch.float32)
        )

    def f(self, xi: torch.Tensor) -> torch.Tensor:
        return self.f_net(self.embed(xi)) * self.y_std

    def g(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        return self.g_net(torch.cat([self.embed(xi), self.embed(xj)], -1)) * self.y_std

    def h(self, xi: torch.Tensor, xj: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:
        ei, ej, ek = self.embed(xi), self.embed(xj), self.embed(xk)
        a = self.h_net(torch.cat([ei, ej, ek], -1))
        b = self.h_net(torch.cat([ei, ek, ej], -1))
        return 0.5 * (a + b) * self.y_std
