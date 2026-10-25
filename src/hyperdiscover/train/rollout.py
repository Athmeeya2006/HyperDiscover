"""Differentiable short-horizon rollout (Stage 0 option C / Stage 3 fine-tuning).

Fixed-step RK4 in torch by default; ``torchdiffeq.odeint`` can be used instead if installed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def rk4_rollout(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int,
    dt: float,
    substeps: int = 2,
) -> torch.Tensor:
    """x0 [B, N, d] -> trajectory [B, n_steps + 1, N, d] at spacing dt."""
    xs = [x0]
    x = x0
    h = dt / substeps
    for _ in range(n_steps):
        for _ in range(substeps):
            k1 = rhs(x)
            k2 = rhs(x + 0.5 * h * k1)
            k3 = rhs(x + 0.5 * h * k2)
            k4 = rhs(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs, dim=1)


def odeint_rollout(
    rhs, x0: torch.Tensor, n_steps: int, dt: float, method: str = "dopri5"
) -> torch.Tensor:
    from torchdiffeq import odeint  # optional dependency

    t = torch.arange(n_steps + 1, dtype=x0.dtype, device=x0.device) * dt
    traj = odeint(lambda _t, x: rhs(x), x0, t, method=method, rtol=1e-5, atol=1e-6)
    return traj.transpose(0, 1)
