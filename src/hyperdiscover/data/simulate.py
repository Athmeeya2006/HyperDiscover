"""Batched ground-truth simulation (float64, CPU).

We use fixed-step classical RK4 with ``substeps`` internal steps per saved interval. A fixed
step makes batched and sequential simulation bit-for-bit identical (each trajectory evolves
independently of the others), which an adaptive solver over the stacked batch would not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..systems.base import HigherOrderSystem


def rk4_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t: np.ndarray,
    substeps: int = 10,
    post: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Integrate an autonomous ODE at saved times ``t``.

    Args:
        rhs: x [..., N, d] -> dx/dt [..., N, d].
        x0: initial state [..., N, d].
        t: saved times [T] (t[0] is the time of x0), need not be uniform.
        substeps: RK4 steps per saved interval.
        post: optional projection applied after each step (e.g. clipping).

    Returns:
        trajectory [..., T, N, d].
    """
    x = np.array(x0, dtype=np.float64)
    out = np.empty(x.shape[:-2] + (len(t),) + x.shape[-2:], dtype=np.float64)
    out[..., 0, :, :] = x
    for n in range(1, len(t)):
        h = (t[n] - t[n - 1]) / substeps
        for _ in range(substeps):
            k1 = rhs(x)
            k2 = rhs(x + 0.5 * h * k1)
            k3 = rhs(x + 0.5 * h * k2)
            k4 = rhs(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if post is not None:
                x = post(x)
        out[..., n, :, :] = x
    return out


@dataclass
class TrajectoryData:
    """Observed data from one system.

    Attributes:
        X: observed (possibly noisy) states [M, T, N, d].
        t: saved times [T].
        X_clean: noiseless states [M, T, N, d].
        dX_clean: true derivatives at the saved points [M, T, N, d].
        meta: provenance (system id, params, seeds, noise level ...).
    """

    X: np.ndarray
    t: np.ndarray
    X_clean: np.ndarray | None = None
    dX_clean: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.X.shape  # type: ignore[return-value]


def simulate_system(
    system: HigherOrderSystem,
    n_traj: int = 50,
    n_steps: int = 200,
    dt: float | None = None,
    rng: np.random.Generator | None = None,
    substeps: int = 10,
    x0: np.ndarray | None = None,
    burn_in: int = 0,
    kick_std: float = 0.0,
    kick_every: int = 0,
) -> TrajectoryData:
    """Simulate ``n_traj`` trajectories from random (or given) initial conditions.

    Args:
        n_steps: number of saved points T per trajectory.
        dt: saved sampling interval (defaults to ``system.default_dt``).
        burn_in: saved intervals discarded at the start.
        kick_std / kick_every: optional random state kicks (std, in units of the IC box
            width) every ``kick_every`` saved points; the kick intervals are integrated
            separately so derivative targets stay exact between kicks.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    dt = system.default_dt if dt is None else dt
    if x0 is None:
        x0 = system.sample_ic(n_traj, rng)
    t_full = np.arange(n_steps + burn_in) * dt
    if kick_std > 0 and kick_every > 0:
        width = system.ic_box[1] - system.ic_box[0]
        segs, x = [], x0
        for start in range(0, len(t_full), kick_every):
            tt = t_full[start : start + kick_every + 1]
            seg = rk4_integrate(system.rhs, x, tt - tt[0], substeps, system.postprocess)
            segs.append(seg[:, : min(kick_every, len(tt))])
            x = system.postprocess(
                seg[:, -1] + kick_std * width * rng.standard_normal(seg[:, -1].shape)
            )
        traj = np.concatenate(segs, axis=1)[:, : len(t_full)]
    else:
        traj = rk4_integrate(system.rhs, x0, t_full, substeps, system.postprocess)
    traj = traj[:, burn_in:]
    t = t_full[burn_in:] - t_full[burn_in]
    dX = system.rhs(traj)
    meta = {
        "system_id": system.system_id,
        "params": dict(system.params),
        "n_traj": n_traj,
        "n_steps": n_steps,
        "dt": dt,
        "substeps": substeps,
        "kick_std": kick_std,
        "kick_every": kick_every,
        "periodic": bool(system.periodic),
        # with kicks, the state jumps between saved points segment_starts[n]-1 and segment_starts[n]
        "segment_starts": _segment_starts(n_steps, burn_in, kick_every if kick_std > 0 else 0),
    }
    return TrajectoryData(X=traj.copy(), t=t, X_clean=traj, dX_clean=dX, meta=meta)


def _segment_starts(n_steps: int, burn_in: int, kick_every: int) -> list[int]:
    """Saved-point indices (after burn-in) where a new continuous segment starts."""
    if kick_every <= 0:
        return [0]
    jumps = {s - burn_in for s in range(kick_every, n_steps + burn_in, kick_every) if s > burn_in}
    return sorted({0} | jumps)
