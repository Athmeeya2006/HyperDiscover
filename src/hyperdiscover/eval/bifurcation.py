"""Out-of-sample transition tests (CLAUDE.md §9.3): coupling sweeps with hysteresis.

Both the ground-truth and the discovered system expose ``g_scale`` / ``h_scale``; we sweep
one of them up and down (adiabatic continuation from the previous steady state) and record
an order parameter. For S4 (SIS) this is mean prevalence; for S1 (Kuramoto) |⟨e^{iθ}⟩|.
"""

from __future__ import annotations

import numpy as np

from ..data.simulate import rk4_integrate


def order_parameter(x: np.ndarray, kind: str) -> np.ndarray:
    """x [..., N, 1] -> [...]."""
    if kind == "kuramoto":
        return np.abs(np.mean(np.exp(1j * x[..., 0]), axis=-1))
    return np.mean(x[..., 0], axis=-1)


def sweep(
    system,
    scales: np.ndarray,
    which: str = "g",
    x_low=None,
    x_high=None,
    t_relax: float = 50.0,
    dt: float = 0.1,
    kind: str = "mean",
    avg_last: int = 50,
    rng=None,
) -> dict:
    """Forward (from low state) and backward (from high state) continuation over ``scales``."""
    rng = rng if rng is not None else np.random.default_rng(0)
    N = system.N
    t = np.arange(int(t_relax / dt) + 1) * dt
    if x_low is None:
        x_low = (
            np.full((1, N, 1), 0.01) if kind != "kuramoto" else rng.uniform(0, 2 * np.pi, (1, N, 1))
        )
    if x_high is None:
        x_high = np.full((1, N, 1), 0.99) if kind != "kuramoto" else np.zeros((1, N, 1))
    attr = f"{which}_scale"
    old = getattr(system, attr)
    out = {"scales": np.asarray(scales).tolist()}
    for direction, x0, order in (("forward", x_low, scales), ("backward", x_high, scales[::-1])):
        x, vals = x0.copy(), []
        for s in order:
            setattr(system, attr, float(s))
            with np.errstate(all="ignore"):
                traj = rk4_integrate(system.rhs, x, t, 4, system.postprocess)
            vals.append(float(np.mean(order_parameter(traj[:, -avg_last:], kind))))
            x = traj[:, -1]
            if kind != "kuramoto" and direction == "forward":
                x = np.maximum(x, 1e-3)  # small seed so the absorbing state can be left
        out[direction] = vals if direction == "forward" else vals[::-1]
    setattr(system, attr, old)
    return out


def transition_point(scales, values, level: float = 0.05) -> float:
    """First scale at which the order parameter exceeds ``level`` (nan if never)."""
    values = np.asarray(values)
    idx = np.where(values > level)[0]
    return float(np.asarray(scales)[idx[0]]) if len(idx) else float("nan")


def transition_level(sw: dict, kind: str = "mean") -> float:
    """Level that defines "the transition" for a sweep.

    SIS-like prevalence (kind "mean"): an absolute 0.05 (the epidemic is "on"). Kuramoto order
    parameter r: an absolute level is meaningless because incoherent phases already give
    r ≈ √(π/4N) (≈ 0.2 at N = 20), so use the midpoint between the lowest and highest r on either branch
    (audit E6).
    """
    if kind != "kuramoto":
        return 0.05
    v = np.concatenate([np.asarray(sw["forward"]), np.asarray(sw["backward"])])
    v = v[np.isfinite(v)]
    return float(0.5 * (v.min() + v.max())) if len(v) else float("nan")


def hysteresis_width(sw: dict, level: float | None = None, kind: str = "mean") -> float:
    level = transition_level(sw, kind) if level is None else level
    f = transition_point(sw["scales"], sw["forward"], level)
    b = transition_point(sw["scales"], sw["backward"], level)
    return float(f - b)


def compare_transitions(true_system, pred_system, scales, which="g", kind="mean", **kw) -> dict:
    st = sweep(true_system, scales, which, kind=kind, **kw)
    sp_ = sweep(pred_system, scales, which, kind=kind, **kw)
    level = transition_level(st, kind)  # the TRUE system's level is used for both (same yardstick)
    return {
        "true": st,
        "pred": sp_,
        "level": level,
        "threshold_true": transition_point(st["scales"], st["forward"], level),
        "threshold_pred": transition_point(sp_["scales"], sp_["forward"], level),
        "hysteresis_true": hysteresis_width(st, level),
        "hysteresis_pred": hysteresis_width(sp_, level),
    }
