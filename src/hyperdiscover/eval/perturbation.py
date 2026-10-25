"""Identifiability option (c): held-out perturbation-response test (CLAUDE.md §7, Stage 3).

Models that fit observational data equally well can disagree under interventions. We clamp
a set of nodes to fixed values (a "do" intervention: dx/dt = 0 for clamped nodes) and compare
the responses of the free nodes under the true and the discovered dynamics.

Interventions are designed to separate pairwise from triadic explanations: for every true
hyperedge {i, j, k} we clamp (j, k) to (high, low) and (high, high). A triadic term
h ∝ x_j x_k is silent in the first case and active in the second, whereas a pairwise model that
absorbed the triadic effect responds similarly to both.
"""

from __future__ import annotations

import numpy as np

from ..data.simulate import rk4_integrate


def clamped_rhs(system, clamp_nodes: np.ndarray):
    """Vector field with dx/dt = 0 on ``clamp_nodes``."""
    clamp_nodes = np.asarray(clamp_nodes, dtype=np.int64)

    def rhs(x):
        out = system.rhs(x)
        out[..., clamp_nodes, :] = 0.0
        return out

    return rhs


def simulate_intervention(
    system, x0: np.ndarray, t: np.ndarray, clamp_nodes, clamp_values, substeps: int = 10
):
    """x0 [B, N, d] -> trajectories [B, T, N, d] with the given nodes held at ``clamp_values``."""
    x0 = np.array(x0, dtype=np.float64, copy=True)
    clamp_nodes = np.asarray(clamp_nodes, dtype=np.int64)
    x0[..., clamp_nodes, :] = np.asarray(clamp_values, dtype=np.float64).reshape(-1, 1)
    post = getattr(system, "postprocess", None)
    with np.errstate(all="ignore"):
        return rk4_integrate(clamped_rhs(system, clamp_nodes), x0, t, substeps, post)


def design_interventions(
    true_system, rng: np.random.Generator, n_random: int = 10, low=None, high=None
):
    """List of (clamp_nodes, clamp_values, label). Hyperedge-targeted + random single-node clamps."""
    lo, hi = true_system.ic_box
    low = lo + 0.05 * (hi - lo) if low is None else low
    high = lo + 0.9 * (hi - lo) if high is None else high
    out = []
    for a, b, c in true_system.structure.hyperedges:
        for focal, j, k in ((a, b, c), (b, a, c), (c, a, b)):
            out.append(([j, k], [high, low], f"hyper({focal}|{j},{k}) high/low"))
            out.append(([j, k], [high, high], f"hyper({focal}|{j},{k}) high/high"))
    N = true_system.N
    for _ in range(n_random):
        v = int(rng.integers(0, N))
        out.append(([v], [float(rng.uniform(low, high))], f"node {v}"))
    return out


def response_error(
    true_traj: np.ndarray, pred_traj: np.ndarray, free: np.ndarray, periodic: bool = False
) -> float:
    """nRMSE of free-node trajectories (relative to the spread of the true response)."""
    d = pred_traj[..., free, :] - true_traj[..., free, :]
    if periodic:
        d = np.angle(np.exp(1j * d))
    den = true_traj[..., free, :].std() + 1e-12
    v = float(np.sqrt(np.mean(d**2)) / den)
    return v if np.isfinite(v) else float("inf")


def perturbation_test(
    true_system,
    models: dict,
    t: np.ndarray,
    n_ic: int = 5,
    n_random: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compare several candidate models (name -> system with ``rhs``) under the same interventions.

    Returns per-model mean / max response error and the per-intervention table.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x0 = true_system.sample_ic(n_ic, rng)
    plan = design_interventions(true_system, rng, n_random)
    N = true_system.N
    rows = []
    for nodes, vals, label in plan:
        free = np.setdiff1d(np.arange(N), nodes)
        tt = simulate_intervention(true_system, x0, t, nodes, vals)
        row = {"intervention": label, "nodes": list(map(int, nodes))}
        for name, m in models.items():
            row[name] = response_error(
                tt, simulate_intervention(m, x0, t, nodes, vals), free, true_system.periodic
            )
        rows.append(row)
    summary = {}
    for name in models:
        errs = np.array([r[name] for r in rows])
        hyper = np.array([r[name] for r in rows if r["intervention"].startswith("hyper")])
        summary[name] = {
            "mean": float(np.mean(errs)),
            "max": float(np.max(errs)),
            "hyper_targeted_mean": float(np.mean(hyper)) if len(hyper) else float("nan"),
        }
    return {"summary": summary, "rows": rows}
