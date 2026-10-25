"""Dynamics metrics (CLAUDE.md §9.3): rollout nRMSE vs horizon, valid prediction time."""

from __future__ import annotations

import numpy as np

from ..data.simulate import rk4_integrate


def rollout_errors(
    true_traj: np.ndarray, pred_traj: np.ndarray, periodic: bool = False
) -> np.ndarray:
    """nRMSE(t) over trajectories and nodes: [T]. Phases compared on the circle."""
    diff = pred_traj - true_traj
    if periodic:
        diff = np.angle(np.exp(1j * diff))
        ref = np.pi / np.sqrt(3)  # std of a uniform angle
    else:
        ref = true_traj.std() + 1e-12
    return np.sqrt(np.mean(diff**2, axis=(0, 2, 3))) / ref


def valid_prediction_time(err: np.ndarray, t: np.ndarray, threshold: float = 0.3) -> float:
    above = np.where(err > threshold)[0]
    return float(t[above[0]] if len(above) else t[-1])


def forecast_report(
    true_system, pred_system, X0: np.ndarray, t: np.ndarray, periodic=False, threshold=0.3
) -> dict:
    """Integrate both from the same ICs X0 [M, N, d] on times t."""
    tt = rk4_integrate(true_system.rhs, X0, t, 10, true_system.postprocess)
    with np.errstate(all="ignore"):
        tp = rk4_integrate(pred_system.rhs, X0, t, 10)
    err = rollout_errors(tt, tp, periodic)
    err = np.where(np.isfinite(err), err, np.inf)
    return {
        "nrmse_vs_t": err.tolist(),
        "t": t.tolist(),
        "vpt": valid_prediction_time(err, t, threshold),
        "final_nrmse": float(err[-1]),
    }
