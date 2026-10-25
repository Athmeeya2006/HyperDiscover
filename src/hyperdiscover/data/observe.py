"""Observation model: measurement noise, subsampling, hidden nodes."""

from __future__ import annotations

import numpy as np

from .simulate import TrajectoryData


def add_noise(data: TrajectoryData, sigma_rel: float, rng: np.random.Generator) -> TrajectoryData:
    """Add Gaussian measurement noise with std ``sigma_rel`` × (per-node, per-dim signal std).

    The signal std is computed over all trajectories and times of the clean data. For phase
    systems (``meta['periodic']``) the std of an unwrapped phase is dominated by drift, so the
    reference std is that of a uniform angle, 2π/√12 (DECISIONS.md D-009). ``meta['noise_ref_std']``
    overrides both.
    """
    out = TrajectoryData(data.X.copy(), data.t.copy(), data.X_clean, data.dX_clean, dict(data.meta))
    if sigma_rel <= 0:
        out.meta["noise"] = 0.0
        return out
    ref = data.meta.get("noise_ref_std")
    if ref is None and data.meta.get("periodic", False):
        ref = 2 * np.pi / np.sqrt(12.0)
    std = np.asarray(ref) if ref is not None else data.X_clean.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 0, std, 1.0)
    out.X = data.X_clean + sigma_rel * std * rng.standard_normal(data.X_clean.shape)
    out.meta["noise"] = float(sigma_rel)
    return out


def subsample(data: TrajectoryData, every: int) -> TrajectoryData:
    """Keep every ``every``-th saved point (tests sampling-interval sensitivity)."""
    sl = slice(None, None, every)
    t_new = data.t[sl]
    meta = {**data.meta, "subsample": every, "dt": data.meta.get("dt", 1.0) * every}
    if "segment_starts" in data.meta:
        # kept point q is old point q*every, so a segment starting at old index s starts at
        # new index ceil(s / every)
        new = {0} | {-(-int(s) // every) for s in data.meta["segment_starts"]}
        meta["segment_starts"] = sorted(s for s in new if s < len(t_new))
    return TrajectoryData(
        data.X[:, sl],
        t_new,
        None if data.X_clean is None else data.X_clean[:, sl],
        None if data.dX_clean is None else data.dX_clean[:, sl],
        meta,
    )


def hide_nodes(data: TrajectoryData, frac_observed: float, rng: np.random.Generator):
    """Partial observation (stretch): keep a random subset of nodes. Returns (data, kept_idx)."""
    N = data.X.shape[2]
    keep = np.sort(rng.choice(N, max(1, int(round(frac_observed * N))), replace=False))
    sub = TrajectoryData(
        data.X[:, :, keep],
        data.t,
        None if data.X_clean is None else data.X_clean[:, :, keep],
        None if data.dX_clean is None else data.dX_clean[:, :, keep],
        {**data.meta, "observed_nodes": keep.tolist()},
    )
    return sub, keep


def train_val_test_split(n_traj: int, rng: np.random.Generator, frac=(0.7, 0.15, 0.15)):
    """Split trajectory indices (whole trajectories, never time points) into train/val/test."""
    perm = rng.permutation(n_traj)
    n_tr = max(1, int(round(frac[0] * n_traj)))
    n_va = max(1, int(round(frac[1] * n_traj))) if n_traj > 2 else 0
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
