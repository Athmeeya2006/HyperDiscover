"""Torch-side batch samplers for derivative, weak-form and rollout losses.

All samplers draw from whole training trajectories (never mixing train/val time points)
and use an explicit ``torch.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .derivatives import estimate_derivatives
from .weak_form import WeakForm, make_weak_form, weak_targets


@dataclass
class Normalizer:
    """Global affine normalisation shared by all nodes (f, g, h are shared).

    x_mean/x_std: [d] statistics of the states; y_std: [d] std of the derivative targets.
    For periodic systems inputs are embedded as sin/cos and x_mean/x_std are not used.
    """

    x_mean: np.ndarray
    x_std: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> Normalizer:
        d = X.shape[-1]
        xm = X.reshape(-1, d).mean(0)
        xs = X.reshape(-1, d).std(0)
        ys = Y.reshape(-1, d).std(0)
        return cls(xm, np.where(xs > 0, xs, 1.0), np.where(ys > 0, ys, 1.0))

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "y_std")}


class TrainData:
    """Holds one system's training data and samples batches for each loss type.

    Args:
        X: observed states [M, T, N, d] (training trajectories only).
        t: times [T].
        deriv_method: derivative estimator for the "derivative" loss.
        trim: points dropped at each end of every segment for the derivative loss
            (finite-difference / filter edge effects).
        device / dtype: torch placement (float32 on GPU, always).
    """

    def __init__(
        self,
        X: np.ndarray,
        t: np.ndarray,
        deriv_method: str = "savgol",
        deriv_kw: dict | None = None,
        weak_kw: dict | None = None,
        segment_starts=None,
        trim: int = 2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.M, self.T, self.N, self.d = X.shape
        self.t = t
        self.device = torch.device(device)
        self.dtype = dtype
        dX = estimate_derivatives(X, t, deriv_method, segment_starts, **(deriv_kw or {}))
        mask = np.ones(self.T, dtype=bool)
        bounds = list(segment_starts or [0]) + [self.T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            mask[a : a + trim] = False
            mask[max(a, b - trim) : b] = False
        self.time_mask = mask
        S = X[:, mask].reshape(-1, self.N, self.d)
        Y = dX[:, mask].reshape(-1, self.N, self.d)
        self.norm = Normalizer.fit(S, Y)
        self.S = torch.as_tensor(S, dtype=dtype, device=self.device)
        self.Y = torch.as_tensor(Y, dtype=dtype, device=self.device)
        self.X = torch.as_tensor(X, dtype=dtype, device=self.device)
        self.X_np = X
        self.dX_np = dX
        # weak form
        self.wf: WeakForm = make_weak_form(t, segment_starts=segment_starts, **(weak_kw or {}))
        self.W = len(self.wf.starts)
        self.weak_Y = torch.as_tensor(
            weak_targets(X, self.wf).reshape(-1, self.N, self.d), dtype=dtype, device=self.device
        )
        self.phi = torch.as_tensor(self.wf.phi, dtype=dtype, device=self.device)
        self._win_offsets = torch.arange(self.wf.length, device=self.device)
        self._starts = torch.as_tensor(self.wf.starts, device=self.device)
        self.segment_starts = list(segment_starts or [0])
        self.y_var = torch.as_tensor(self.norm.y_std**2, dtype=dtype, device=self.device)
        self.x_var = torch.as_tensor(self.norm.x_std**2, dtype=dtype, device=self.device)

    @property
    def n_samples(self) -> int:
        return self.S.shape[0]

    def derivative_batch(self, batch: int, gen: torch.Generator):
        """(states [B, N, d], derivative targets [B, N, d])."""
        idx = torch.randint(0, self.n_samples, (batch,), generator=gen, device="cpu").to(
            self.device
        )
        return self.S[idx], self.Y[idx]

    def weak_batch(self, batch: int, gen: torch.Generator):
        """(window states [B, L, N, d], weak targets [B, N, d])."""
        n_win = self.M * self.W
        idx = torch.randint(0, n_win, (batch,), generator=gen, device="cpu").to(self.device)
        m, w = idx // self.W, idx % self.W
        tidx = self._starts[w][:, None] + self._win_offsets[None, :]
        return self.X[m[:, None], tidx], self.weak_Y[idx]

    def rollout_batch(self, batch: int, horizon: int, gen: torch.Generator):
        """(segments [B, H+1, N, d], dt) of consecutive observed states (no kick crossing)."""
        valid = []
        bounds = self.segment_starts + [self.T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            valid.extend(range(a, b - horizon))
        if not valid:
            raise ValueError(
                f"no segment is longer than the rollout horizon {horizon}; reduce rollout_horizon"
            )
        valid_t = torch.as_tensor(valid, device=self.device)
        m = torch.randint(0, self.M, (batch,), generator=gen, device="cpu").to(self.device)
        s = valid_t[
            torch.randint(0, len(valid), (batch,), generator=gen, device="cpu").to(self.device)
        ]
        tidx = s[:, None] + torch.arange(horizon + 1, device=self.device)[None, :]
        dt = float(self.t[1] - self.t[0])
        return self.X[m[:, None], tidx], dt
