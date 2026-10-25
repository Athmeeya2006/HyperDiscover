"""Training losses. All data terms are normalised by the target variance (per feature)."""

from __future__ import annotations

import torch

from ..data.dataset import TrainData
from ..model.structure_model import StructureModel
from .rollout import rk4_rollout


def derivative_loss(model: StructureModel, data: TrainData, batch: int, gen, gate_mode="sample"):
    S, Y = data.derivative_batch(batch, gen)
    pred = model(S, gate_mode)
    return (((pred - Y) ** 2) / data.y_var).mean(), S


def weak_loss(model: StructureModel, data: TrainData, batch: int, gen, gate_mode="sample"):
    """Σ_l φ_l F(x_l) vs −Σ_l φ'_l x_l; one gate sample shared by the whole window."""
    Xw, Yw = data.weak_batch(batch, gen)  # [B, L, N, d], [B, N, d]
    B, L, N, d = Xw.shape
    F = model(Xw.reshape(B * L, N, d), gate_mode).reshape(B, L, N, d)
    pred = torch.einsum("l,blnd->bnd", data.phi, F)
    return (((pred - Yw) ** 2) / data.y_var).mean(), Xw.reshape(B * L, N, d)


def rollout_loss(
    model: StructureModel, data: TrainData, batch: int, horizon: int, gen, gate_mode="sample"
):
    seg, dt = data.rollout_batch(batch, horizon, gen)  # [B, H+1, N, d]
    # sample the gates once per rollout so the vector field is fixed along the trajectory
    if gate_mode == "sample":
        zp = model.pair_gate("sample")
        zt = model.tri_gate("sample")
        rhs = _fixed_gate_rhs(model, zp, zt)
    else:
        rhs = lambda x: model(x, gate_mode)  # noqa: E731
    traj = rk4_rollout(rhs, seg[:, 0], horizon, dt)
    return (((traj[:, 1:] - seg[:, 1:]) ** 2) / data.x_var).mean(), seg[:, 0]


def _fixed_gate_rhs(model: StructureModel, zp: torch.Tensor, zt: torch.Tensor):
    def rhs(x: torch.Tensor) -> torch.Tensor:
        out = model.msg.f(x)
        if model.use_c:
            out = out + model.c
        if model.use_pairs and len(model.pair_index):
            i, j = model.pair_index[:, 0], model.pair_index[:, 1]
            out = out.index_add(1, i, model.msg.g(x[:, i], x[:, j]) * zp[None, :, None])
        if model.use_tris and len(model.tri_index):
            i, j, k = model.tri_index[:, 0], model.tri_index[:, 1], model.tri_index[:, 2]
            out = out.index_add(1, i, model.msg.h(x[:, i], x[:, j], x[:, k]) * zt[None, :, None])
        return out

    return rhs


def sparsity_penalty(model: StructureModel, lam_pair: float, lam_tri: float) -> torch.Tensor:
    """λ_p Σ E[z_ij]/N + λ_t Σ E[z_ijk]/N.

    With the data term averaged over nodes, a gate on node i survives roughly iff removing it
    raises node i's normalised MSE by more than λ (a 'fraction of explained variance' cost).
    """
    pen = torch.zeros((), device=model.pair_gate.log_alpha.device)
    if model.use_pairs and model.pair_gate.n and model.pair_gate.log_alpha.requires_grad:
        pen = pen + lam_pair * model.l0_pair() / model.N
    if model.use_tris and model.tri_gate.n and model.tri_gate.log_alpha.requires_grad:
        pen = pen + lam_tri * model.l0_tri() / model.N
    return pen
