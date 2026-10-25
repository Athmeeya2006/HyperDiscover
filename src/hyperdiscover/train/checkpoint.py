"""Checkpoint / auto-resume (jobs on Ada have time limits)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from ..model.structure_model import StructureModel


def load_model_state(model: StructureModel, sd: dict) -> None:
    """Load a state dict whose candidate lists may have been pruned (shapes differ)."""
    model.pair_index = sd["pair_index"].clone().to(model.pair_index.device)
    model.tri_index = sd["tri_index"].clone().to(model.tri_index.device)
    model.pair_gate.log_alpha = nn.Parameter(sd["pair_gate.log_alpha"].clone())
    model.tri_gate.log_alpha = nn.Parameter(sd["tri_gate.log_alpha"].clone())
    for name in (
        "pair_w",
        "tri_w",
    ):  # explicit weights (alternating gating); absent in old checkpoints
        if name in sd:
            setattr(model, name, sd[name].clone().to(getattr(model, name).device))
    model.load_state_dict(sd)


def save_checkpoint(
    path: str | Path,
    model: StructureModel,
    opt,
    step: int,
    stage: str,
    history,
    fingerprint: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict() if opt else None,
            "step": step,
            "stage": stage,
            "fingerprint": fingerprint,
            "history": history,
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written checkpoint


def load_checkpoint(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
