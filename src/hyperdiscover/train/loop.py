"""Generic stage runner shared by the pairwise, triadic and joint stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.dataset import TrainData
from ..model.structure_model import StructureModel
from .checkpoint import load_checkpoint, load_model_state, save_checkpoint
from .losses import derivative_loss, rollout_loss, sparsity_penalty, weak_loss


@dataclass
class StageConfig:
    name: str
    steps: int = 2000
    batch: int = 128
    lr_net: float = 3e-3
    lr_gate: float = 5e-2
    weight_decay: float = 1e-5
    loss: str = "derivative"  # derivative | weak | mixed
    lam_pair: float = 0.01
    lam_tri: float = 0.02
    anova: float = 0.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 5
    rollout_batch: int = 32
    beta_start: float = 2.0 / 3.0
    beta_end: float = 2.0 / 3.0
    # first `warmup_steps` steps: gates held open (z = 1), no sparsity penalty, gates not updated,
    # so the message MLPs fit before structure selection starts (attempt at the gate-saturation
    # problem in RESEARCH_LOG 2026-10-24)
    warmup_steps: int = 0
    grad_clip: float = 5.0
    log_every: int = 200
    history_every: int = 25
    checkpoint_every: int = 500
    trainable: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> StageConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(name=name, **known)


def stage_fingerprint(cfg: StageConfig, model: StructureModel) -> str:
    """Hash of everything that must match for a checkpoint to be resumed (config + candidate lists)."""
    import hashlib
    import json
    from dataclasses import asdict

    d = {
        k: v
        for k, v in asdict(cfg).items()
        if k not in ("log_every", "history_every", "checkpoint_every")
    }
    d["pairs"] = int(model.pair_index.shape[0])
    d["tris"] = int(model.tri_index.shape[0])
    d["hidden"] = [tuple(p.shape) for p in model.msg.parameters()]
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_stage(
    model: StructureModel,
    data: TrainData,
    cfg: StageConfig,
    seed: int = 0,
    ckpt_dir: str | Path | None = None,
    logger=print,
) -> dict[str, Any]:
    """Train ``model`` for one stage. Returns a history dict (losses and gate snapshots)."""
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 12345)  # fixed monitoring batch
    torch.manual_seed(seed)
    if cfg.trainable:
        model.set_trainable(**cfg.trainable)
    opt = torch.optim.Adam(model.param_groups(cfg.lr_net, cfg.lr_gate, cfg.weight_decay))
    history: dict[str, list] = {
        "step": [],
        "loss": [],
        "data": [],
        "pair_probs": [],
        "tri_probs": [],
    }
    start = 0
    ckpt = Path(ckpt_dir) / f"{cfg.name}.pt" if ckpt_dir else None
    fp = stage_fingerprint(cfg, model)
    if ckpt is not None:
        state = load_checkpoint(ckpt)
        if state is not None and state.get("fingerprint", "") != fp:
            logger(
                f"[{cfg.name}] checkpoint {ckpt} was made with a different config/candidates; starting fresh"
            )
            state = None
        if state is not None and state["stage"] == cfg.name:
            load_model_state(model, state["model"])
            if cfg.trainable:
                model.set_trainable(**cfg.trainable)
            opt = torch.optim.Adam(model.param_groups(cfg.lr_net, cfg.lr_gate, cfg.weight_decay))
            if state["opt"] is not None:
                try:
                    opt.load_state_dict(state["opt"])
                except ValueError:
                    pass  # parameter layout changed; continue with a fresh optimiser
            start, history = state["step"], state["history"]
            torch.set_rng_state(state["torch_rng"])
            logger(f"[{cfg.name}] resumed at step {start}")
    t0 = time.time()
    model.train()
    for step in range(start, cfg.steps):
        warm = step < cfg.warmup_steps
        frac = max(0, step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps - 1)
        beta = cfg.beta_start * (cfg.beta_end / cfg.beta_start) ** frac
        model.pair_gate.set_temperature(beta)
        model.tri_gate.set_temperature(beta)
        mode = "ones" if warm else "sample"

        if cfg.loss == "derivative":
            ld, S = derivative_loss(model, data, cfg.batch, gen, gate_mode=mode)
        elif cfg.loss == "weak":
            ld, S = weak_loss(model, data, max(1, cfg.batch // 4), gen, gate_mode=mode)
        elif cfg.loss == "mixed":
            l1, S = derivative_loss(model, data, cfg.batch, gen, gate_mode=mode)
            l2, _ = weak_loss(model, data, max(1, cfg.batch // 4), gen, gate_mode=mode)
            ld = 0.5 * (l1 + l2)
        else:
            raise ValueError(cfg.loss)
        loss = ld if warm else ld + sparsity_penalty(model, cfg.lam_pair, cfg.lam_tri)
        if cfg.rollout_weight > 0 and not warm:
            lr_, _ = rollout_loss(model, data, cfg.rollout_batch, cfg.rollout_horizon, gen)
            loss = loss + cfg.rollout_weight * lr_
        if cfg.anova > 0 and model.use_tris:
            loss = loss + cfg.anova * model.anova_penalty(S[: min(len(S), 256)])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip
            )
        opt.step()

        if step % cfg.history_every == 0 or step == cfg.steps - 1:
            pp, pt = model.gate_probs()
            history["step"].append(step)
            history["loss"].append(float(loss.detach()))
            history["data"].append(float(ld.detach()))
            history["pair_probs"].append(pp.astype(np.float32))
            history["tri_probs"].append(pt.astype(np.float32))
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            pp, pt = model.gate_probs()
            with torch.no_grad():
                ev, _ = derivative_loss(model, data, 512, eval_gen, gate_mode="deterministic")
            eval_gen.manual_seed(seed + 12345)
            logger(
                f"[{cfg.name}] step {step:5d}  loss {float(loss.detach()):.3e}  data {float(ld.detach()):.3e}  det {float(ev):.2e}  "
                f"pairs>0.5 {(pp > 0.5).sum()}/{len(pp)}  tris>0.5 {(pt > 0.5).sum()}/{len(pt)}  "
                f"β {beta:.3f}  {time.time() - t0:.1f}s"
            )
        if ckpt is not None and (step + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(ckpt, model, opt, step + 1, cfg.name, history, fp)
    if ckpt is not None:
        save_checkpoint(ckpt, model, opt, cfg.steps, cfg.name, history, fp)
    model.eval()
    history["stage"] = cfg.name
    history["seconds"] = time.time() - t0
    return history
