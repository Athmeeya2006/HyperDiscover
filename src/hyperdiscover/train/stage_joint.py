"""Stage 3.3: joint fine-tuning of everything, optional rollout loss, annealed gate temperature."""

from __future__ import annotations

from .loop import StageConfig, run_stage


def train_joint(model, data, cfg: dict, seed: int = 0, ckpt_dir=None, logger=print):
    model.use_tris = True
    sc = StageConfig.from_dict("joint", cfg)
    sc.trainable = {
        **{"f": True, "g": True, "h": True, "pair_gates": True, "tri_gates": True, "c": True},
        **cfg.get("trainable", {}),
    }
    return run_stage(model, data, sc, seed + 2, ckpt_dir, logger)
