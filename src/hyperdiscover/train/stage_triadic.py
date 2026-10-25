"""Stage 3.2: triadic stage, freeze f, g, c and pairwise gates; train h + triadic gates on the residual."""

from __future__ import annotations

from .loop import StageConfig, run_stage


def train_triadic(model, data, cfg: dict, seed: int = 0, ckpt_dir=None, logger=print):
    model.use_tris = True
    sc = StageConfig.from_dict("triadic", cfg)
    sc.trainable = {
        **{"f": False, "g": False, "h": True, "pair_gates": False, "tri_gates": True, "c": False},
        **cfg.get("trainable", {}),
    }
    return run_stage(model, data, sc, seed + 1, ckpt_dir, logger)
