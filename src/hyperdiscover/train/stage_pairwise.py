"""Stage 3.1: pairwise stage, train f, g (and c_i) and the pairwise gates; no triadic terms."""

from __future__ import annotations

from .loop import StageConfig, run_stage


def train_pairwise(model, data, cfg: dict, seed: int = 0, ckpt_dir=None, logger=print):
    model.use_tris = False
    sc = StageConfig.from_dict("pairwise", cfg)
    sc.trainable = {
        **{"f": True, "g": True, "h": False, "pair_gates": True, "tri_gates": False, "c": True},
        **cfg.get("trainable", {}),
    }
    hist = run_stage(model, data, sc, seed, ckpt_dir, logger)
    model.use_tris = True
    return hist
