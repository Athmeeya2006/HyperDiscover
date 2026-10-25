"""Staged training for identifiability (pairwise -> triadic -> joint)."""

from .loop import StageConfig, run_stage
from .stage_joint import train_joint
from .stage_pairwise import train_pairwise
from .stage_triadic import train_triadic

__all__ = ["StageConfig", "run_stage", "train_pairwise", "train_triadic", "train_joint"]
