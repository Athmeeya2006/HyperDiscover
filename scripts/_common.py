"""Run-directory provenance helpers shared by the scripts (CLAUDE.md §14)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    # Kaggle bundles carry no .git: kaggle_job.py writes GIT_COMMIT at upload time
    for p in (Path(__file__).resolve().parents[1] / "GIT_COMMIT", Path("GIT_COMMIT")):
        if p.exists():
            return p.read_text().strip()
    return "unknown (no commits yet or git unavailable)"


def write_provenance(run_dir: Path, config: dict, seed: int, stage: str = "all") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "python": sys.version,
        "argv": sys.argv,
        "seed": seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # one provenance file per stage, so a later distill stage never overwrites the fit's record
    name = "provenance.json" if stage in ("all", "fit") else f"provenance_{stage}.json"
    (run_dir / name).write_text(json.dumps(prov, indent=2))
    if stage in ("all", "fit"):  # the distill stage reuses the fit's config from structure.pt
        (run_dir / "config.resolved.json").write_text(json.dumps(config, indent=2, default=str))
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
        (run_dir / "pip_freeze.txt").write_text(freeze)
    except Exception:
        pass


def to_jsonable(o):
    import numpy as np

    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o
