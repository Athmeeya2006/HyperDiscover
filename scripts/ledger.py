#!/usr/bin/env python
"""Append a compute-ledger line for a finished job (CLAUDE.md §3.5): GPU-hours / CPU core-hours.

Uses ``sacct`` for the current SLURM job when available, otherwise wall-clock × resources.
Writes to <run>/ledger.json; ``--summary <runs root>`` prints totals for RESEARCH_LOG.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def job_elapsed_hours() -> float | None:
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "-X", "-n", "-o", "ElapsedRaw"], text=True
        ).split()
        return int(out[0]) / 3600 if out else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--kind", choices=["gpu", "cpu"])
    ap.add_argument("--summary")
    ap.add_argument("--job-dir", help="write one row for the whole SLURM job here (packed jobs)")
    ap.add_argument(
        "--gpus", type=int, default=1, help="GPUs held by the job (GPU-hours = hours x gpus)"
    )
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.summary:
        tot = {"gpu_hours": 0.0, "cpu_core_hours": 0.0, "n": 0}
        for p in [
            *Path(a.summary).glob("**/ledger.json"),
            *Path(a.summary).glob("**/*.ledger.json"),
        ]:
            for row in json.loads(p.read_text()):
                tot["gpu_hours"] += row["gpu_hours"]
                tot["cpu_core_hours"] += row["cpu_core_hours"]
                tot["n"] += 1
        print(json.dumps(tot, indent=1))
        return
    if a.job_dir:
        hours = job_elapsed_hours()
        if hours is None:
            raise SystemExit("--job-dir needs a SLURM job (sacct elapsed time)")
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        row = {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "job": os.environ.get("SLURM_JOB_ID"),
            "kind": a.kind,
            "hours": hours,
            "gpu_hours": hours * a.gpus if a.kind == "gpu" else 0.0,
            "cpu_core_hours": hours * cpus,
            "source": "sacct (whole job)",
            "note": a.note,
        }
        d = Path(a.job_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{row['job']}_{int(time.time())}.ledger.json").write_text(json.dumps([row], indent=1))
        print("ledger:", row)
        return
    run = Path(a.run)
    hours = job_elapsed_hours()
    source = "sacct"
    if hours is None:  # not under SLURM: sum the stage timings the run recorded
        m = (
            json.loads((run / "metrics.json").read_text())
            if (run / "metrics.json").exists()
            else {}
        )
        hours = sum(float(v) for v in m.get("timings", {}).values()) / 3600
        source = "run timings (excludes start-up and evaluation)"
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    row = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "kind": a.kind,
        "hours": hours,
        "gpu_hours": hours if a.kind == "gpu" else 0.0,
        "cpu_core_hours": hours * cpus,
        "source": source,
    }
    path = run / "ledger.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(row)
    path.write_text(json.dumps(rows, indent=1))
    print("ledger:", row)


if __name__ == "__main__":
    main()
