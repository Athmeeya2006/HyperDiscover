#!/usr/bin/env python
"""Pack many small fits onto the allocated GPUs (CLAUDE.md §3.5 "pack many fits onto one GPU").

Runs ``run_pipeline.py --stage fit`` for a slice of benchmark instances with ``--per-gpu``
concurrent worker processes on each visible GPU (CUDA_VISIBLE_DEVICES from SLURM), or on CPU.
Every run is resumable (checkpoints), and finished runs (structure.pt present) are skipped.

    python scripts/run_many.py --grid G --bench-dir B --runs-dir R --start 0 --count 40 --per-gpu 4
    python scripts/run_many.py ... --dry-run          # print the plan only

Measure memory first (scripts/profile_memory.py) to choose --per-gpu.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_benchmark import instances, name_of  # noqa: E402


def visible_gpus() -> list[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        return [g for g in v.split(",") if g != ""]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--cpu-workers", type=int, default=4, help="workers when no GPU is visible")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads per worker")
    ap.add_argument("--extra", default="", help="extra args for run_pipeline.py (quoted)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    insts = instances(yaml.safe_load(Path(a.grid).read_text()))
    sel = insts[a.start : a.start + a.count if a.count else None]
    gpus = visible_gpus()
    slots = [g for g in gpus for _ in range(a.per_gpu)] or [None] * a.cpu_workers
    here = Path(__file__).resolve().parent
    jobs = []
    for q, inst in enumerate(sel):
        name = name_of(inst)
        out = Path(a.runs_dir) / name / "seed0"
        if (out / "structure.pt").exists():
            continue
        jobs.append((q, name, out))
    print(
        f"{len(sel)} instances, {len(jobs)} to run, {len(slots)} slots on GPUs {gpus or 'none (CPU)'}"
    )
    if a.dry_run:
        for q, name, out in jobs:
            print(q, name, "->", out)
        return

    def run(slot_idx, name, out):
        gpu = slots[slot_idx % len(slots)]
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(a.threads),
            "SLURM_CPUS_PER_TASK": str(a.threads),
        }
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        bench = Path(a.bench_dir) / f"{name}.h5"
        if not bench.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(here / "make_benchmark.py"),
                    "--grid",
                    a.grid,
                    "--out",
                    a.bench_dir,
                    "--index",
                    str(insts.index(next(i for i in insts if name_of(i) == name))),
                ],
                check=True,
            )
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(here / "run_pipeline.py"),
            "--benchmark-file",
            str(bench),
            "--stage",
            "fit",
            "--device",
            "cuda" if gpu is not None else "cpu",
            "--out",
            str(out),
            *a.extra.split(),
        ]
        t0 = time.time()
        with open(out / "fit.log", "w") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, rc, time.time() - t0, gpu

    failures = 0
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futs = [pool.submit(run, k, name, out) for k, (_, name, out) in enumerate(jobs)]
        for f in as_completed(futs):
            name, rc, sec, gpu = f.result()
            failures += rc != 0
            print(f"{'OK ' if rc == 0 else 'ERR'} {name} ({sec:.0f}s, gpu {gpu})", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
