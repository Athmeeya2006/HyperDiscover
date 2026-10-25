#!/usr/bin/env python
"""Run one experiment under several config variants in parallel and rank them.

    python scripts/run_sweep.py --experiment configs/experiment/sanity_s4.yaml \
        --sweep configs/sweeps/gate_saturation.yaml --out runs/sweeps/gate_sat --per-gpu 4

The sweep YAML maps variant name -> config overrides (merged into the experiment's config); a
variant's optional ``benchmark:`` block overrides the experiment's benchmark spec (data design).
Each variant is a normal run directory (run_pipeline.py --stage all); summary.csv / summary.md
rank the variants by structure F1 (from metrics.json only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_many import visible_gpus  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--engine", default="library")
    ap.add_argument("--only", nargs="*", default=None, help="run only these variants")
    a = ap.parse_args()

    exp = yaml.safe_load(Path(a.experiment).read_text())
    variants = yaml.safe_load(Path(a.sweep).read_text())
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = visible_gpus()
    slots = [g for g in gpus for _ in range(a.per_gpu)] or [None] * a.cpu_workers
    print(
        f"{len(variants)} variants, {len(slots)} slots on GPUs {gpus or 'none (CPU)'}", flush=True
    )

    def run(k, name, overrides):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "metrics.json").exists():
            return name, 0, 0.0
        overrides = dict(overrides or {})
        bench = overrides.pop("benchmark", None)  # optional per-variant data design
        cfg = {**exp, "config": _merge(exp.get("config", {}), overrides)}
        if bench:
            cfg["benchmark"] = {**exp["benchmark"], **bench}
        (d / "experiment.yaml").write_text(yaml.safe_dump(cfg))
        gpu = slots[k % len(slots)]
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(a.threads),
            "SLURM_CPUS_PER_TASK": str(a.threads),
        }
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            str(HERE / "run_pipeline.py"),
            "--experiment",
            str(d / "experiment.yaml"),
            "--out",
            str(d),
            "--engine",
            a.engine,
            "--device",
            "cuda" if gpu is not None else "cpu",
        ]
        t0 = time.time()
        with open(d / "run.log", "w") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, rc, time.time() - t0

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futs = [pool.submit(run, k, n, o) for k, (n, o) in enumerate(variants.items())]
        for f in as_completed(futs):
            n, rc, sec = f.result()
            print(f"{'OK ' if rc == 0 else 'ERR'} {n} ({sec:.0f}s)", flush=True)
    summarize(out, list(variants))


def _merge(a: dict, b: dict) -> dict:
    o = dict(a)
    for k, v in b.items():
        o[k] = _merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
    return o


def summarize(out: Path, names: list[str]) -> None:
    rows = []
    for n in names:
        p = out / n / "metrics.json"
        if not p.exists():
            rows.append({"variant": n, "status": "missing"})
            continue
        m = json.loads(p.read_text())
        e, h = m["edges"], (m.get("hyperedges") or {}).get("all", {})
        rows.append(
            {
                "variant": n,
                "status": "ok",
                "edge_f1": e.get("f1"),
                "edge_auroc": e.get("auroc"),
                "edge_fp": e.get("fp"),
                "edge_fn": e.get("fn"),
                "hyper_f1": h.get("f1"),
                "hyper_auroc": h.get("auroc"),
                "hyper_fp": h.get("fp"),
                "hyper_fn": h.get("fn"),
                "equations_exact": (m.get("equations") or {}).get("all_exact"),
                "train_s": sum(
                    v
                    for k, v in m.get("timings", {}).items()
                    if k in ("pairwise", "triadic", "joint", "alternating")
                ),
            }
        )

    def key(r):
        return (-(r.get("edge_f1") or 0) - (r.get("hyper_f1") or 0),)

    rows.sort(key=key)
    cols = [
        "variant",
        "status",
        "edge_f1",
        "edge_auroc",
        "edge_fp",
        "edge_fn",
        "hyper_f1",
        "hyper_auroc",
        "hyper_fp",
        "hyper_fn",
        "equations_exact",
        "train_s",
    ]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(fmt(r.get(c, "")) for c in cols) for r in rows])
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
