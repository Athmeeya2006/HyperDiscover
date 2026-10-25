#!/usr/bin/env python
"""Run HyperDiscover on one benchmark instance and write a complete run directory.

    # tiny CPU sanity case (CLAUDE.md §12 test 6)
    python scripts/run_pipeline.py --experiment configs/experiment/sanity_s4.yaml --out runs/sanity_s4

    # benchmark file from make_benchmark.py; split GPU fit and CPU distillation
    python scripts/run_pipeline.py --benchmark-file B.h5 --stage fit --device cuda --out runs/X
    python scripts/run_pipeline.py --benchmark-file B.h5 --stage distill --out runs/X

    # extra config overlays (merged left to right) and dotted overrides
    python scripts/run_pipeline.py ... --config configs/train/weak.yaml --set train.joint.steps=3000

Run directory: provenance.json (git commit, SLURM ids, host), config.resolved.json,
pip_freeze.txt, structure.pt, gates.npz, histories.npz, summary.json, report.md,
metrics.json, pysr/ (hall of fame), equations.json (strings + LaTeX).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import to_jsonable, write_provenance  # noqa: E402


def parse_set(items):
    out: dict = {}
    for it in items or []:
        key, val = it.split("=", 1)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--experiment", help="YAML with a `benchmark:` spec (+ optional `config:`)")
    src.add_argument("--benchmark-file", help=".h5 written by make_benchmark.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[], help="YAML overlay(s)")
    ap.add_argument(
        "--set", action="append", default=[], help="dotted override, e.g. train.joint.steps=100"
    )
    ap.add_argument("--stage", choices=["all", "fit", "distill"], default="all")
    ap.add_argument("--engine", default=None, help="pysr | library | auto | none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-order", type=int, default=None)
    ap.add_argument("--visualize", action="store_true", help="also write walkthrough figures")
    a = ap.parse_args()

    import torch

    from hyperdiscover import HyperDiscover, benchmarks
    from hyperdiscover.api import resolve_config
    from hyperdiscover.pipeline import merge_config

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = {}
    if a.experiment:
        exp = yaml.safe_load(Path(a.experiment).read_text())
        b = benchmarks.load(**exp["benchmark"])
        data, system, spec = b.data, b.system, b.spec
    else:
        data, spec, truth = benchmarks.load_file(a.benchmark_file)
        system = benchmarks.rebuild_system(truth)
    cfg = resolve_config(exp.get("config", {}))
    for c in a.config:
        cfg = merge_config(cfg, yaml.safe_load(Path(c).read_text()) or {})
    cfg = merge_config(cfg, parse_set(a.set))
    if a.device:
        cfg["device"] = a.device
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but CUDA is unavailable")
    seed = a.seed if a.seed is not None else int(spec.get("seed", 0))
    max_order = a.max_order or int(exp.get("max_order", 3))
    engine = a.engine or exp.get("distill_engine", "auto")
    periodic = bool(system.periodic)
    torch.set_num_threads(max(1, int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "4"))))

    # held-out trajectories for evaluation: the last 15% (structure/equations never see them)
    M = data.X.shape[0]
    n_test = max(1, int(round(0.15 * M)))
    X_fit, X_test = data.X[: M - n_test], data.X_clean[M - n_test :]
    t = data.t
    seg = data.meta.get("segment_starts")
    write_provenance(
        out, {"config": cfg, "spec": spec, "engine": engine, "max_order": max_order}, seed, a.stage
    )

    if a.stage in ("all", "fit"):
        hd = HyperDiscover(
            1, max_order, cfg, periodic, "none" if a.stage == "fit" else engine, seed=seed
        )
        hd.fit(X_fit, t, seg, out_dir=out)
        hd.save_structure(out / "structure.pt")
        res = hd.structure_result_
        np.savez_compressed(
            out / "gates.npz",
            pair_index=res.pair_index_all,
            pair_scores=res.pair_scores,
            tri_index=res.tri_index_all,
            tri_scores=res.tri_scores,
        )
        np.savez_compressed(
            out / "histories.npz",
            **{
                f"{h['stage']}__{k}": np.asarray(v)
                for h in res.histories
                for k, v in h.items()
                if k in ("step", "loss", "data", "pair_ids", "tri_ids")
            },
        )
    if a.stage == "distill":
        hd = HyperDiscover.from_structure_checkpoint(
            out / "structure.pt", X_fit, t, engine, segment_starts=seg
        )
        hd.fit_symbolic(X_fit, t, seg, out_dir=out)
    # structure metrics after every stage; equation metrics once equations exist
    from hyperdiscover.eval.run_metrics import run_metrics

    metrics = run_metrics(hd, system, X_test, t)
    metrics["spec"] = spec
    metrics["engine"] = engine if hd.equations_ else "none"
    (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2, default=float))
    if hd.equations_:
        import sympy as sp

        (out / "equations.json").write_text(
            json.dumps(
                {k: {"str": str(v), "latex": sp.latex(v)} for k, v in hd.equations_.items()},
                indent=2,
            )
        )
    hd.save(out)
    s = metrics["edges"]
    msg = f"edges AUROC {s['auroc']:.3f} F1 {s['f1']:.3f}"
    if "hyperedges" in metrics:
        h = metrics["hyperedges"]["all"]
        msg += f" | hyperedges AUROC {h['auroc']:.3f} F1 {h['f1']:.3f}"
    if "equations" in metrics:
        msg += f" | equations exact: {metrics['equations']['all_exact']}"
    print("[metrics]", msg)
    if a.visualize:
        from visualize_pipeline import walkthrough

        walkthrough(hd, system, data, out / "viz")


if __name__ == "__main__":
    main()
