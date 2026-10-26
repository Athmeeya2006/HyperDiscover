#!/usr/bin/env python
"""Re-score saved equations with the current equation metric (D-041) without retraining.

    python scripts/rescore_equations.py --runs runs/kaggle/recipe-n20-v3/output/results/runs \
        --grid configs/experiment/grid_recipe_n20.yaml --out paper/tables/rescored_recipe_n20.csv

For every run dir with seed0/equations.json, the benchmark instance is regenerated deterministically from the grid
(name → spec), held-out inputs are drawn from its last 20 % of trajectories (as the pipeline's test split does; the
exact trajectories may differ from the original split, so nRMSE can differ slightly) and equation_report is applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import sympy as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_benchmark import instances, name_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from hyperdiscover import benchmarks
    from hyperdiscover.eval.equation_metrics import equation_report
    from hyperdiscover.eval.run_metrics import held_out_inputs

    specs = {name_of(i): i for i in instances(yaml.safe_load(a.grid.read_text()))}
    rows = []
    for run in sorted(p for p in a.runs.iterdir() if p.is_dir()):
        inst_name = run.name.split("__")[0]
        eq = run / "seed0" / "equations.json"
        if inst_name not in specs or not eq.exists():
            continue
        sp_ = dict(specs[inst_name])
        sid = sp_.pop("system_id")
        b = benchmarks.load(sid, **sp_)
        truth = b.system.sympy_exprs()
        if not len(b.structure.hyperedges):
            truth = {k: v for k, v in truth.items() if k != "h"}
        X = b.data.X
        X_test = X[int(0.8 * len(X)) :]
        inputs = held_out_inputs(b.system, X_test)
        pred = {k: sp.sympify(v["str"]) for k, v in json.loads(eq.read_text()).items()}
        rep = equation_report(pred, truth, inputs, node_constants=b.system.has_node_constants)
        row = {"run": run.name, "system": sid, "all_exact": rep["all_exact"]}
        for k in ("f", "g", "h"):
            if k in rep:
                row[f"{k}_exact"] = rep[k]["exact"]
                row[f"{k}_nrmse"] = rep[k]["nrmse"]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    print("\nexact rate by system:\n", df.groupby("system")[["f_exact", "g_exact", "h_exact", "all_exact"]].mean().round(2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
