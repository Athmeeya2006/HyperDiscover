#!/usr/bin/env python
"""One-line-per-run summary of a pulled grid (Kaggle/Ada results directory), plus missing runs.

python scripts/summarize_grid.py runs/kaggle/recipe-n50/output/results/runs [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = {
    "edges.auroc": "eAUC",
    "edges.f1": "eF1",
    "edges.fp": "eFP",
    "hyperedges.all.auroc": "hAUC",
    "hyperedges.all.f1": "hF1",
    "hyperedges.all.tp": "hTP",
    "hyperedges.all.fp": "hFP",
    "hyperedges.all.fn": "hFN",
    "false_hyperedges.n_false_hyperedges": "ctrlFP",
    "screening.tri_recall": "triRec",
    "equations.f.exact": "f=",
    "equations.g.exact": "g=",
    "equations.h.exact": "h=",
    "equations.g.nrmse": "g_nrmse",
    "equations.h.nrmse": "h_nrmse",
}


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows, missing = [], []
    for run in sorted(p for p in a.root.iterdir() if p.is_dir()):
        mets = sorted(run.glob("seed*/metrics.json"))
        if not mets:
            missing.append(run.name)
            continue
        for m in mets:
            d = json.loads(m.read_text())
            row = {"run": run.name, "seed": m.parent.name}
            row.update({short: _get(d, long) for long, short in COLS.items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(df.round(3).to_string(index=False))
    if missing:
        print(f"\nMISSING ({len(missing)}):", *missing, sep="\n  ")
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
