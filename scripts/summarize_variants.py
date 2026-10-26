#!/usr/bin/env python
"""Compare config variants of a grid run with --variants (run dirs <instance>__<variant>).

python scripts/summarize_variants.py runs/kaggle/p0-ab-sparse/output/results/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    rows, missing = [], []
    for run in sorted(p for p in a.root.iterdir() if p.is_dir() and "__" in p.name):
        inst, variant = run.name.split("__", 1)
        m = run / "seed0" / "metrics.json"
        if not m.exists():
            missing.append(run.name)
            continue
        d = json.loads(m.read_text())
        h = (d.get("hyperedges") or {}).get("all") or {}
        e = d.get("edges") or {}
        rows.append(dict(
            variant=variant, control="_pw_" in inst, noisy="noise0_" not in inst,
            edge_f1=e.get("f1"), edge_auroc=e.get("auroc"), hyper_f1=h.get("f1"), hyper_fp=h.get("fp"),
            hyper_fn=h.get("fn"), control_fp=(d.get("false_hyperedges") or {}).get("n_false_hyperedges"),
        ))
    df = pd.DataFrame(rows)
    agg = df.groupby(["variant", "control", "noisy"]).agg(
        n=("edge_f1", "size"), edge_f1=("edge_f1", "mean"), edge_auroc=("edge_auroc", "mean"),
        hyper_f1=("hyper_f1", "mean"), hyper_fp=("hyper_fp", "sum"), hyper_fn=("hyper_fn", "sum"),
        control_fp=("control_fp", "sum"),
    )
    print(agg.round(3).to_string())
    if missing:
        print(f"\nMISSING ({len(missing)}): " + ", ".join(sorted({m.split('__')[1] for m in missing})))


if __name__ == "__main__":
    main()
