#!/usr/bin/env python
"""Mean baseline metrics from a pulled baselines directory (<run>/<method>/baseline_metrics.json).

python scripts/summarize_baselines.py runs/kaggle/p0-baselines/output/results/baselines [--by system]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--by", nargs="*", default=[], help="extra group keys: system kind noisy")
    a = ap.parse_args()
    rows = []
    for f in sorted(a.root.glob("*/*/baseline_metrics.json")):
        run, meth = f.parent.parent.name, f.parent.name
        d = json.loads(f.read_text())
        m = re.match(r"(S\d)_(\w+?)_N(\d+)_noise([\d.]+)", run)
        e = d.get("edges") or {}
        h = (d.get("hyperedges") or {}).get("all") or {}
        fh = d.get("false_hyperedges") or {}
        rows.append(
            dict(system=m[1], kind=m[2], N=int(m[3]), noisy=float(m[4]) > 0, method=meth,
                 eAUC=e.get("auroc"), eF1=e.get("f1"), hAUC=h.get("auroc"), hF1=h.get("f1"),
                 ctrlFP=fh.get("n_false_hyperedges"))
        )
    df = pd.DataFrame(rows)
    for c in ["eAUC", "eF1", "hAUC", "hF1", "ctrlFP"]:
        df[c] = pd.to_numeric(df[c])
    keys = ["method", "noisy", *a.by]
    print(df.groupby(keys)[["eAUC", "eF1", "hAUC", "hF1", "ctrlFP"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
