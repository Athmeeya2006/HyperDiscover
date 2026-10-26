#!/usr/bin/env python
"""Ours vs baselines on the SAME instances (paired), from pulled metrics only (paper Table 2 skeleton).

python scripts/compare_methods.py --ours runs/kaggle/recipe-n20-v3/output/results/runs \
    --baselines runs/kaggle/baselines-recipe-n20/output/results/baselines --out paper/tables/compare_n20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _row(d: dict) -> dict:
    e = d.get("edges") or {}
    h = (d.get("hyperedges") or {}).get("all") or {}
    return dict(edge_auroc=e.get("auroc"), edge_f1=e.get("f1"), hyper_auroc=h.get("auroc"),
                hyper_f1=h.get("f1"), control_fp=(d.get("false_hyperedges") or {}).get("n_false_hyperedges"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for run in sorted(p for p in a.ours.iterdir() if p.is_dir()):
        m = run / "seed0" / "metrics.json"
        if m.exists():
            rows.append(dict(instance=run.name, method="HyperDiscover", **_row(json.loads(m.read_text()))))
    ours = {r["instance"] for r in rows}
    for f in sorted(a.baselines.glob("*/*/baseline_metrics.json")):
        inst, meth = f.parent.parent.name, f.parent.name
        if inst in ours:
            rows.append(dict(instance=inst, method=meth, **_row(json.loads(f.read_text()))))
    df = pd.DataFrame(rows)
    for c in ["edge_auroc", "edge_f1", "hyper_auroc", "hyper_f1", "control_fp"]:
        df[c] = pd.to_numeric(df[c])
    df["system"] = df.instance.str.split("_").str[0]
    df["kind"] = df.instance.str.split("_").str[1]
    df["noisy"] = ~df.instance.str.contains("_noise0_")
    common = set.intersection(*[set(g.instance) for _, g in df.groupby("method")])
    df = df[df.instance.isin(common)]
    agg = df.groupby("method").agg(
        n=("instance", "nunique"), edge_auroc=("edge_auroc", "mean"), edge_f1=("edge_f1", "mean"),
        hyper_auroc=("hyper_auroc", "mean"), hyper_f1=("hyper_f1", "mean"),
        control_fp_per_run=("control_fp", "mean"),
    ).sort_values("hyper_f1", ascending=False)
    by_kind = df[df.kind != "pw"].pivot_table(index="method", columns="kind", values="hyper_f1", aggfunc="mean")
    # paired Wilcoxon of our hyperedge F1 vs each baseline (instances with hyperedges)
    from scipy.stats import wilcoxon

    hp = df[df.kind != "pw"].pivot_table(index="instance", columns="method", values="hyper_f1")
    tests = {}
    for m in hp.columns:
        if m == "HyperDiscover" or hp[m].isna().all():
            continue
        x, y = hp["HyperDiscover"], hp[m]
        ok = x.notna() & y.notna()
        diff = (x[ok] - y[ok]).to_numpy()
        tests[m] = (float(np.mean(diff)), float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0, int(ok.sum()))
    txt = (agg.round(3).to_markdown() + "\n\nHyperedge F1 by kind:\n\n" + by_kind.round(3).to_markdown()
           + "\n\nPaired hyperedge F1, HyperDiscover − baseline (mean diff, Wilcoxon p, n):\n\n"
           + "\n".join(f"- {m}: {d:+.3f}, p = {p:.3g}, n = {n}" for m, (d, p, n) in tests.items()))
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out.with_suffix(".csv"), index=False)
        a.out.with_suffix(".md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
