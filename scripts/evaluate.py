#!/usr/bin/env python
"""Build every paper table/figure from run metrics JSON (CLAUDE.md §9, §14). No hand-copied numbers.

python scripts/evaluate.py --runs runs/ --out paper/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STRUCT = [
    "edges.auroc",
    "edges.auprc",
    "edges.f1",
    "hyperedges.all.auroc",
    "hyperedges.all.auprc",
    "hyperedges.all.f1",
    "hyperedges.open.auroc",
    "hyperedges.partial.auroc",
    "hyperedges.closed.auroc",
    "false_hyperedges.n_false_hyperedges",
    "screening.pair_recall",
    "screening.tri_recall",
]
EQN = [
    "equations.all_exact",
    "equations.f.exact",
    "equations.g.exact",
    "equations.h.exact",
    "equations.f.nrmse",
    "equations.g.nrmse",
    "equations.h.nrmse",
    "equations.g.max_coef_rel_err",
    "equations.h.max_coef_rel_err",
    "coupling.coupling_sum",
    "coupling.full_rhs",
]
DYN = [
    "forecast.vpt",
    "forecast.final_nrmse",
    "perturbation.mean",
    "perturbation.hyper_targeted_mean",
]
KEYS = ["spec.system_id", "spec.hyper_kind", "spec.pairwise_only", "spec.N", "spec.noise"]
# a group must differ ONLY by seed: every other spec field, the config and the engine are keys too
EXTRA_KEYS = ["config_hash", "engine"]


def group_keys(df: pd.DataFrame) -> list[str]:
    spec = [c for c in df.columns if c.startswith("spec.") and c != "spec.seed"]
    spec = [c for c in spec if c not in KEYS and df[c].map(lambda v: not isinstance(v, list)).all()]
    return [k for k in KEYS if k in df.columns] + spec + [k for k in EXTRA_KEYS if k in df.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baselines", default=None, help="root of baseline metrics (run_baselines.py)")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()

    from hyperdiscover.eval.figures import plot_metric_curve
    from hyperdiscover.eval.tables import aggregate, load_metrics, to_latex, to_markdown

    out = Path(a.out)
    df = load_metrics(a.runs)
    if df.empty:
        print(f"no metrics.json under {a.runs}; nothing to do")
        return
    df["method"] = "HyperDiscover"
    if a.baselines:
        bdf = load_metrics(a.baselines, "**/baseline_metrics.json")
        if not bdf.empty:
            df = pd.concat([df, bdf], ignore_index=True)
    if "seconds" in df.columns:  # baselines record wall time as `seconds`, fits as `timings.*`
        tcols = [c for c in df.columns if c.startswith("timings.")]
        df["timings.total"] = df[tcols].sum(axis=1, min_count=1).fillna(df["seconds"])
    for k in EXTRA_KEYS:
        if k in df.columns:
            df[k] = df[k].fillna("-")
    keys = group_keys(df) + ["method"]
    # missing spec fields (e.g. S5-only parameters on other systems) must not drop rows from groupby;
    # pandas >= 3 keeps NaN through astype(str), so fill first
    df[keys] = df[keys].astype(object).where(df[keys].notna(), "-").astype(str)
    dup = (
        df.groupby(keys)["spec.seed"].agg(lambda s: s.duplicated().any())
        if "spec.seed" in df
        else None
    )
    if dup is not None and dup.any():
        n_bad = int(dup.sum())
        print(
            f"WARNING: {n_bad} groups contain repeated seeds (same config run twice?); "
            "keeping those runs separate by run_dir"
        )
        bad = set(dup[dup].index)
        df["dup_run"] = [
            r if tuple(row) in bad else "-" for r, row in zip(df["run_dir"], df[keys].values)
        ]
        keys = keys + ["dup_run"]
    for name, cols in (("structure", STRUCT), ("equations", EQN), ("dynamics", DYN)):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        tab = aggregate(df, keys, cols)
        # grouping uses every spec field; display only the ones that vary (plus the main keys)
        show = [c for c in keys if c in KEYS or c == "method" or tab[c].nunique() > 1]
        tab = tab[show + [c for c in tab.columns if c not in keys]]
        to_markdown(tab, out / "tables" / f"{name}.md")
        to_latex(
            tab,
            out / "tables" / f"{name}.tex",
            caption=f"{name} metrics (mean ± std over seeds)",
            label=f"tab:{name}",
        )
        print("wrote", out / "tables" / f"{name}.md")
    figs = out / "figures"
    # one curve per METHOD, one figure per (system, hyperedge kind); x-axis averaged over seeds only
    for y in ("edges.auroc", "hyperedges.all.auroc", "hyperedges.closed.auroc"):
        if y not in df.columns:
            continue
        for (sid, kind), sub in df.dropna(subset=[y]).groupby(
            ["spec.system_id", "spec.hyper_kind"]
        ):
            for x, logx in (("spec.noise", False), ("spec.N", True)):
                if x in sub.columns and sub[x].nunique() > 1:
                    sub = sub.assign(**{x: sub[x].astype(float)})
                    name = f"{x.split('.')[-1]}_{y}_{sid}_{kind}.png"
                    plot_metric_curve(sub, x, y, "method", figs / name, logx=logx)
    if "timings.total" in df.columns and "spec.N" in df.columns:
        tdf = df.dropna(subset=["timings.total"]).assign(
            **{"spec.N": lambda d: d["spec.N"].astype(float)}
        )
        plot_metric_curve(
            tdf, "spec.N", "timings.total", "method", figs / "scaling_time.png", logx=True
        )
    (out / "tables" / "runs_index.csv").write_text(df[["run_dir"] + keys].to_csv(index=False))
    print(f"{len(df)} runs aggregated")


if __name__ == "__main__":
    main()
