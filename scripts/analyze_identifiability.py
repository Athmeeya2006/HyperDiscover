#!/usr/bin/env python
"""Pure-interaction fraction of the true triadic term for every benchmark instance (no training).

    python scripts/analyze_identifiability.py --bench runs/benchmarks_v0.1 --pattern "*_N20_noise0_*" --out paper/tables

Writes identifiability.csv / identifiability.md (one row per instance) and a figure.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pattern", default="*_N20_noise0_*")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    import pandas as pd

    from hyperdiscover import benchmarks
    from hyperdiscover.eval.identifiability import identifiability_report

    rows = []
    for f in sorted(glob.glob(str(Path(a.bench) / f"{a.pattern}.h5"))):
        data, spec, truth = benchmarks.load_file(f)
        if spec["pairwise_only"]:
            continue
        system = benchmarks.rebuild_system(truth)
        rep = identifiability_report(system, data.X_clean, a.n)
        rows.append(
            {
                "system": spec["system_id"],
                "kind": spec["hyper_kind"],
                "N": spec["N"],
                "noise": spec["noise"],
                "pure_fraction": rep["pure_fraction"],
                "tri_var_ratio": rep["tri_var_ratio"],
                "h_std": rep["h_std"],
                "file": Path(f).name,
            }
        )
        print(
            f"{spec['system_id']} {spec['hyper_kind']:6s} pure-interaction fraction {rep['pure_fraction']:.3f}  "
            f"Var(tri)/Var(rhs) {rep['tri_var_ratio']:.3f}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "identifiability.csv", index=False)
    try:
        (out / "identifiability.md").write_text(
            df.drop(columns=["file"]).to_markdown(index=False, floatfmt=".3f")
        )
    except ImportError:
        pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    systems = sorted(df["system"].unique())
    kinds = ["open", "mixed", "closed"]
    w = 0.25
    for q, kind in enumerate(kinds):
        sub = df[df["kind"] == kind].groupby("system")["pure_fraction"].mean().reindex(systems)
        ax.bar(np.arange(len(systems)) + (q - 1) * w, sub.values, w, label=kind)
    ax.set_xticks(np.arange(len(systems)), systems)
    ax.set_ylim(0, 1)
    ax.set_ylabel("pure-interaction fraction of h")
    ax.set_title("How much of each triadic term a pairwise model cannot mimic")
    ax.legend(fontsize=8, title="hyperedge type", title_fontsize=8)
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "identifiability.png", dpi=150, bbox_inches="tight")
    print("wrote", out / "identifiability.csv", fig_dir / "identifiability.png")


if __name__ == "__main__":
    main()
