#!/usr/bin/env python
"""E11 / W10 (CLAUDE.md §18.5, §18.15): EXPLORATORY case study on the Benincà et al. plankton mesocosm.

    python scripts/case_study_plankton.py --out paper/tables

No ground truth exists; the shared-f,g,h model class is certainly an idealisation for different species (stated
in the output). What is measured:
  * held-out forecast skill: the last ``--test-frac`` of the series is never used for fitting; from each held-out
    window start the discovered model is integrated for ``--horizon`` steps and compared with the data
    (nRMSE vs horizon), for the full model (max_order 3) and the pairwise-only model (max_order 2), plus a
    persistence baseline;
  * the discovered structure (edges, hyperedges with scores) and equations, for qualitative discussion only.
Preprocessing: fourth-root transform of abundances (as in Benincà et al. 2008), each variable scaled to [0, 1]
on the training part; the series is cut into overlapping windows of T points (the pipeline's M × T input).
Writes case_study_plankton.json / .md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/case_studies/plankton_interp.csv")
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--drop", nargs="*", default=["Day Number"], help="columns that are not nodes")
    ap.add_argument("--engine", default="library")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd

    from hyperdiscover import HyperDiscover

    df = pd.read_csv(a.data)
    day = df["Day Number"].to_numpy(float)
    nodes = [c for c in df.columns if c not in a.drop]
    Y = np.sqrt(np.sqrt(np.clip(df[nodes].to_numpy(float), 0, None)))  # fourth root
    n_tr = int(len(Y) * (1 - a.test_frac))
    lo, hi = Y[:n_tr].min(0), Y[:n_tr].max(0)
    Z = (Y - lo) / np.where(hi > lo, hi - lo, 1.0)
    dt = float(np.median(np.diff(day))) / 10.0  # time unit: 10 days
    starts = list(range(0, n_tr - a.T + 1, a.stride))
    X = np.stack([Z[s : s + a.T] for s in starts])[..., None]  # [M, T, N, 1]
    t = np.arange(a.T) * dt
    out = {"nodes": nodes, "n_train_points": n_tr, "n_test_points": len(Z) - n_tr, "M": len(starts),
           "T": a.T, "caveat": "exploratory; shared f, g, h across species is an idealisation; no ground truth"}
    test_starts = list(range(n_tr, len(Z) - a.horizon, 2))
    truth = np.stack([Z[s : s + a.horizon + 1] for s in test_starts])  # [K, H+1, N]
    scale = Z[n_tr:].std(0) + 1e-9
    out["persistence_nrmse"] = [
        float(np.sqrt(np.mean(((truth[:, h] - truth[:, 0]) / scale) ** 2))) for h in range(a.horizon + 1)
    ]
    for order in (3, 2):
        hd = HyperDiscover(max_order=order, distill_engine=a.engine, seed=a.seed, verbose=False)
        hd.fit(X, t)
        x0 = truth[:, 0, :, None]
        th = np.arange(a.horizon + 1) * dt
        key = "full" if order == 3 else "pairwise_only"
        try:
            pred = hd.simulate(x0, th)[..., 0] if hd.equations_ else hd.simulate_neural(x0, th)[..., 0]
            nr = [float(np.sqrt(np.nanmean(((pred[:, h] - truth[:, h]) / scale) ** 2)))
                  for h in range(a.horizon + 1)]
        except Exception as e:  # unstable discovered model etc.
            nr = repr(e)
        out[key] = {
            "forecast_nrmse": nr,
            "edges": [(nodes[i], nodes[j], float(s)) for i, j, s in hd.pairwise_edges_],
            "hyperedges": [(nodes[i], nodes[j], nodes[k], float(s)) for i, j, k, s in hd.hyperedges_],
            "equations": {k: str(v) for k, v in (hd.equations_ or {}).items()},
        }
        print(key, "edges", len(out[key]["edges"]), "hyperedges", len(out[key]["hyperedges"]),
              "forecast nRMSE", nr if isinstance(nr, str) else [round(v, 3) for v in nr], flush=True)
    print("persistence", [round(v, 3) for v in out["persistence_nrmse"]])
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "case_study_plankton.json").write_text(json.dumps(out, indent=2))
    md = [f"# Plankton case study (exploratory)\n\n{out['caveat']}\n",
          "| horizon | persistence | full | pairwise-only |", "|---|---|---|---|"]
    for h in range(a.horizon + 1):
        def g(k, h=h):
            v = out[k]["forecast_nrmse"]
            return f"{v[h]:.3f}" if isinstance(v, list) else "failed"
        md.append(f"| {h} | {out['persistence_nrmse'][h]:.3f} | {g('full')} | {g('pairwise_only')} |")
    md.append("\nHyperedges (full model): " + "; ".join(f"{a_}–{b_}–{c_} ({s:.2f})" for a_, b_, c_, s in out["full"]["hyperedges"]))
    (o / "case_study_plankton.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
