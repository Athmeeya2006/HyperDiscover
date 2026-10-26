#!/usr/bin/env python
"""E5 / W4 (CLAUDE.md §18.5, §18.15): passive recording vs joint kicks at the SAME measurement budget.

    python scripts/equal_budget_experiment.py --out paper/tables

Budget unit: one full-network snapshot (all N node states read once). Passive: M trajectories × T = 25 saved
points (recipe data: sparse ICs), fitted with HyperDiscover (structure only, default config) → M = budget / 25.
Kicks: exhaustive joint-kick design from S operating states with R averaged readouts per intervention, read at
τ and 2τ (Richardson) → snapshots = S · (1 + N + N(N−1)/2 + 1 replicate base) · 2 · R; S and R chosen to fit the
budget (R as large as allowed with S = --states). Measurement noise identical (fraction of signal std).
Scores: hyperedge F1 (truth-free thresholds: χ² for kicks, the pipeline's readout for passive) and AUROC.
Writes equal_budget.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def kick_snapshots(N: int, S: int, R: int) -> int:
    return S * (2 + N + N * (N - 1) // 2) * 2 * R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--kinds", nargs="+", default=["open", "pairwise"])
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--budgets", nargs="+", type=int, default=[2500, 10000, 40000])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from hyperdiscover import HyperDiscover, benchmarks
    from hyperdiscover.intervene.joint_kick import (
        calibrated_pvalues,
        estimate_readout_noise,
        hyperedge_scores,
        kick_responses,
    )

    def f1(p, t):
        return 2 * len(p & t) / max(len(p) + len(t), 1)

    rows = []
    for sid, kind, budget, noise in itertools.product(a.systems, a.kinds, a.budgets, a.noises):
        T = 25
        M = max(budget // T, 4)
        b = benchmarks.load(sid, N=a.N, hyper_kind="open" if kind == "pairwise" else kind,
                            pairwise_only=kind == "pairwise", noise=noise, M=M, T=T, ic_mode="sparse",
                            seed=a.seed)
        true_h = {tuple(h) for h in np.asarray(b.structure.hyperedges).tolist()}
        allt = list(itertools.combinations(range(a.N), 3))
        y = np.array([t in true_h for t in allt])
        # passive
        hd = HyperDiscover(config={"device": a.device}, periodic=sid == "S1", distill_engine="none",
                           seed=a.seed, verbose=False)
        hd.fit(b.data.X, b.data.t)
        sc = {tuple(sorted((i, j, k))): s for i, j, k, s in hd.hyperedges_}
        und = hd.structure_result_.hyperedges(-1.0, hd.config["readout"]["undirected"])
        allsc = {tuple(sorted((i, j, k))): s for i, j, k, s in und}
        s_p = np.array([allsc.get(t, 0.0) for t in allt])
        rows.append(dict(system=sid, kind=kind, budget=budget, noise=noise, design="passive", M=M, R=None,
                         snapshots=M * T, hyper_f1=f1(set(sc), true_h) if (true_h or sc) else np.nan,
                         hyper_fp=len(set(sc) - true_h),
                         hyper_auroc=roc_auc_score(y, s_p) if 0 < y.sum() < len(y) else np.nan))
        # kicks with the same budget
        R = budget // kick_snapshots(a.N, a.states, 1)
        if R >= 1:
            rng = np.random.default_rng([a.seed, a.N, int(sid[1:])])
            Xc = b.data.X_clean.reshape(-1, a.N, 1)
            xs = Xc[rng.choice(len(Xc), a.states, replace=False)]
            sig = float(b.data.X_clean[..., 0].std(axis=(0, 1)).mean())
            width = b.system.ic_box[1] - b.system.ic_box[0]
            resp = kick_responses(b.system, xs, a.delta, b.system.default_dt, noise * sig / width, rng,
                                  n_repeats=R)
            sigma = estimate_readout_noise(resp["base"], resp["base_rep"])
            _, pt = calibrated_pvalues(resp, max(sigma, 1e-12))
            hs = hyperedge_scores(-pt, resp["pairs"])
            n_tests = a.N * (a.N - 1) * (a.N - 2) // 2
            det = {k for k, v in hs.items() if -v < a.alpha / n_tests}
            s_k = np.array([-np.log10(max(-hs[t], 1e-300)) for t in allt])
            rows.append(dict(system=sid, kind=kind, budget=budget, noise=noise, design="joint_kicks", M=None,
                             R=R, snapshots=kick_snapshots(a.N, a.states, R),
                             hyper_f1=f1(det, true_h) if (true_h or det) else np.nan,
                             hyper_fp=len(det - true_h),
                             hyper_auroc=roc_auc_score(y, s_k) if 0 < y.sum() < len(y) else np.nan))
        for r in rows[-2:]:
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "equal_budget.csv", index=False)
    (out / "equal_budget.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "equal_budget.csv")


if __name__ == "__main__":
    main()
