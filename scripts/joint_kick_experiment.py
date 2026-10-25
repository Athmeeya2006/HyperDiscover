#!/usr/bin/env python
"""Hyperedge and edge detection from single and joint two-node kicks (DECISIONS.md D-030). No training.

    python scripts/joint_kick_experiment.py --out paper/tables

For every system / hyperedge kind (incl. pairwise-only, for false positives) / N / noise level, kicks
every node and every node pair from S operating states (sampled from the benchmark trajectories),
reads the states after τ, and scores
  edges      by the single-kick response D_ij,
  hyperedges by the mixed difference D_ijk (zero for ANY pairwise coupling g, see intervene/joint_kick.py).
Detection is truth-free: χ² test against readout noise estimated from replicate unkicked readouts,
Bonferroni-corrected at level alpha over all tests of a kind.
Writes joint_kick.csv / joint_kick.md.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--kinds", nargs="+", default=["open", "closed", "pairwise"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.01, 0.05])
    ap.add_argument(
        "--taus", nargs="+", type=float, default=[1.0, 4.0], help="in units of system dt"
    )
    ap.add_argument("--delta", type=float, default=0.3, help="kick, units of IC-box width")
    ap.add_argument("--states", type=int, default=20, help="operating states S")
    ap.add_argument(
        "--state-source",
        choices=["trajectory", "sparse_ic"],
        default="trajectory",
        help="operating states sampled from the benchmark trajectories or sparse initial conditions",
    )
    ap.add_argument("--repeats", nargs="+", type=int, default=[1, 25], help="readouts averaged")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument(
        "--richardson", nargs="+", type=int, default=[1], help="1: cancel O(τ²) indirect paths"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    from hyperdiscover import benchmarks
    from hyperdiscover.data.simulate import sparse_ic
    from hyperdiscover.intervene.joint_kick import (
        calibrated_pvalues,
        estimate_readout_noise,
        hyperedge_scores,
        kick_responses,
        mixed_difference_scores,
    )

    rows = []
    for sid, kind, N in itertools.product(a.systems, a.kinds, a.Ns):
        b = benchmarks.load(
            sid,
            N=N,
            hyper_kind="open" if kind == "pairwise" else kind,
            pairwise_only=kind == "pairwise",
            seed=a.seed,
        )
        st, sysm = b.structure, b.system
        rng = np.random.default_rng([a.seed, N, int(sid[1:])])
        X = b.data.X_clean.reshape(-1, N, b.data.X_clean.shape[-1])
        if a.state_source == "sparse_ic":  # low-activity states, away from saturation
            xs = sparse_ic(sysm, a.states, rng)
        else:
            xs = X[rng.choice(len(X), a.states, replace=False)]
        sig = float(b.data.X_clean[..., 0].std(axis=(0, 1)).mean())  # mean per-node signal std
        width = sysm.ic_box[1] - sysm.ic_box[0]
        A = st.adjacency() > 0
        # single kicks see every DIRECT influence: pairwise edges and hyperedge co-membership
        infl = A.copy()
        for h in np.asarray(st.hyperedges).tolist():
            for u, v in itertools.permutations(h, 2):
                infl[u, v] = True
        off = ~np.eye(N, dtype=bool)
        true_h = {tuple(sorted(h)) for h in np.asarray(st.hyperedges).tolist()}
        n_tri_tests = N * (N - 1) * (N - 2) // 2
        for noise, tau_dt, R, rich in itertools.product(a.noises, a.taus, a.repeats, a.richardson):
            if noise == 0 and R > 1:
                continue
            tau = tau_dt * sysm.default_dt
            t0 = time.time()
            resp = kick_responses(
                sysm, xs, a.delta, tau, noise * sig / width, rng, n_repeats=R, richardson=bool(rich)
            )
            ps, ts = mixed_difference_scores(resp, a.delta, tau, width)
            hs = hyperedge_scores(ts, resp["pairs"])
            keys = list(hs)
            y_h = np.array([k in true_h for k in keys])
            s_h = np.array([hs[k] for k in keys])
            row = {
                "system": sid,
                "kind": kind,
                "N": N,
                "noise": noise,
                "tau_dt": tau_dt,
                "repeats": R,
                "richardson": rich,
                "state_source": a.state_source,
                "infl_auroc": roc_auc_score(infl[off], ps[off]),
                "infl_auprc": average_precision_score(infl[off], ps[off]),
                "pair_edge_auroc": roc_auc_score(A[off], ps[off]),
                "hyper_auroc": roc_auc_score(y_h, s_h) if 0 < y_h.sum() < len(y_h) else np.nan,
                "hyper_auprc": average_precision_score(y_h, s_h) if y_h.any() else np.nan,
                "n_hyper_true": int(y_h.sum()),
                "interventions": a.states * (1 + N + N * (N - 1) // 2),
            }
            if noise > 0:
                sigma = estimate_readout_noise(resp["base"], resp["base_rep"])
                pp, pt = calibrated_pvalues(resp, sigma)
                e_on = pp < a.alpha / (N * (N - 1))
                h_on = hyperedge_scores(-pt, resp["pairs"])  # max(−p) = −min p
                h_sel = {k for k, v in h_on.items() if -v < a.alpha / n_tri_tests}
                row.update(
                    infl_tp=int((e_on & infl)[off].sum()),
                    infl_fp=int((e_on & ~infl)[off].sum()),
                    infl_fn=int((~e_on & infl)[off].sum()),
                    hyper_tp=len(h_sel & true_h),
                    hyper_fp=len(h_sel - true_h),
                    hyper_fn=len(true_h - h_sel),
                )
            # noiseless: no noise model, so no truth-free threshold; ranking metrics only
            row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            print(
                f"{sid} {kind:8s} N={N:3d} noise={noise:.2f} tau={tau_dt}dt R={R:2d} rich={rich}  "
                f"influence AUROC {row['infl_auroc']:.3f} TP/FP/FN {row.get('infl_tp')}/"
                f"{row.get('infl_fp')}/{row.get('infl_fn')}  hyper AUROC {row['hyper_auroc']:.3f} "
                f"TP/FP/FN {row.get('hyper_tp')}/{row.get('hyper_fp')}/{row.get('hyper_fn')}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "joint_kick" if a.state_source == "trajectory" else f"joint_kick_{a.state_source}"
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / f"{stem}.csv")


if __name__ == "__main__":
    main()
