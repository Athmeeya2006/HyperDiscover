#!/usr/bin/env python
"""Budgeted joint-kick design vs the exhaustive design (CLAUDE.md §18.12 G1; E4/E5 intervention counts).

    python scripts/active_kick_experiment.py --out paper/tables

Designs (intervene/active.py): exhaustive (all pairs, r0 readouts), screened (pairs of single-kick
influencers of a common target), screened+adaptive (extra readouts only on undecided kicks),
exhaustive with the SAME readout total as screened+adaptive spread uniformly (equal budget), and
random_same_pairs (as many joint kicks as screened, pairs uniformly at random: the selection baseline
reviewers expect; references/LITERATURE.md topic survey 3).
Truth-free detection (χ², Bonferroni over the exhaustive family). Writes active_kick.csv / .md.
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
    ap.add_argument("--Ns", nargs="+", type=int, default=[10, 20, 50])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--tau-dt", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--r0s", nargs="+", type=int, default=[4, 25])
    ap.add_argument("--max-per-target", nargs="+", type=int, default=[5, 8])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from hyperdiscover import benchmarks
    from hyperdiscover.intervene.active import ActiveKickConfig, active_joint_kicks

    rows = []
    for sid, kind, N, noise, r0_base, m in itertools.product(
        a.systems, a.kinds, a.Ns, a.noises, a.r0s, a.max_per_target
    ):
        b = benchmarks.load(
            sid, N=N, hyper_kind="open" if kind == "pairwise" else kind,
            pairwise_only=kind == "pairwise", seed=a.seed,
        )
        sysm, st = b.system, b.structure
        rng = np.random.default_rng([a.seed, N, int(sid[1:])])
        X = b.data.X_clean.reshape(-1, N, b.data.X_clean.shape[-1])
        xs = X[rng.choice(len(X), a.states, replace=False)]
        sig = float(b.data.X_clean[..., 0].std(axis=(0, 1)).mean())
        width = sysm.ic_box[1] - sysm.ic_box[0]
        true_h = {tuple(sorted(h)) for h in np.asarray(st.hyperedges).tolist()}
        all_t = list(itertools.combinations(range(N), 3))
        y = np.array([t in true_h for t in all_t])
        designs = {
            "exhaustive": dict(screen=False, adaptive=False),
            "screened": dict(screen=True, adaptive=False),
            "screened+adaptive": dict(screen=True, adaptive=True),
        }
        budget_ref = None
        n_screened = None
        extra = [("exhaustive_equal_budget", "equal"), ("random_same_pairs", "random")]
        for name, kw in list(designs.items()) + extra:
            r0 = r0_base
            if kw == "random":  # same number of joint kicks as screened, pairs chosen uniformly at random
                kw = dict(screen=False, adaptive=False, random_pairs=n_screened)
            elif kw == "equal":  # same readout total as screened+adaptive, spread over all pairs
                n_pairs = N * (N - 1) // 2
                r0 = max(1, int(budget_ref // (1 + N + n_pairs)))
                kw = dict(screen=False, adaptive=False)
            cfg = ActiveKickConfig(
                delta=a.delta, tau=a.tau_dt * sysm.default_dt, r0=r0, alpha=a.alpha,
                max_per_target=m, **kw
            )
            t0 = time.time()
            res = active_joint_kicks(sysm, xs, cfg, noise * sig / width, np.random.default_rng(a.seed))
            if name == "screened+adaptive":
                budget_ref = res["n_readouts"]
            if name == "screened":
                n_screened = len(res["pairs_kicked"])
            det = set(res["hyperedges"])
            s = np.array([res["hyper_scores"].get(t, 0.0) for t in all_t])
            kicked = {tuple(map(int, p)) for p in res["pairs_kicked"]}
            reach = np.mean([any(tuple(sorted(set(h) - {i})) in kicked for i in h) for h in true_h]) if true_h else np.nan
            row = dict(
                system=sid, kind=kind, N=N, noise=noise, design=name, r0=r0, m=m,
                pairs_kicked=len(res["pairs_kicked"]), pairs_exhaustive=res["n_pairs_exhaustive"],
                interventions=res["n_interventions"] * a.states, readouts=res["n_readouts"] * a.states,
                screen_recall=reach,
                hyper_auroc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                hyper_tp=len(det & true_h), hyper_fp=len(det - true_h), hyper_fn=len(true_h - det),
                seconds=round(time.time() - t0, 2),
            )
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "active_kick.csv", index=False)
    (out / "active_kick.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "active_kick.csv")


if __name__ == "__main__":
    main()
