#!/usr/bin/env python
"""G4b (CLAUDE.md §18.12): recover edges, hyperedges AND their coupling delays, with known/discovered f, g, h.

    python scripts/delay_experiment.py --out paper/tables

Ground truth: the benchmark system with every edge (and hyperedge) acting with a random delay of 0..``--max-trues``
saved steps (data/delays.py, method of steps). Data: M short trajectories from sparse ICs with a constant
history, measurement noise, Savitzky–Golay derivatives. Inference: one column per (candidate, lag) for lags
0..max_true + 2; per-node non-negative BIC LASSO (train/alternating.select_weights); an edge is present if any
of its lags is selected and its delay is the lag with the largest weight; hyperedge readout = support at
≥ 2 of its 3 focal nodes. Compared with the same selection restricted to lag 0 (a model that ignores delays).
Writes delays.csv / delays.md.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S4", "S2", "S1", "S3", "S5"])
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--max-trues", nargs="+", type=int, default=[2, 5, 10],
                    help="true delays drawn from 0..max_true saved steps; candidate lags 0..max_true + 2")
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.01])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd

    from hyperdiscover import benchmarks
    from hyperdiscover.data.delays import lagged_design, simulate_delayed
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.data.observe import add_noise
    from hyperdiscover.data.simulate import TrajectoryData, sparse_ic
    from hyperdiscover.train.alternating import AlternatingConfig, select_weights
    from hyperdiscover.train.windowed import all_candidates

    rows = []
    for sid, noise, max_true in itertools.product(a.systems, a.noises, a.max_trues):
        max_lag_c = max_true + 2
        rng = np.random.default_rng([a.seed, int(sid[1:])])
        b = benchmarks.load(sid, N=a.N, hyper_kind="open", M=2, T=2, seed=a.seed)
        s = b.system
        plag = rng.integers(0, max_true + 1, len(s._pairs))
        tlag = rng.integers(0, max_true + 1, len(s._tris))
        # one delay per undirected hyperedge / edge direction: focal triples of one hyperedge share it
        hyp_lag = {tuple(sorted(map(int, t))): int(lg) for t, lg in zip(s._tris, tlag)}
        tlag = np.array([hyp_lag[tuple(sorted(map(int, t)))] for t in s._tris])
        x0 = sparse_ic(s, a.M, rng)
        hist = np.repeat(x0[:, None], max_true + 1, 1)
        Xc = simulate_delayed(s, plag, tlag, hist, a.T)
        t = np.arange(Xc.shape[1]) * s.default_dt
        data = add_noise(TrajectoryData(Xc.copy(), t, Xc, None, {}), noise, rng)
        dX = estimate_derivatives(data.X, t, "savgol", window=7, polyorder=3)
        pairs, tris = all_candidates(a.N)
        true_edge_lag = {(int(i), int(j)): int(lg) for (i, j), lg in zip(s._pairs, plag)}
        true_H = {tuple(h) for h in np.asarray(b.structure.hyperedges).tolist()}
        true_E = {tuple(e) for e in np.asarray(b.structure.edges).tolist()}
        for max_lag, label in [(max_lag_c, "lagged"), (0, "no_delay_model")]:
            base, G, H, gcol, hcol = lagged_design(s, data.X, pairs, tris, max_lag)
            dS = dX[:, max_lag:, :, 0].reshape(-1, a.N)
            sel = rng.choice(len(dS), min(4000, len(dS)), replace=False)
            wp, wt = select_weights(dS[sel] - base[sel], G[sel], H[sel], pairs[gcol[:, 0], 0],
                                    tris[hcol[:, 0], 0], a.N, AlternatingConfig(), True)
            # edges: any lag selected; delay = argmax weight over lags
            best: dict = {}
            for q in np.where(wp > 0)[0]:
                e, lg = int(gcol[q, 0]), int(gcol[q, 1])
                key = (int(pairs[e, 0]), int(pairs[e, 1]))
                if key not in best or wp[q] > best[key][1]:
                    best[key] = (lg, wp[q])
            E = {tuple(sorted(k)) for k in best}
            lag_ok = [best[k][0] == true_edge_lag[k] for k in best if k in true_edge_lag]
            cnt: dict = {}
            for q in np.where(wt > 0)[0]:
                key = tuple(sorted(map(int, tris[hcol[q, 0]])))
                cnt.setdefault(key, set()).add(int(tris[hcol[q, 0], 0]))
            Hs = {k for k, v in cnt.items() if len(v) >= 2}

            def f1(p, tr):
                return 2 * len(p & tr) / max(len(p) + len(tr), 1)

            row = dict(system=sid, noise=noise, max_true=max_true, model=label, edge_f1=f1(E, true_E), hyper_f1=f1(Hs, true_H),
                       edge_fp=len(E - true_E), hyper_fp=len(Hs - true_H),
                       lag_accuracy=float(np.mean(lag_ok)) if lag_ok else np.nan,
                       n_edges=len(true_E), n_hyper=len(true_H))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "delays.csv", index=False)
    (out / "delays.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "delays.csv")


if __name__ == "__main__":
    main()
