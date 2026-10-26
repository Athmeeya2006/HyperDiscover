#!/usr/bin/env python
"""W11 (CLAUDE.md §18.15): joint kicks when couplings act with delays (data/delays.py).

    python scripts/kick_delay_experiment.py --out paper/tables

Per edge / hyperedge delay drawn from 0..max_delay saved steps. From S operating states (constant history),
every node and node pair is kicked at t = 0 (only the current state changes; the delayed arguments still see the
pre-kick history), and node states are read at τ and 2τ (Richardson combination, as in intervene/joint_kick.py).
A kick reaches a neighbour only after the coupling delay, so the readout time must exceed the delay. Reported:
hyperedge AUROC and influence (edge ∪ co-membership) AUROC vs τ / max_delay, noiseless, with and without
delays, and whether delays create false non-additive responses on pairwise-only systems (max score ratio).
Writes kick_delays.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S4", "S2", "S1"])
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--max-delays", nargs="+", type=int, default=[0, 2, 5])
    ap.add_argument("--taus", nargs="+", type=int, default=[1, 3, 6, 12], help="readout time, saved steps")
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from hyperdiscover import benchmarks
    from hyperdiscover.data.delays import simulate_delayed
    from hyperdiscover.intervene.joint_kick import hyperedge_scores, mixed_difference_scores

    rows = []
    for sid, kind, D in itertools.product(a.systems, ["open", "pairwise"], a.max_delays):
        b = benchmarks.load(sid, N=a.N, hyper_kind="open", pairwise_only=kind == "pairwise", seed=a.seed)
        s, st, N = b.system, b.structure, a.N
        rng = np.random.default_rng([a.seed, int(sid[1:]), D])
        plag = rng.integers(0, D + 1, len(s._pairs))
        hl = {tuple(sorted(map(int, t))): int(rng.integers(0, D + 1)) for t in s._tris}
        tlag = np.array([hl[tuple(sorted(map(int, t)))] for t in s._tris], dtype=int)
        Xc = b.data.X_clean.reshape(-1, N, 1)
        xs = Xc[rng.choice(len(Xc), a.states, replace=False)]
        width = s.ic_box[1] - s.ic_box[0]
        d = a.delta * width
        pairs = np.array(list(itertools.combinations(range(N), 2)))
        kicks = [()] + [(v,) for v in range(N)] + [tuple(p) for p in pairs]
        L = D + 1
        hist = np.repeat(np.repeat(xs[:, None, None], len(kicks), 1), L, 2)  # [S, K, L, N, 1]
        for q, nodes in enumerate(kicks):
            for v in nodes:
                hist[:, q, -1, v] += d  # kick only the current state
        Tmax = 2 * max(a.taus)
        traj = simulate_delayed(s, plag, tlag, hist.reshape(-1, L, N, 1), Tmax)  # [S·K, Tmax+1, N, 1]
        traj = traj.reshape(a.states, len(kicks), Tmax + 1, N)
        true_h = {tuple(h) for h in st.hyperedges.tolist()}
        A = st.adjacency() > 0
        infl = A.copy()
        for h in st.hyperedges.tolist():
            for u, v in itertools.permutations(h, 2):
                infl[u, v] = True
        off = ~np.eye(N, dtype=bool)
        for tau in a.taus:
            eff = 2 * traj[:, :, tau] - 0.5 * traj[:, :, 2 * tau]  # Richardson readout [S, K, N]
            resp = {"base": eff[:, 0], "single": eff[:, 1 : 1 + N], "joint": eff[:, 1 + N :], "pairs": pairs}
            ps, ts = mixed_difference_scores(resp, a.delta, tau * s.default_dt, width)
            hs = hyperedge_scores(ts, pairs)
            keys = list(hs)
            y = np.array([k in true_h for k in keys])
            sc = np.array([hs[k] for k in keys])
            rows.append(dict(
                system=sid, kind=kind, max_delay=D, tau=tau, tau_over_delay=tau / D if D else np.inf,
                infl_auroc=roc_auc_score(infl[off], ps[off]) if infl[off].any() else np.nan,
                hyper_auroc=roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan,
                # pairwise-only: largest mixed response relative to the largest single-kick response
                max_tri_over_pair=float(sc.max() / max(ps.max(), 1e-300)),
            ))
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()),
                  flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "kick_delays.csv", index=False)
    (out / "kick_delays.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "kick_delays.csv")


if __name__ == "__main__":
    main()
