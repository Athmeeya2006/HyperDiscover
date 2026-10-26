#!/usr/bin/env python
"""G4a (CLAUDE.md §18.12): detect a mid-recording rewiring of edges and hyperedges, with known/discovered f, g, h.

    python scripts/timevarying_experiment.py --out paper/tables

Data: system with structure A for the first half of every trajectory and structure B (a fraction
``--rewire`` of hyperedges and edges replaced at random) for the second half; sparse ICs, random kicks every
``--kick-every`` saved steps keep the data informative; measurement noise as in the benchmark. Structure is
selected per time window with f, g, h fixed (train/windowed.py), using (a) the true functions (oracle) and
(b) the equations of a discovered run (``--equations``; skipped if absent). Reported per window: edge and
hyperedge F1 against the structure active in that window, support flips vs the previous window, and
whether the largest flip count falls at the true switch. Writes timevarying.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def rewire(structure, frac: float, rng):
    """Replace a fraction of hyperedges and of edges by random new ones (same counts)."""
    from hyperdiscover.systems.base import Structure

    N = structure.n_nodes
    E = [tuple(e) for e in np.asarray(structure.edges).tolist()]
    H = [tuple(h) for h in np.asarray(structure.hyperedges).tolist()]
    allE = set(itertools.combinations(range(N), 2))
    allH = set(itertools.combinations(range(N), 3))

    def swap(items, pool, k):
        keep = [items[q] for q in rng.permutation(len(items))[k:]]
        free = sorted(pool - set(items))
        new = [free[q] for q in rng.choice(len(free), k, replace=False)]
        return keep + new

    E2 = swap(E, allE, int(round(frac * len(E))))
    H2 = swap(H, allH, int(round(frac * len(H))))
    return Structure(N, np.array(sorted(E2), dtype=np.int64).reshape(-1, 2),
                     np.array(sorted(H2), dtype=np.int64).reshape(-1, 3))


def f1(pred: set, true: set) -> float:
    tp = len(pred & true)
    return 2 * tp / max(len(pred) + len(true), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S4", "S2", "S1"])
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--T", type=int, default=200, help="saved steps per trajectory (switch at T/2)")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--rewire", type=float, default=0.5)
    ap.add_argument("--kick-every", type=int, default=25)
    ap.add_argument("--kick-std", type=float, default=0.2)
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.01])
    ap.add_argument("--designs", nargs="+", default=["continuous", "restart"],
                    help="continuous: one recording, rewired mid-way; restart: new sparse-IC runs after the switch")
    ap.add_argument("--flip-tol", type=int, default=3, help="a window 'changes' if > this many support flips")
    ap.add_argument("--equations", default="configs/transfer_sources/recipe-n20-v3",
                    help="dir with <S?>_open_N20_noise0_* runs whose equations.json is used as (b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    import sympy as sp

    from hyperdiscover import benchmarks
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.data.observe import add_noise
    from hyperdiscover.data.simulate import TrajectoryData, simulate_system, sparse_ic
    from hyperdiscover.distill.refit import SymbolicModel
    from hyperdiscover.systems import make_system
    from hyperdiscover.train.windowed import all_candidates, windowed_structure

    rows = []
    for sid, noise, design in itertools.product(a.systems, a.noises, a.designs):
        rng = np.random.default_rng([a.seed, int(sid[1:])])
        b = benchmarks.load(sid, N=a.N, hyper_kind="open", M=2, T=2, seed=a.seed)
        stA = b.structure
        stB = rewire(stA, a.rewire, rng)
        kw = {"rng": np.random.default_rng(a.seed)} if sid == "S1" else {}
        sysA = make_system(sid, stA, b.system.params, **kw)
        sysB = make_system(sid, stB, b.system.params, **kw)
        if sid == "S1":  # same natural frequencies before and after the switch
            sysB.c = sysA.c
        half = a.T // 2
        d1 = simulate_system(sysA, a.M, half, None, rng, x0=sparse_ic(sysA, a.M, rng),
                             kick_std=a.kick_std, kick_every=a.kick_every)
        x0B = d1.X_clean[:, -1] if design == "continuous" else sparse_ic(sysB, a.M, rng)
        d2 = simulate_system(sysB, a.M, a.T - half, None, rng, x0=x0B,
                             kick_std=a.kick_std, kick_every=a.kick_every)
        # keep d2's first point so its kick times stay aligned with the derivative segments below
        # (continuous: index half repeats the last state of A, as a new segment)
        Xc = np.concatenate([d1.X_clean, d2.X_clean], 1)
        t = np.arange(Xc.shape[1]) * sysA.default_dt
        data = add_noise(TrajectoryData(Xc.copy(), t, Xc, None, {}), noise, rng)
        seg = sorted(set(range(0, half, a.kick_every)) | set(range(half, a.T, a.kick_every)))
        dX = estimate_derivatives(data.X, t, "savgol", segment_starts=seg, window=7, polyorder=3)
        pairs, tris = all_candidates(a.N)
        models = {"oracle": sysA}
        eq_runs = sorted(Path(a.equations).glob(f"{sid}_open_N20_noise0_*/seed0/equations.json"))
        if eq_runs:
            eq = json.loads(eq_runs[0].read_text())
            exprs = {k: sp.sympify(v["str"]) for k, v in eq.items()}
            models["discovered"] = SymbolicModel(stA, exprs, sysA.c if sysA.has_node_constants else None,
                                                 sid == "S1")
        trueE = {0: {tuple(e) for e in np.asarray(stA.edges).tolist()},
                 1: {tuple(e) for e in np.asarray(stB.edges).tolist()}}
        trueH = {0: {tuple(h) for h in np.asarray(stA.hyperedges).tolist()},
                 1: {tuple(h) for h in np.asarray(stB.hyperedges).tolist()}}
        for mname, model in models.items():
            res = windowed_structure(data.X, dX, model, a.window, pairs, tris,
                                     rng=np.random.default_rng(a.seed))
            big = np.where(res["changes"] > a.flip_tol)[0]
            first = int(big[0]) if len(big) else -1  # first change: between windows first and first + 1
            switch_win = half // a.window - 1  # true change between window switch_win and switch_win + 1
            for q, s in enumerate(res["starts"]):
                phase = 0 if s + a.window <= half else (1 if s >= half else -1)
                wp, wt = res["w_pair"][q], res["w_tri"][q]
                E = {tuple(sorted(map(int, pairs[e]))) for e in np.where(wp > 0)[0]}
                # undirected hyperedge = supported at ≥ 2 of its 3 focal nodes (median readout, D-036)
                cnt: dict = {}
                for h in np.where(wt > 0)[0]:
                    key = tuple(sorted(map(int, tris[h])))
                    cnt[key] = cnt.get(key, 0) + 1
                Hs = {k for k, v in cnt.items() if v >= 2}
                rows.append(dict(
                    system=sid, noise=noise, design=design, model=mname, window_start=int(s), phase=phase,
                    edge_f1=f1(E, trueE[phase]) if phase >= 0 else np.nan,
                    hyper_f1=f1(Hs, trueH[phase]) if phase >= 0 else np.nan,
                    flips_prev=int(res["changes"][q - 1]) if q > 0 else 0,
                    switch_detected=bool(first == switch_win),
                ))
            r = [x for x in rows if x["system"] == sid and x["noise"] == noise and x["model"] == mname
                 and x["design"] == design]
            print(f"{sid} noise={noise} {design} {mname}: edge F1 {np.nanmean([x['edge_f1'] for x in r]):.3f} "
                  f"hyper F1 {np.nanmean([x['hyper_f1'] for x in r]):.3f} flips {res['changes'].tolist()} "
                  f"switch detected {r[0]['switch_detected']}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "timevarying.csv", index=False)
    (out / "timevarying.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "timevarying.csv")


if __name__ == "__main__":
    main()
