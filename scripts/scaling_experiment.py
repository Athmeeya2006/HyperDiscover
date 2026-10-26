#!/usr/bin/env python
"""E8 / W6 (CLAUDE.md §18.5, §18.15): runtime, memory and accuracy vs N, ours vs THIS.

    python scripts/scaling_experiment.py --Ns 20 50 100 200 --device cuda --out paper/tables

Recipe data (sparse ICs, M=400, T=25), open hyperedges (density 0.3), 1 % noise. HyperDiscover: structure only,
default config; wall-clock per stage (screen / alternating), peak memory (CUDA max allocated, or CPU RSS peak),
screening recall, edge/hyperedge F1 and AUROC. THIS (GPL port, tuned λ): wall-clock, its library size
(1 + N + N(N+1)/2 monomials at degree 2) and the same metrics; skipped above --this-max-n, or reported as
failed on MemoryError. Writes scaling.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
import resource
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="S4")
    ap.add_argument("--Ns", nargs="+", type=int, default=[20, 50, 100, 200])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--this-max-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score

    from hyperdiscover import HyperDiscover, benchmarks
    from hyperdiscover.baselines import directed_to_undirected
    from hyperdiscover.baselines.this_reimpl import this_reconstruct  # GPL-3.0 port (D-040)
    from hyperdiscover.baselines.tuning import tune_this
    from hyperdiscover.data.derivatives import estimate_derivatives

    def f1(p, t):
        return 2 * len(p & t) / max(len(p) + len(t), 1)

    rows = []
    for N in a.Ns:
        b = benchmarks.load(a.system, N=N, hyper_kind="open", noise=a.noise, M=400, T=25, ic_mode="sparse",
                            seed=a.seed)
        st = b.structure
        trueE = {tuple(e) for e in st.edges.tolist()}
        trueH = {tuple(h) for h in st.hyperedges.tolist()}
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        hd = HyperDiscover(config={"device": a.device}, periodic=a.system == "S1", distill_engine="none",
                           seed=a.seed, verbose=False)
        hd.fit(b.data.X, b.data.t)
        wall = time.time() - t0
        res = hd.structure_result_
        mem = (torch.cuda.max_memory_allocated() / 2**20 if a.device == "cuda"
               else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        mode = hd.config["readout"]["undirected"]
        se = {tuple(sorted((i, j))): s for i, j, s in res.edges(-1.0, mode)}
        sh = {tuple(sorted((i, j, k))): s for i, j, k, s in res.hyperedges(-1.0, mode)}
        allp = list(itertools.combinations(range(N), 2))
        yE = np.array([p in trueE for p in allp])
        cand_t = list(sh)
        yH = np.array([t in trueH for t in cand_t])
        E = {tuple(sorted((i, j))) for i, j, _ in hd.pairwise_edges_}
        Hs = {tuple(sorted((i, j, k))) for i, j, k, _ in hd.hyperedges_}
        from hyperdiscover.screen.triadic import triadic_screening_recall

        rows.append(dict(N=N, method="HyperDiscover", seconds=wall,
                         seconds_screen=res.timings.get("screen"), seconds_fit=res.timings.get("alternating"),
                         peak_mem_mb=mem, n_tri_candidates=len(res.tri_index_all),
                         tri_screen_recall=triadic_screening_recall(res.tri_index_all, st.tri_index),
                         edge_f1=f1(E, trueE), hyper_f1=f1(Hs, trueH),
                         edge_auroc=roc_auc_score(yE, [se.get(p, 0) for p in allp]),
                         hyper_auroc_candidates=roc_auc_score(yH, [sh[t] for t in cand_t]) if 0 < yH.sum() < len(yH) else np.nan))
        print(rows[-1], flush=True)
        if N <= a.this_max_n:
            X = b.data.X
            dX = estimate_derivatives(X, b.data.t, "savgol", window=9, polyorder=3)[:, 3:-3]
            X = X[:, 3:-3]
            t0 = time.time()
            try:
                nv = max(X.shape[0] // 5, 1)
                lam, _ = tune_this(X[:-nv], dX[:-nv], X[-nv:], dX[-nv:], x0=None)
                r = this_reconstruct(X, dX, lam=lam, x0=None)
                pu, tu = directed_to_undirected(r["pair_scores"], r.get("tri_scores", {}))
                E2 = {k for k, v in pu.items() if v > 1e-12}
                H2 = {k for k, v in tu.items() if v > 1e-12}
                rows.append(dict(N=N, method="THIS", seconds=time.time() - t0,
                                 n_library=1 + N + N * (N + 1) // 2, edge_f1=f1(E2, trueE),
                                 hyper_f1=f1(H2, trueH),
                                 edge_auroc=roc_auc_score(yE, [pu.get(p, 0) for p in allp])))
            except MemoryError as e:
                rows.append(dict(N=N, method="THIS", failed=repr(e), seconds=time.time() - t0))
            print(rows[-1], flush=True)
        else:
            rows.append(dict(N=N, method="THIS", failed=f"skipped (N > {a.this_max_n})"))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scaling.csv", index=False)
    (out / "scaling.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "scaling.csv")


if __name__ == "__main__":
    main()
