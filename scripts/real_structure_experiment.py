#!/usr/bin/env python
"""E10 / W5 (CLAUDE.md §18.5, §18.15): real higher-order structures, simulated dynamics.

    python scripts/real_structure_experiment.py --out paper/tables

Structures from XGI (names verified against the xgi-data index on 2026-10-26): contact-high-school,
contact-primary-school, hospital-lyon (SocioPatterns face-to-face contacts) with S4 (SIS); plant-pollinator-mpl-014
with S2 and S5; email-enron with S3 (coauth-dblp is too large to load). A connected induced sub-hypergraph of N nodes is taken
(graphs/real_loaders.connected_subhypergraph; size-2 hyperedges → edges, size-3 → hyperedges, larger ones
projected to pairwise cliques), hyperedges capped at --max-hyper. Dynamics simulated on it with recipe data
(sparse ICs, M=400, T=25), measurement noise, then HyperDiscover (structure only, default config) and the
regression baselines (SINDy generic, ARNI) on the same data. Writes real_structures.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

DATASETS = {
    "contact-high-school": ["S4"],
    "contact-primary-school": ["S4"],
    "hospital-lyon": ["S4"],
    "plant-pollinator-mpl-014": ["S2", "S5"],
    "email-enron": ["S3"],  # coauth-dblp is too large to load (millions of hyperedges)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--max-hyper", type=int, default=15)
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.01])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default="runs/xgi_cache")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    try:
        import xgi  # noqa: F401
    except ImportError:  # Kaggle: kernel has internet
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "xgi"])

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from hyperdiscover import HyperDiscover
    from hyperdiscover import baselines as B
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.data.observe import add_noise
    from hyperdiscover.data.simulate import simulate_system, sparse_ic
    from hyperdiscover.graphs.real_loaders import connected_subhypergraph, load_xgi
    from hyperdiscover.systems import make_system

    def f1(p, t):
        return 2 * len(p & t) / max(len(p) + len(t), 1)

    Path(a.cache).mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.datasets:
        full = load_xgi(name, cache_dir=None)
        for sid, noise, seed in itertools.product(DATASETS[name], a.noises, a.seeds):
            rng = np.random.default_rng([seed, int(sid[1:]), len(name)])
            st = connected_subhypergraph(full, a.N, rng, a.max_hyper)
            kw = {"rng": rng} if sid == "S1" else {}
            sysm = make_system(sid, st, None, **kw)
            data = simulate_system(sysm, 400, 25, None, rng, x0=sparse_ic(sysm, 400, rng))
            data = add_noise(data, noise, rng)
            trueE = {tuple(e) for e in np.asarray(st.edges).tolist()}
            trueH = {tuple(h) for h in np.asarray(st.hyperedges).tolist()}
            n = st.n_nodes
            allp = list(itertools.combinations(range(n), 2))
            allt = list(itertools.combinations(range(n), 3))
            yE = np.array([p in trueE for p in allp])
            yH = np.array([t in trueH for t in allt])
            base = dict(dataset=name, system=sid, noise=noise, seed=seed, N=n, n_edges=len(trueE),
                        n_hyper=len(trueH), density=2 * len(trueE) / (n * (n - 1)))
            hd = HyperDiscover(config={"device": a.device}, periodic=sid == "S1", distill_engine="none",
                               seed=seed, verbose=False)
            hd.fit(data.X, data.t)
            res = hd.structure_result_
            mode = hd.config["readout"]["undirected"]
            se = {tuple(sorted((i, j))): s for i, j, s in res.edges(-1.0, mode)}
            sh = {tuple(sorted((i, j, k))): s for i, j, k, s in res.hyperedges(-1.0, mode)}
            E = {tuple(sorted((i, j))) for i, j, _ in hd.pairwise_edges_}
            Hs = {tuple(sorted((i, j, k))) for i, j, k, _ in hd.hyperedges_}
            rows.append(dict(**base, method="HyperDiscover", edge_f1=f1(E, trueE), hyper_f1=f1(Hs, trueH),
                             hyper_fp=len(Hs - trueH),
                             edge_auroc=roc_auc_score(yE, [se.get(p, 0) for p in allp]),
                             hyper_auroc=roc_auc_score(yH, [sh.get(t, 0) for t in allt]) if 0 < yH.sum() < len(yH) else np.nan))
            X = data.X
            dX = estimate_derivatives(X, data.t, "savgol", window=9, polyorder=3)[:, 3:-3]
            X = X[:, 3:-3]
            for meth in ("sindy_generic", "arni"):
                r = (B.sindy_full(X, dX, "generic", None, sid == "S1") if meth == "sindy_generic"
                     else B.arni(X, dX, sid == "S1"))
                pu, tu = B.directed_to_undirected(r["pair_scores"], r.get("tri_scores", {}))
                E2 = {k for k, v in pu.items() if v > 1e-12}
                H2 = {k for k, v in tu.items() if v > 1e-12}
                rows.append(dict(**base, method=meth, edge_f1=f1(E2, trueE),
                                 hyper_f1=f1(H2, trueH) if meth != "arni" else np.nan,
                                 hyper_fp=len(H2 - trueH) if meth != "arni" else np.nan,
                                 edge_auroc=roc_auc_score(yE, [pu.get(p, 0) for p in allp]),
                                 hyper_auroc=(roc_auc_score(yH, [tu.get(t, 0) for t in allt])
                                              if meth != "arni" and 0 < yH.sum() < len(yH) else np.nan)))
            for r in rows[-3:]:
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                      flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "real_structures.csv", index=False)
    (out / "real_structures.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "real_structures.csv")


if __name__ == "__main__":
    main()
