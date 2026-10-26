#!/usr/bin/env python
"""G4c (CLAUDE.md §18.12): structure inference when a fraction of the nodes is never observed.

    python scripts/hidden_nodes_experiment.py --out paper/tables

Recipe data (sparse ICs, M=400, T=25), N nodes simulated, a random fraction ``--hidden`` removed before fitting
(structure only, alternating gating). Scored on the observed nodes: edges among observed nodes, hyperedges
whose three members are observed. Also counted: how the "cut" interactions show up — false edges between two
observed members of a hyperedge whose third member is hidden ("shadow edges"), false hyperedges that contain
both observed members of such a cut hyperedge ("shadow hyperedges": a proxy third node), false edges between
observed nodes with a common hidden neighbour, and all other false positives. Writes hidden_nodes.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--hidden", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument("--kinds", nargs="+", default=["open", "closed"])
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd

    from hyperdiscover import HyperDiscover, benchmarks

    rows = []
    for sid, kind, frac in itertools.product(a.systems, a.kinds, a.hidden):
        b = benchmarks.load(sid, N=a.N, hyper_kind=kind, noise=a.noise, M=400, T=25, ic_mode="sparse",
                            seed=a.seed)
        rng = np.random.default_rng([a.seed, int(sid[1:]), int(frac * 100)])
        n_hide = int(round(frac * a.N))
        hidden = set(rng.choice(a.N, n_hide, replace=False).tolist()) if n_hide else set()
        obs = [v for v in range(a.N) if v not in hidden]
        relabel = {v: q for q, v in enumerate(obs)}
        X = b.data.X[:, :, obs]
        hd = HyperDiscover(config={"device": a.device}, periodic=sid == "S1", distill_engine="none",
                           seed=a.seed, verbose=False)
        hd.fit(X, b.data.t)
        back = {q: v for v, q in relabel.items()}
        E = {tuple(sorted((back[i], back[j]))) for i, j, _ in hd.pairwise_edges_}
        Hs = {tuple(sorted((back[i], back[j], back[k]))) for i, j, k, _ in hd.hyperedges_}
        trueE = {tuple(e) for e in np.asarray(b.structure.edges).tolist() if not set(e) & hidden}
        allH = [tuple(h) for h in np.asarray(b.structure.hyperedges).tolist()]
        trueH = {h for h in allH if not set(h) & hidden}
        shadow_pairs = set()
        for h in allH:
            hid = set(h) & hidden
            if len(hid) == 1:
                shadow_pairs.add(tuple(sorted(set(h) - hid)))
        # observed pairs (u, v) with a common hidden neighbour w (edges u–w and v–w)
        nbr = {v: set() for v in range(a.N)}
        for u, v in np.asarray(b.structure.edges).tolist():
            nbr[u].add(v)
            nbr[v].add(u)
        adj_hidden = {tuple(sorted((u, v))) for w in hidden for u, v in itertools.combinations(sorted(nbr[w] - hidden), 2)}
        fpE = E - trueE

        def f1(p, tr):
            return 2 * len(p & tr) / max(len(p) + len(tr), 1)

        row = dict(
            system=sid, kind=kind, hidden_frac=frac, n_hidden=n_hide,
            edge_f1=f1(E, trueE), hyper_f1=f1(Hs, trueH) if trueH or Hs else np.nan,
            edge_fp=len(fpE), hyper_fp=len(Hs - trueH), hyper_fn=len(trueH - Hs),
            shadow_edges=len(fpE & shadow_pairs), shadow_candidates=len(shadow_pairs),
            common_hidden_neighbour_edges=len(fpE & adj_hidden),
            # false hyperedges that contain both observed members of a hyperedge cut by a hidden node
            # (the cut interaction re-routed through a proxy third node)
            shadow_hyperedges=sum(any(set(p) <= set(h) for p in shadow_pairs) for h in Hs - trueH),
        )
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hidden_nodes.csv", index=False)
    (out / "hidden_nodes.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "hidden_nodes.csv")


if __name__ == "__main__":
    main()
