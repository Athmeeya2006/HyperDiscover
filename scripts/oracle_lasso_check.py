#!/usr/bin/env python
"""Oracle check of the alternating-gating selection step: TRUE f, g, h as messages, S4 sanity instance.

    python scripts/oracle_lasso_check.py [--targets savgol|exact] [--msg-err 0.05]

Runs the exact selection code used in training (train/alternating.py:select_weights) with every ordered
pair and every focal triple as a candidate. If selection fails even with perfect messages, the structure
step (not MLP training) is the problem. Use ``--targets savgol`` (estimated derivatives, as in training):
the original version used exact derivatives, which hid the over-selection of the "w > 0" LASSO readout
(model audit 2026-10-26).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=["savgol", "exact"], default="savgol")
    ap.add_argument("--msg-err", type=float, default=0.0, help="relative per-column scale error")
    ap.add_argument("--samples", type=int, default=3000)
    a = ap.parse_args()

    from hyperdiscover import benchmarks
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.train.alternating import AlternatingConfig, select_weights

    b = benchmarks.load(
        "S4",
        N=10,
        noise=0.0,
        hyper_kind="open",
        n_hyper=2,
        graph="er",
        mean_degree=3.0,
        M=50,
        T=200,
        seed=0,
    )
    sysm, st, N = b.system, b.structure, b.structure.n_nodes
    if a.targets == "exact":
        dX = b.data.dX_clean
    else:
        dX = estimate_derivatives(b.data.X, b.data.t, "savgol", None, window=9, polyorder=3)
    trim = slice(2, -2)  # as TrainData: drop the derivative-filter edges
    X = b.data.X[:, trim].reshape(-1, N, 1)
    Y = dX[:, trim].reshape(-1, N, 1)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), a.samples, replace=False)
    X, Y = X[sel], Y[sel]
    R = (Y - sysm.f(X) - sysm.c)[..., 0]
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    tris = np.array(
        [
            (i, j, k)
            for i in range(N)
            for j, k in itertools.combinations(range(N), 2)
            if i not in (j, k)
        ]
    )
    G = sysm.g(X[:, pairs[:, 0]], X[:, pairs[:, 1]])[..., 0]
    H = sysm.h(X[:, tris[:, 0]], X[:, tris[:, 1]], X[:, tris[:, 2]])[..., 0]
    if a.msg_err:
        G = G * (1 + a.msg_err * rng.standard_normal(G.shape[1]))
        H = H * (1 + a.msg_err * rng.standard_normal(H.shape[1]))
    true_p = {tuple(p) for p in st.pair_index.tolist()}
    true_t = {(t[0], *sorted(t[1:])) for t in st.tri_index.tolist()}
    print(
        f"targets={a.targets} msg_err={a.msg_err}: {len(true_p)} true directed edges, "
        f"{len(true_t)} true focal triples; {len(pairs)} / {len(tris)} candidates"
    )
    for readout in ("lasso", "relaxed"):
        for tp in (1.0, 2.0, 4.0):
            cfg = AlternatingConfig(readout=readout, tri_penalty=tp)
            wp, wt = select_weights(R, G, H, pairs[:, 0], tris[:, 0], N, cfg)
            on_p = {tuple(p) for p in pairs[wp > 0].tolist()}
            on_t = {tuple(t) for t in tris[wt > 0].tolist()}
            print(
                f"  readout={readout:7s} tri_penalty={tp}: edges FP {len(on_p - true_p)} "
                f"FN {len(true_p - on_p)} | triples FP {len(on_t - true_t)} FN {len(true_t - on_t)}"
            )


if __name__ == "__main__":
    main()
