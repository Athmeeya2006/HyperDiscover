#!/usr/bin/env python
"""Oracle check for alternating gating (D-029): per-node non-negative LASSO with the TRUE f, g, h.

If this recovers the exact structure, the data support it and the failure is in learning the
message functions; if not, the selection step itself (or the data) is the bottleneck.

    python scripts/diagnose_oracle_lasso.py --experiment configs/experiment/sanity_s4.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml


def lasso_support(R, G, H, pair_t, tri_t, N, tri_penalty, criterion):
    from sklearn.linear_model import LassoLarsIC

    wp, wt = np.zeros(G.shape[1]), np.zeros(H.shape[1])
    for i in range(N):
        pc, tc = np.where(pair_t == i)[0], np.where(tri_t == i)[0]
        Xd = np.concatenate([G[:, pc], H[:, tc] / tri_penalty], 1)
        m = LassoLarsIC(criterion=criterion, positive=True, fit_intercept=False).fit(Xd, R[:, i])
        wp[pc] = m.coef_[: len(pc)]
        wt[tc] = m.coef_[len(pc) :] / tri_penalty
    return wp, wt


def score(w, true_mask, name):
    on = w > 1e-3 * max(w.max(), 1e-12)
    tp, fp, fn = int((on & true_mask).sum()), int((on & ~true_mask).sum()), int((~on & true_mask).sum())
    tw = w[true_mask]
    print(
        f"  {name:6s} TP {tp} FP {fp} FN {fn}   true-weight range [{tw.min():.3f}, {tw.max():.3f}]  "
        f"max false weight {w[~true_mask].max() if (~true_mask).any() else 0:.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/experiment/sanity_s4.yaml")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from hyperdiscover import benchmarks
    from hyperdiscover.data.dataset import TrainData

    spec = yaml.safe_load(open(a.experiment))["benchmark"]
    b = benchmarks.load(**spec)
    sysm, N = b.system, spec["N"]
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    tris = np.array(
        [(i, j, k) for i in range(N) for j in range(N) for k in range(j + 1, N) if i not in (j, k)]
    )
    true_p = {tuple(p) for p in b.structure.pair_index}
    true_t = {(t[0], *sorted(t[1:])) for t in b.structure.tri_index}
    mp = np.array([tuple(p) in true_p for p in pairs])
    mt = np.array([(t[0], *sorted(t[1:])) in true_t for t in tris])
    print(f"{spec['system_id']} N={N}: {mp.sum()} true directed pairs, {mt.sum()} true focal triples")

    td = TrainData(b.data.X, b.data.t, segment_starts=b.data.meta["segment_starts"])
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(td.n_samples, a.samples, replace=False)
    S = td.S.numpy()[idx].astype(np.float64)
    targets = {
        "savgol": td.Y.numpy()[idx].astype(np.float64),
        "exact": sysm.rhs(S),
    }
    G = sysm.g(S[:, pairs[:, 0]], S[:, pairs[:, 1]])[..., 0] * sysm.g_scale
    H = sysm.h(S[:, tris[:, 0]], S[:, tris[:, 1]], S[:, tris[:, 2]])[..., 0] * sysm.h_scale
    for tname, Y in targets.items():
        R = (Y - sysm.f(S) - sysm.c)[..., 0]
        for crit in ("bic", "aic"):
            for tp in (1.0, 2.0, 4.0):
                print(f"targets={tname} criterion={crit} tri_penalty={tp}")
                wp, wt = lasso_support(R, G, H, pairs[:, 0], tris[:, 0], N, tp, crit)
                score(wp, mp, "pairs")
                score(wt, mt, "tris")


if __name__ == "__main__":
    main()
