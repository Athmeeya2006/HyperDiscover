#!/usr/bin/env python
"""W12 (CLAUDE.md §18.15): 4-node interactions from third-order joint kicks.

    python scripts/fourbody_kick_experiment.py --out paper/tables

Extension of the joint-kick idea (intervene/joint_kick.py) to size-4 hyperedges. For the model class with terms up to
triadic, ẋ_i is a sum of functions of at most two neighbours, so the third-order mixed difference

    D_ijkl = Σ_{S ⊆ {j,k,l}} (−1)^{3−|S|} x_i^{S}(τ)  ≈  τ δ³ ∂³ẋ_i / ∂x_j ∂x_k ∂x_l

vanishes for ANY pairwise and triadic couplings and is non-zero only through a genuine 4-body term. Test system:
SIS with pairwise, triadic AND 4-body infection, ẋ_i = −μ x_i + β(1−x_i)Σ_j A_ij x_j + β_Δ(1−x_i)Σ B_ijk x_j x_k
+ β_4(1−x_i)Σ C_ijkl x_j x_k x_l; controls: the same system without 4-body terms (must give no 4-body detections)
and the triadic mixed difference on it (detects 3-body hyperedges but also responds to 4-body ones: report).
Noiseless (ranking) and noisy (χ²-calibrated, Bonferroni) detection. Writes fourbody_kicks.csv / .md.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


class SIS4:
    """Minimal SIS with up to 4-body infection terms (d = 1), batched over leading axes."""

    default_dt = 0.05
    ic_box = (0.0, 1.0)

    def __init__(self, N, edges, tris, quads, mu=1.0, beta=0.4, beta_d=1.2, beta_4=2.0):
        self.N, self.mu, self.beta, self.beta_d, self.beta_4 = N, mu, beta, beta_d, beta_4
        A = np.zeros((N, N))
        for a, b in edges:
            A[a, b] = A[b, a] = 1
        self.A = A
        self.tri_focal = [(i, *[v for v in t if v != i]) for t in tris for i in t]
        self.quad_focal = [(i, *[v for v in q if v != i]) for q in quads for i in q]

    def rhs(self, x):
        x1 = x[..., 0]
        inf = self.beta * x1 @ self.A.T
        for i, j, k in self.tri_focal:
            inf[..., i] += self.beta_d * x1[..., j] * x1[..., k]
        for i, j, k, l in self.quad_focal:
            inf[..., i] += self.beta_4 * x1[..., j] * x1[..., k] * x1[..., l]
        return (-self.mu * x1 + (1 - x1) * inf)[..., None]

    @staticmethod
    def postprocess(x):
        return np.clip(x, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--n-edges", type=int, default=12)
    ap.add_argument("--n-tris", type=int, default=3)
    ap.add_argument("--n-quads", type=int, default=2)
    ap.add_argument("--states", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--tau-dt", type=float, default=1.0)
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.001, 0.005])
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    from scipy.stats import chi2
    from sklearn.metrics import roc_auc_score

    from hyperdiscover.data.simulate import rk4_integrate

    rows = []
    for seed, with_quads, noise in itertools.product(a.seeds, [True, False], a.noises):
        rng = np.random.default_rng(seed)
        N = a.N
        allE = list(itertools.combinations(range(N), 2))
        edges = [allE[q] for q in rng.choice(len(allE), a.n_edges, replace=False)]
        allT = list(itertools.combinations(range(N), 3))
        tris = [allT[q] for q in rng.choice(len(allT), a.n_tris, replace=False)]
        allQ = list(itertools.combinations(range(N), 4))
        quads = [allQ[q] for q in rng.choice(len(allQ), a.n_quads, replace=False)] if with_quads else []
        s = SIS4(N, edges, tris, quads)
        xs = rng.uniform(0.05, 0.6, (a.states, N, 1))
        triples = list(itertools.combinations(range(N), 3))  # kicked source triples (j, k, l)
        subsets = [()] + [(v,) for v in range(N)] + list(itertools.combinations(range(N), 2)) + triples
        index = {sub: q for q, sub in enumerate(subsets)}
        X0 = np.repeat(xs[:, None], len(subsets), 1).copy()
        d = a.delta
        for q, sub in enumerate(subsets):
            for v in sub:
                X0[:, q, v] += d
        tau = a.tau_dt * s.default_dt
        xt = rk4_integrate(s.rhs, X0, np.array([0.0, tau, 2 * tau]), 10, s.postprocess)[..., 0]
        clean = 2 * xt[:, :, 1] - 0.5 * xt[:, :, 2]  # Richardson readout [S, K, N]
        sig = noise * float(np.sqrt(4.25)) / np.sqrt(a.repeats)  # per effective averaged readout
        eff = clean + sig * rng.standard_normal(clean.shape) if noise > 0 else clean

        def D3(j, k, l, eff=eff, index=index):  # third-order mixed difference at every target [S, N]
            tot = 0
            for r in range(4):
                for sub in itertools.combinations((j, k, l), r):
                    tot = tot + (-1) ** (3 - r) * eff[:, index[tuple(sorted(sub))]]
            return tot

        true_q = {tuple(q) for q in quads}
        cand, score, pval = [], [], []
        for j, k, l in triples:
            Dv = D3(j, k, l)
            for i in range(N):
                if i in (j, k, l):
                    continue
                cand.append(tuple(sorted((i, j, k, l))))
                score.append(float(np.sqrt(np.mean(Dv[:, i] ** 2))))
                if noise > 0:
                    pval.append(float(chi2.sf((Dv[:, i] ** 2).sum() / (8 * sig**2), a.states)))
        best: dict = {}
        bestp: dict = {}
        for q, c in enumerate(cand):
            best[c] = max(best.get(c, 0.0), score[q])
            if pval:
                bestp[c] = min(bestp.get(c, 1.0), pval[q])
        keys = list(best)
        y = np.array([k in true_q for k in keys])
        row = dict(seed=seed, with_quads=with_quads, noise=noise, n_quads=len(true_q),
                   quad_auroc=roc_auc_score(y, [best[k] for k in keys]) if 0 < y.sum() < len(y) else np.nan,
                   max_score_ratio=float(max(best.values()) / max(np.abs(clean).max(), 1e-300)))
        if pval:
            level = a.alpha / len(pval)
            det = {k for k, p in bestp.items() if p < level}
            row.update(quad_tp=len(det & true_q), quad_fp=len(det - true_q), quad_fn=len(true_q - det))
        rows.append(row)
        print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "fourbody_kicks.csv", index=False)
    (out / "fourbody_kicks.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "fourbody_kicks.csv")


if __name__ == "__main__":
    main()
