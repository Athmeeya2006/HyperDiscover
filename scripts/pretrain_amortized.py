#!/usr/bin/env python
"""G2 (CLAUDE.md §18.12): train the amortized triadic screening scorer on simulated systems; test transfer.

    python scripts/pretrain_amortized.py --out runs/amortized

Instances: S1–S5 × {open, closed} × N ∈ --Ns × noise ∈ --noises × --seeds, recipe data (sparse ICs, M=400,
T=25), Savitzky–Golay derivatives (as in the pipeline). For every instance the wide candidate pool and its
statistics come from model/amortized.py; labels = true focal triples.
Evaluations (all on instances never seen in training):
  * in-distribution: new seeds of the training families;
  * leave-one-family-out: train on 4 systems, test on the 5th (repeated for each system);
Metrics per test instance: candidate AUROC and recall@K_t (top-K_t per focal node) for the learned score vs
the hand score (max |corr| over joint features, = screen/triadic.py's rule on the same pool), plus the
pool's own recall (ceiling). Writes amortized.csv / amortized.md and the in-distribution model
(amortized_scorer.pt with feature standardisation).
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np


def build_instance(sid, kind, N, noise, seed, pool):
    from hyperdiscover import benchmarks
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.model.amortized import triple_candidates_and_features
    from hyperdiscover.screen import screen_pairs

    b = benchmarks.load(sid, N=N, hyper_kind=kind, noise=noise, M=400, T=25, ic_mode="sparse", seed=seed)
    X = b.data.X
    if X.ndim == 3:
        X = X[..., None]
    t = b.data.t
    dX = estimate_derivatives(X, t, "savgol", window=9, polyorder=3)
    X, dX = X[:, 3:-3], dX[:, 3:-3]  # trim filter edges as the pipeline does
    per = sid == "S1"
    _, ps = screen_pairs(X, dX, 30, 200, per, max_samples=5000)
    tri, feats = triple_candidates_and_features(X, dX, ps, per, pool=pool)
    true = {tuple(map(int, r)) for r in b.structure.tri_index}
    y = np.array([tuple(map(int, r)) in true for r in tri], dtype=np.float32)
    pool_recall = float(y.sum() / max(len(true), 1))
    return dict(system=sid, kind=kind, N=N, noise=noise, seed=seed, tri=tri, X=feats, y=y,
                n_true=len(true), pool_recall=pool_recall)


def recall_at_k(tri, score, y, n_true, k):
    hit = 0
    for i in np.unique(tri[:, 0]):
        m = tri[:, 0] == i
        top = np.argsort(-score[m])[:k]
        hit += y[m][top].sum()
    return float(hit / max(n_true, 1))


def train(insts, epochs, seed):
    import torch

    from hyperdiscover.model.amortized import AmortizedTripleScorer, standardize

    torch.manual_seed(seed)
    Xall = np.concatenate([d["X"] for d in insts])
    _, mu, sd = standardize(Xall)
    model = AmortizedTripleScorer(Xall.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    pos = sum(d["y"].sum() for d in insts)
    neg = sum(len(d["y"]) for d in insts) - pos
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg / max(pos, 1))))
    tens = [(torch.tensor((d["X"] - mu) / sd), torch.tensor(d["tri"][:, 0]), torch.tensor(d["y"])) for d in insts]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for q in rng.permutation(len(tens)):
            xf, foc, y = tens[q]
            loss = lossf(model(xf, foc), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


def evaluate(model, mu, sd, insts, k_t, split, holdout):
    import torch
    from sklearn.metrics import roc_auc_score

    rows = []
    for d in insts:
        with torch.no_grad():
            s_l = model(torch.tensor((d["X"] - mu) / sd), torch.tensor(d["tri"][:, 0])).numpy()
        s_h = d["X"][:, :6].max(1)
        y = d["y"]

        def auc(s, y=y):
            return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

        rows.append(dict(
            split=split, holdout=holdout, system=d["system"], kind=d["kind"], N=d["N"], noise=d["noise"],
            seed=d["seed"], n_cand=len(y), pool_recall=d["pool_recall"],
            auroc_learned=auc(s_l), auroc_hand=auc(s_h),
            recall_learned=recall_at_k(d["tri"], s_l, y, d["n_true"], k_t),
            recall_hand=recall_at_k(d["tri"], s_h, y, d["n_true"], k_t),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--kinds", nargs="+", default=["open", "closed"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[20, 30, 50])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.01])
    ap.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--test-seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--k-t", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="runs/amortized")
    a = ap.parse_args()

    import pandas as pd
    import torch

    t0 = time.time()
    data = {}
    for sid, kind, N, noise, seed in itertools.product(
        a.systems, a.kinds, a.Ns, a.noises, a.train_seeds + a.test_seeds
    ):
        data[(sid, kind, N, noise, seed)] = build_instance(sid, kind, N, noise, seed, a.pool)
        print(f"built {sid} {kind} N={N} noise={noise} seed={seed} pool recall "
              f"{data[(sid, kind, N, noise, seed)]['pool_recall']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    tr = [d for k, d in data.items() if k[4] in a.train_seeds]
    te = [d for k, d in data.items() if k[4] in a.test_seeds]
    rows = []
    model, mu, sd = train(tr, a.epochs, 0)
    rows += evaluate(model, mu, sd, te, a.k_t, "in_distribution", "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "k_t": a.k_t, "pool": a.pool},
               out / "amortized_scorer.pt")
    for hold in a.systems:
        m2, mu2, sd2 = train([d for d in tr if d["system"] != hold], a.epochs, 0)
        rows += evaluate(m2, mu2, sd2, [d for d in te if d["system"] == hold], a.k_t, "leave_family_out", hold)
        print(f"held-out family {hold} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "amortized.csv", index=False)
    summ = df.groupby(["split", "system"])[["pool_recall", "auroc_learned", "auroc_hand",
                                             "recall_learned", "recall_hand"]].mean()
    (out / "amortized.md").write_text(summ.to_markdown(floatfmt=".3f") + "\n\n" + df.to_markdown(index=False, floatfmt=".3f"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
