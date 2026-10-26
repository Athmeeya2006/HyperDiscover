"""§18.12 extensions: structure with known equations (windowed), budgeted kicks, amortized scorer."""

import numpy as np
import torch

from hyperdiscover import benchmarks
from hyperdiscover.intervene.active import ActiveKickConfig, active_joint_kicks
from hyperdiscover.model.amortized import N_FEATS, AmortizedTripleScorer
from hyperdiscover.train.windowed import all_candidates, select_structure


def test_known_equations_recover_structure_s4():
    b = benchmarks.load("S4", N=8, hyper_kind="open", n_hyper=2, M=200, T=25, ic_mode="sparse", seed=0)
    X = b.data.X_clean[..., 0].reshape(-1, 8)
    dX = b.data.dX_clean[..., 0].reshape(-1, 8)
    pairs, tris = all_candidates(8)
    wp, wt = select_structure(X, dX, b.system, pairs, tris)
    E = {tuple(sorted(map(int, pairs[e]))) for e in np.where(wp > 0)[0]}
    assert E == {tuple(e) for e in b.structure.edges.tolist()}
    cnt = {}
    for q in np.where(wt > 0)[0]:
        key = tuple(sorted(map(int, tris[q])))
        cnt[key] = cnt.get(key, 0) + 1
    assert {k for k, v in cnt.items() if v >= 2} == {tuple(h) for h in b.structure.hyperedges.tolist()}


def test_active_kicks_noiseless_s4_and_pairwise_control():
    for pw in (False, True):
        b = benchmarks.load("S4", N=10, hyper_kind="open", n_hyper=2, pairwise_only=pw, seed=0)
        xs = b.data.X_clean.reshape(-1, 10, 1)[:: 997][:10]
        cfg = ActiveKickConfig(tau=b.system.default_dt, max_per_target=9)
        res = active_joint_kicks(b.system, xs, cfg, 0.0)
        true = {tuple(h) for h in b.structure.hyperedges.tolist()}
        sc = res["hyper_scores"]
        top = sorted(sc, key=sc.get, reverse=True)[: len(true)]
        if true:  # noiseless: ranking only — true hyperedges score highest, far above the rest
            assert set(top) == true
            rest = max(v for k, v in sc.items() if k not in true)
            assert min(sc[k] for k in true) > 100 * rest
        else:
            assert max(sc.values()) < 1e-2  # pairwise-only: only tiny O(τ³) indirect residues


def test_amortized_scorer_shapes():
    m = AmortizedTripleScorer()
    out = m(torch.randn(7, N_FEATS), torch.tensor([0, 0, 1, 1, 1, 3, 3]))
    assert out.shape == (7,)
