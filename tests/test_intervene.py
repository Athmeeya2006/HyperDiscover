"""Joint-kick hyperedge detection: the mixed difference vanishes for pairwise couplings."""

import numpy as np
import pytest

from hyperdiscover import benchmarks
from hyperdiscover.intervene.joint_kick import (
    calibrated_pvalues,
    estimate_readout_noise,
    hyperedge_scores,
    kick_responses,
    mixed_difference_scores,
)


def _states(b, n=8, seed=0):
    X = b.data.X_clean.reshape(-1, b.structure.n_nodes, 1)
    return X[np.random.default_rng(seed).choice(len(X), n, replace=False)]


@pytest.mark.parametrize("sid", ["S1", "S2", "S4"])
def test_pairwise_only_has_no_mixed_response(sid):
    b = benchmarks.load(sid, N=8, pairwise_only=True, M=5, T=20, seed=0)
    tau = b.system.default_dt
    width = b.system.ic_box[1] - b.system.ic_box[0]
    r = kick_responses(b.system, _states(b), 0.2, tau)
    ps, ts = mixed_difference_scores(r, 0.2, tau, width)
    # pairwise responses are O(1); joint (mixed) responses only from O(τ³) indirect-path remainders
    assert ts.max() < 1e-2 * ps.max()


def test_s4_hyperedges_ranked_first():
    b = benchmarks.load(
        "S4", N=10, hyper_kind="open", n_hyper=2, graph="er", mean_degree=3.0, M=10, T=50, seed=0
    )
    tau = b.system.default_dt
    r = kick_responses(b.system, _states(b), 0.2, tau)
    _, ts = mixed_difference_scores(r, 0.2, tau)
    hs = hyperedge_scores(ts, r["pairs"])
    true_h = {tuple(sorted(h)) for h in np.asarray(b.structure.hyperedges).tolist()}
    top = sorted(hs, key=hs.get, reverse=True)[: len(true_h)]
    assert set(top) == true_h


def test_calibrated_pvalues_uniform_under_null():
    b = benchmarks.load("S4", N=8, pairwise_only=True, M=5, T=20, seed=0)
    rng = np.random.default_rng(3)
    tau = b.system.default_dt
    r = kick_responses(b.system, _states(b, 20), 0.2, tau, noise=0.05, rng=rng)
    sigma = estimate_readout_noise(r["base"], r["base_rep"])
    _, pt = calibrated_pvalues(r, sigma)
    j, k = r["pairs"][:, 0], r["pairs"][:, 1]
    mask = np.ones_like(pt, dtype=bool)
    mask[j, np.arange(len(j))] = False
    mask[k, np.arange(len(k))] = False
    p = pt[mask]
    # null p-values: no tiny ones after Bonferroni, and roughly uniform
    assert p.min() > 0.01 / p.size
    assert 0.35 < p.mean() < 0.65
