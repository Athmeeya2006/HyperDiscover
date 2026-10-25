"""Regression tests for bugs found in the 2026-10-25 independent review (DECISIONS.md D-023)."""

import numpy as np
import pytest
import sympy as sp

from hyperdiscover import benchmarks
from hyperdiscover.data.dataset import TrainData
from hyperdiscover.data.derivatives import estimate_derivatives
from hyperdiscover.data.observe import subsample
from hyperdiscover.data.simulate import simulate_system
from hyperdiscover.data.weak_form import make_weak_form
from hyperdiscover.distill.refit import RefitResult
from hyperdiscover.distill.select import _val_metrics
from hyperdiscover.graphs import make_structure
from hyperdiscover.systems import XI, make_system


def test_nan_candidate_scores_infinite():
    b = benchmarks.load("S4", N=6, n_hyper=1, mean_degree=2.0, M=3, T=30, seed=0)
    bad = RefitResult(
        {"f": sp.sqrt(-1 - XI**2), "g": sp.Integer(0), "h": sp.Integer(0)}, None, 1.0, 1, 10, True
    )
    mse, bic, roll = _val_metrics(
        bad, b.structure, b.data.X_clean, b.data.dX_clean, b.data.t, False
    )
    assert mse == np.inf and bic == np.inf and roll == np.inf


def test_subsample_rescales_segment_starts(rng):
    st = make_structure(6, rng, "er", 2.0, "open", n_hyper=1)
    d = simulate_system(make_system("S4", st), 3, 200, rng=rng, kick_std=0.2, kick_every=50)
    s = subsample(d, 4)
    assert s.meta["segment_starts"] == [0, 13, 25, 38]
    TrainData(s.X, s.t, segment_starts=s.meta["segment_starts"], weak_kw={"window": 9, "stride": 2})


def test_two_point_segment_derivative():
    t = np.arange(52) * 0.1
    X = np.sin(t)[None, :, None, None]
    out = estimate_derivatives(X, t, "savgol", [0, 50])
    assert np.all(np.isfinite(out))
    assert np.allclose(out[0, 50:, 0, 0], (X[0, 51, 0, 0] - X[0, 50, 0, 0]) / 0.1)


def test_weak_form_window_shrinks_to_short_segments():
    # kick segments shorter than the requested window: the window shrinks to fit (it used to raise,
    # which crashed every kicked run on Kaggle, RESEARCH_LOG 2026-10-26)
    wf = make_weak_form(np.arange(40) * 0.1, window=21, segment_starts=[0, 20])
    assert wf.length == 20 and len(wf.starts) >= 2
    assert all(s + wf.length <= 20 or s >= 20 for s in wf.starts)  # never straddles the kick
    with pytest.raises(ValueError, match="too short"):
        make_weak_form(np.arange(8) * 0.1, window=21, segment_starts=[0, 4])


def test_mixed_hyperedges_keep_their_class():
    for seed in range(20):
        r = np.random.default_rng(seed)
        st = make_structure(12, r, "er", 3.0, "mixed", n_hyper=8)
        want = [0, 1, 2, 3] * 2
        # classes are assigned in order 0,1,2,3,0,... and must survive later edge additions
        got = st.closure_class(np.array(sorted(map(tuple, st.hyperedges))))
        assert sorted(got.tolist()) == sorted(want), (seed, got)
