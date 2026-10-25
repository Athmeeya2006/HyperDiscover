"""Baselines: SINDy oracle exactness (Phase-2 gate item), scalability guard, output shapes."""

import numpy as np
import pytest

from hyperdiscover import baselines as B
from hyperdiscover import benchmarks
from hyperdiscover.baselines.common import stlsq
from hyperdiscover.eval.structure_metrics import edge_metrics, hyperedge_metrics


@pytest.fixture(scope="module")
def s4():
    b = benchmarks.load(
        "S4", N=8, n_hyper=2, hyper_kind="open", mean_degree=3.0, M=20, T=100, seed=0
    )
    return b, b.data.X_clean, b.data.dX_clean


def test_stlsq_recovers_sparse_vector(rng):
    A = rng.standard_normal((500, 12))
    w = np.zeros(12)
    w[[1, 5, 9]] = [2.0, -1.0, 0.5]
    est = stlsq(A, A @ w, threshold=0.05)
    np.testing.assert_allclose(est, w, rtol=1e-5, atol=1e-9)  # tiny ridge shrinkage (alpha=1e-6)
    assert np.array_equal(est != 0, w != 0)


def test_sindy_oracle_exact(s4):
    b, X, dX = s4
    r = B.sindy_full(X, dX, "oracle", b.system.sympy_exprs())
    rep = B.oracle_coefficient_report(r, b.system.sympy_exprs(), b.structure)
    assert rep["exact"], rep
    pu, tu = B.directed_to_undirected(r["pair_scores"], r["tri_scores"])
    assert edge_metrics(pu, b.structure, 1e-12)["f1"] == 1.0
    assert hyperedge_metrics(tu, b.structure, 1e-12)["all"]["f1"] == 1.0


def test_oracle_report_detects_wrong_coefficients(s4):
    b, X, dX = s4
    r = B.sindy_full(X, dX, "oracle", b.system.sympy_exprs())
    r["coefs"][0]["w"][1] *= 1.2  # 20% error on node 0's first f-term
    assert not B.oracle_coefficient_report(r, b.system.sympy_exprs(), b.structure)["exact"]


def test_library_size_guard():
    X = np.zeros((10, 1000))
    from hyperdiscover.baselines.sindy_full import build_node_library, generic_terms

    with pytest.raises(MemoryError):
        build_node_library(X, 0, generic_terms(False), max_cols=1000)


@pytest.mark.parametrize("fn", ["this", "arni", "corr", "pcorr", "granger", "te"])
def test_baselines_return_scores(fn, s4):
    b, X, dX = s4
    N = X.shape[2]
    out = {
        "this": lambda: B.this_reconstruct(X, dX)["pair_scores"],
        "arni": lambda: B.arni(X, dX)["pair_scores"],
        "corr": lambda: B.correlation_scores(X),
        "pcorr": lambda: B.partial_correlation_scores(X),
        "granger": lambda: B.granger_scores(X),
        "te": lambda: B.transfer_entropy_scores(X),
    }[fn]()
    assert all(0 <= i < N and 0 <= j < N and i != j for i, j in out)
    assert all(np.isfinite(v) for v in out.values())


def test_nri_smoke(s4):
    b, X, _ = s4
    r = B.nri(X, b.data.t, window=10, steps=5, batch=4, hidden=16)
    N = X.shape[2]
    assert len(r["pair_scores"]) == N * (N - 1)
    assert all(0.0 <= p <= 1.0 for p in r["pair_scores"].values())


def test_threshold_tuning_is_truth_free_and_sane(s4):
    from hyperdiscover.baselines.tuning import tune_sindy, tune_this

    b, X, dX = s4
    thr, table = tune_sindy(X[:15], dX[:15], X[15:], dX[15:], "oracle", b.system.sympy_exprs())
    assert thr in [t["threshold"] for t in table]
    best = min(t["val_mse"] for t in table)
    chosen = next(t for t in table if t["threshold"] == thr)
    assert chosen["val_mse"] <= 1.05 * best + 1e-15  # sparsest model within 5% of the best
    lam, ttab = tune_this(X[:15], dX[:15], X[15:], dX[15:])
    assert lam in [t["threshold"] for t in ttab]
