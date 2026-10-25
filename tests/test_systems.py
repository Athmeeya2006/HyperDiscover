"""CLAUDE.md §12 test 1: RHS vs sympy ground truth; batched == sequential; pairwise-only B = 0."""

import numpy as np
import pytest
import sympy as sp

from hyperdiscover import benchmarks
from hyperdiscover.data.simulate import simulate_system
from hyperdiscover.graphs import make_structure, triple_class_counts
from hyperdiscover.systems import XI, XJ, XK, make_system

SYSTEM_IDS = ["S1", "S2", "S3", "S4", "S5"]


def _system(sid, rng, N=12, kind="mixed"):
    st = make_structure(N, rng, "er", 3.0, kind, n_hyper=4)
    return make_system(sid, st, rng=rng) if sid == "S1" else make_system(sid, st)


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_messages_match_sympy(sid, rng):
    sysm = _system(sid, rng)
    ex = sysm.sympy_exprs()
    lo, hi = sysm.ic_box
    P = rng.uniform(lo, hi, (500, 3))
    for kind, args, fn in (
        ("f", [XI], sysm.f),
        ("g", [XI, XJ], sysm.g),
        ("h", [XI, XJ, XK], sysm.h),
    ):
        lam = sp.lambdify(args, ex[kind], "numpy")
        want = np.broadcast_to(lam(*[P[:, q] for q in range(len(args))]), (500,))
        got = fn(*[P[:, q : q + 1] for q in range(len(args))])[:, 0]
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12, err_msg=f"{sid}.{kind}")


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_full_rhs_matches_sympy_sum(sid, rng):
    """Full vector field equals the explicit double sum built from the sympy expressions."""
    sysm = _system(sid, rng)
    ex = sysm.sympy_exprs()
    f = sp.lambdify([XI], ex["f"], "numpy")
    g = sp.lambdify([XI, XJ], ex["g"], "numpy")
    h = sp.lambdify([XI, XJ, XK], ex["h"], "numpy")
    lo, hi = sysm.ic_box
    x = rng.uniform(lo, hi, (sysm.N,))
    want = np.array([float(np.broadcast_to(f(x[i]), ())) for i in range(sysm.N)]) + sysm.c[:, 0]
    for a, b in sysm.structure.edges:
        want[a] += g(x[a], x[b])
        want[b] += g(x[b], x[a])
    for a, b, c in sysm.structure.hyperedges:
        want[a] += h(x[a], x[b], x[c])
        want[b] += h(x[b], x[a], x[c])
        want[c] += h(x[c], x[a], x[b])
    got = sysm.rhs(x[None, :, None])[0, :, 0]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_h_symmetric_in_jk(sid, rng):
    sysm = _system(sid, rng)
    lo, hi = sysm.ic_box
    a, b, c = (rng.uniform(lo, hi, (200, 1)) for _ in range(3))
    np.testing.assert_allclose(sysm.h(a, b, c), sysm.h(a, c, b), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_batched_equals_sequential(sid, rng):
    sysm = _system(sid, rng, N=8)
    x0 = sysm.sample_ic(4, rng)
    batch = simulate_system(sysm, 4, 30, x0=x0).X_clean
    for m in range(4):
        single = simulate_system(sysm, 1, 30, x0=x0[m : m + 1]).X_clean
        np.testing.assert_allclose(batch[m], single[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_bounded(sid, rng):
    sysm = _system(sid, rng, N=20)
    X = simulate_system(sysm, 10, 200, rng=rng).X_clean
    assert np.all(np.isfinite(X))
    if sid != "S1":
        assert np.abs(X).max() < 100


@pytest.mark.parametrize("sid", SYSTEM_IDS)
def test_pairwise_only_variant(sid, rng):
    sysm = _system(sid, rng).pairwise_only()
    assert len(sysm.structure.hyperedges) == 0
    b = benchmarks.load(sid, N=10, pairwise_only=True, M=2, T=10)
    assert len(b.structure.hyperedges) == 0
    parts = b.system.coupling_parts(b.data.X_clean)
    assert np.all(parts["tri"] == 0)


@pytest.mark.parametrize("kind,cls", [("open", {0}), ("closed", {3})])
def test_hyperedge_closure(kind, cls, rng):
    st = make_structure(30, rng, "er", 3.0, kind, n_hyper=5)
    assert set(st.closure_class().tolist()) <= cls


def test_triple_class_counts(rng):
    from itertools import combinations

    st = make_structure(12, rng, "er", 4.0, "mixed", n_hyper=3)
    brute = np.bincount(st.closure_class(np.array(list(combinations(range(12), 3)))), minlength=4)
    np.testing.assert_array_equal(triple_class_counts(st), brute)


def test_benchmark_deterministic():
    a = benchmarks.load("S4", N=10, noise=0.05, M=3, T=20, seed=7)
    b = benchmarks.load("S4", N=10, noise=0.05, M=3, T=20, seed=7)
    np.testing.assert_array_equal(a.data.X, b.data.X)
    np.testing.assert_array_equal(a.structure.edges, b.structure.edges)


def test_benchmark_roundtrip(tmp_path):
    b = benchmarks.load("S2", N=8, M=2, T=15, seed=1)
    benchmarks.save(b, tmp_path / "b.h5")
    data, spec, truth = benchmarks.load_file(tmp_path / "b.h5")
    sysm = benchmarks.rebuild_system(truth)
    np.testing.assert_allclose(
        sysm.rhs(data.X_clean), b.system.rhs(data.X_clean), rtol=1e-5, atol=1e-6
    )
    assert spec["N"] == 8
