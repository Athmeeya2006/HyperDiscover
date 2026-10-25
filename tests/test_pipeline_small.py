"""CLAUDE.md §12 test 6 (Phase-4 gate): S4, N = 10, two open hyperedges, zero noise, 50
trajectories → exact structure and equations, on CPU in under ~10 minutes.

This is the acceptance gate, not a smoke test: it is EXPECTED TO FAIL until Phase 3–4
tuning is done (see RESEARCH_LOG.md). Run with `pytest -m slow`. The quick smoke test below
only checks that the pipeline runs end to end and produces well-formed outputs.
"""

import pytest

from hyperdiscover import HyperDiscover, benchmarks
from hyperdiscover.eval.run_metrics import run_metrics


def _bench():
    return benchmarks.load(
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


def test_pipeline_smoke():
    b = benchmarks.load("S4", N=6, n_hyper=1, mean_degree=2.0, M=8, T=60, seed=0)
    cfg = {
        "model": {"hidden": 16, "layers": 2},
        "train": {
            "pairwise": {"steps": 30, "log_every": 1000},
            "triadic": {"steps": 30, "log_every": 1000},
            "joint": {"steps": 20, "log_every": 1000},
        },
    }
    hd = HyperDiscover(config=cfg, distill_engine="library", verbose=False).fit(b.data.X, b.data.t)
    assert isinstance(hd.pairwise_edges_, list) and isinstance(hd.hyperedges_, list)
    assert set(hd.equations_) == {"f", "g", "h"}
    m = run_metrics(hd, b.system, b.data.X_clean[-2:], b.data.t)
    assert "edges" in m and "screening" in m
    assert "auroc" in m["edges"]
    traj = hd.simulate(b.data.X[0, 0], b.data.t[:5])
    assert traj.shape == (5, 6, 1)
    assert isinstance(hd.report(), str)


def test_structure_checkpoint_roundtrip(tmp_path):
    b = benchmarks.load("S4", N=6, n_hyper=1, mean_degree=2.0, M=6, T=40, seed=0)
    cfg = {
        "model": {"hidden": 8, "layers": 1},
        "train": {"pairwise": {"steps": 10}, "triadic": {"steps": 10}, "joint": {"steps": 5}},
    }
    hd = HyperDiscover(config=cfg, distill_engine="none", verbose=False).fit(b.data.X, b.data.t)
    hd.save_structure(tmp_path / "s.pt")
    hd2 = HyperDiscover.from_structure_checkpoint(
        tmp_path / "s.pt", b.data.X, b.data.t, "library", verbose=False
    )
    assert hd2.pairwise_edges_ == hd.pairwise_edges_
    hd2.fit_symbolic(b.data.X, b.data.t)
    assert hd2.equations_


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_sanity_gate_s4():
    b = _bench()
    M = b.data.X.shape[0]
    hd = HyperDiscover(config="default", distill_engine="auto", verbose=True).fit(
        b.data.X[:43], b.data.t
    )
    m = run_metrics(hd, b.system, b.data.X_clean[43:], b.data.t)
    assert m["edges"]["fp"] == 0 and m["edges"]["fn"] == 0, m["edges"]
    assert m["hyperedges"]["all"]["fp"] == 0 and m["hyperedges"]["all"]["fn"] == 0, m["hyperedges"][
        "all"
    ]
    assert m["equations"]["all_exact"], m["equations"]
    assert M == 50
