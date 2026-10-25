"""Identifiability option (c): interventions separate a pairwise-absorbed model from the truth."""

from itertools import combinations

import numpy as np
import sympy as sp

from hyperdiscover import benchmarks
from hyperdiscover.distill.refit import SymbolicModel, refit
from hyperdiscover.eval.perturbation import (
    design_interventions,
    perturbation_test,
    simulate_intervention,
)
from hyperdiscover.systems import XI, XJ, Structure


def test_clamped_nodes_stay_fixed(rng):
    b = benchmarks.load("S4", N=8, n_hyper=1, M=2, T=10, seed=0)
    t = np.arange(40) * 0.05
    traj = simulate_intervention(b.system, b.system.sample_ic(3, rng), t, [2, 5], [0.9, 0.1])
    assert np.allclose(traj[:, :, 2, 0], 0.9) and np.allclose(traj[:, :, 5, 0], 0.1)


def test_design_covers_every_focal_node(rng):
    b = benchmarks.load("S4", N=10, n_hyper=2, M=2, T=10, seed=0)
    plan = design_interventions(b.system, rng, n_random=0)
    assert len(plan) == 2 * 3 * len(b.structure.hyperedges)


def test_perturbation_separates_pairwise_absorption():
    b = benchmarks.load(
        "S4", N=10, n_hyper=2, hyper_kind="open", mean_degree=3.0, M=40, T=200, seed=0
    )
    st = b.structure
    # the failure mode seen in the pilot: hyperedge members linked pairwise, no hyperedges
    extra = [p for h in st.hyperedges for p in combinations(sorted(map(int, h)), 2)]
    absorbed = Structure(
        st.n_nodes, np.concatenate([st.edges, np.array(extra)]), np.zeros((0, 3), int)
    )
    fit = refit(
        {"f": sp.Float(-1.0) * XI, "g": sp.Float(0.4) * (1 - XI) * XJ, "h": sp.Integer(0)},
        absorbed,
        b.data.X_clean,
        b.data.dX_clean,
    )
    pairwise_model = SymbolicModel(absorbed, fit.exprs)
    truth_copy = SymbolicModel(st, b.system.sympy_exprs())
    t = np.arange(100) * 0.05
    res = perturbation_test(
        b.system, {"truth": truth_copy, "pairwise": pairwise_model}, t, n_ic=4, n_random=4
    )
    s = res["summary"]
    assert s["truth"]["max"] < 1e-6
    assert s["pairwise"]["hyper_targeted_mean"] > 10 * max(s["truth"]["hyper_targeted_mean"], 1e-3)
