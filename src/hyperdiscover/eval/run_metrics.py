"""All metrics for one fitted run against a known ground truth -> one JSON-able dict."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..pipeline import undirected_pair_scores, undirected_tri_scores
from .equation_metrics import combined_coupling_error, equation_report
from .forecast_metrics import forecast_report
from .perturbation import perturbation_test
from .structure_metrics import (
    edge_metrics,
    false_hyperedge_rate,
    hyperedge_metrics,
    screening_recall_report,
)


def held_out_inputs(true_system, X: np.ndarray, n: int = 3000, rng=None) -> dict[str, np.ndarray]:
    """Inputs for functional errors drawn from the data at true edges / hyperedges."""
    rng = rng if rng is not None else np.random.default_rng(0)
    S = X.reshape(-1, X.shape[2], X.shape[3])[..., 0]
    s = rng.integers(0, len(S), n)
    out = {"f": S[s, rng.integers(0, S.shape[1], n)][:, None]}
    P, T = true_system.structure.pair_index, true_system.structure.tri_index
    if len(P):
        e = P[rng.integers(0, len(P), n)]
        out["g"] = np.stack([S[s, e[:, 0]], S[s, e[:, 1]]], 1)
    if len(T):
        e = T[rng.integers(0, len(T), n)]
        out["h"] = np.stack([S[s, e[:, 0]], S[s, e[:, 1]], S[s, e[:, 2]]], 1)
    return out


def run_metrics(hd, true_system, X_test: np.ndarray, t: np.ndarray, threshold: float = 0.5) -> dict:
    res = hd.structure_result_
    truth = true_system.structure
    mode = hd.config["readout"]["undirected"]
    pu = undirected_pair_scores(res.pair_index_all, res.pair_scores, mode)
    tu = undirected_tri_scores(res.tri_index_all, res.tri_scores, mode) if hd.max_order >= 3 else {}
    m: dict = {"screening": screening_recall_report(res.pair_index_all, res.tri_index_all, truth)}
    m["edges"] = edge_metrics(pu, truth, threshold)
    if len(truth.hyperedges):
        m["hyperedges"] = hyperedge_metrics(tu, truth, threshold)
    else:
        m["false_hyperedges"] = false_hyperedge_rate(tu, truth.n_nodes, threshold)
    m["timings"] = res.timings
    if hd.equations_:
        truth_exprs = true_system.sympy_exprs()
        if not len(truth.hyperedges):
            truth_exprs = {k: v for k, v in truth_exprs.items() if k != "h"}
        inputs = held_out_inputs(true_system, X_test)
        m["equations"] = equation_report(
            hd.equations_, truth_exprs, inputs, node_constants=true_system.has_node_constants
        )
        m["equations"]["discovered"] = {k: str(v) for k, v in hd.equations_.items()}
        m["equations"]["discovered_latex"] = {k: sp.latex(v) for k, v in hd.equations_.items()}
        disc = hd.discovered_system()
        states = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])
        m["coupling"] = combined_coupling_error(true_system, disc, states[: min(len(states), 2000)])
        n_ic = min(10, X_test.shape[0])
        m["forecast"] = forecast_report(
            true_system, disc, X_test[:n_ic, 0], t, periodic=true_system.periodic
        )
        # identifiability option (c): held-out interventions (never seen in training)
        pt = perturbation_test(
            true_system, {"discovered": disc}, t[: min(len(t), 100)], n_ic=3, n_random=5
        )
        m["perturbation"] = pt["summary"]["discovered"]
    return m
