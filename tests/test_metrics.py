"""Structure metrics with implicit (unscored) candidates match brute-force enumeration."""

from itertools import combinations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from hyperdiscover.eval.structure_metrics import edge_metrics, hyperedge_metrics
from hyperdiscover.graphs import make_structure


def test_edge_metrics_match_full_enumeration(rng):
    st = make_structure(12, rng, "er", 3.0, "open", n_hyper=2)
    all_pairs = list(combinations(range(12), 2))
    scored = {p: float(rng.uniform()) for p in all_pairs if rng.uniform() < 0.5}
    full_s = [scored.get(p, 0.0) for p in all_pairs]
    truth = {tuple(map(int, e)) for e in st.edges}
    full_y = [p in truth for p in all_pairs]
    m = edge_metrics(scored, st, 0.5)
    assert np.isclose(m["auroc"], roc_auc_score(full_y, full_s))
    assert np.isclose(m["auprc"], average_precision_score(full_y, full_s))
    assert m["n_pos"] == len(truth) and m["n_pos"] + m["n_neg"] == len(all_pairs)


def test_hyperedge_metrics_per_class_match_enumeration(rng):
    st = make_structure(10, rng, "er", 3.0, "mixed", n_hyper=4)
    triples = list(combinations(range(10), 3))
    scored = {t: float(rng.uniform()) for t in triples if rng.uniform() < 0.3}
    for h in st.hyperedges:  # make sure positives are partly scored
        scored[tuple(map(int, h))] = 0.9
    truth = {tuple(map(int, h)) for h in st.hyperedges}
    m = hyperedge_metrics(scored, st, 0.5)
    y = [t in truth for t in triples]
    s = [scored.get(t, 0.0) for t in triples]
    assert np.isclose(m["all"]["auroc"], roc_auc_score(y, s))
    cls = st.closure_class(np.array(triples))
    for name, cl in (("open", [0]), ("partial", [1, 2]), ("closed", [3])):
        sel = np.isin(cls, cl)
        ys, ss = np.array(y)[sel], np.array(s)[sel]
        if ys.any() and (~ys).any():
            assert np.isclose(m[name]["auroc"], roc_auc_score(ys, ss)), name
