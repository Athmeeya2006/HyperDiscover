"""Structure metrics (CLAUDE.md §9.1).

Edges and hyperedges are scored at the *undirected* level (a pair's score is the max/mean/median (default: median, a majority vote over members)
over its two directed gates; a triple's over its three focal gates). Candidates never
scored (dropped at screening) are "implicit negatives/positives" with score 0; AUROC/AUPRC
account for all C(N,2) pairs and C(N,3) triples without enumerating them, by giving one
pooled zero-score sample the weight of the unscored count.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..graphs.hypergraphs import triple_class_counts
from ..systems.base import Structure


def _weighted_metrics(scores, labels, n_implicit_neg, n_implicit_pos, threshold):
    s = list(scores)
    y = list(labels)
    w = [1.0] * len(s)
    if n_implicit_neg > 0:
        s.append(0.0)
        y.append(0)
        w.append(float(n_implicit_neg))  # noqa: E702
    if n_implicit_pos > 0:
        s.append(0.0)
        y.append(1)
        w.append(float(n_implicit_pos))  # noqa: E702
    s, y, w = np.asarray(s), np.asarray(y), np.asarray(w)
    out = {"n_pos": int(w[y == 1].sum()), "n_neg": int(w[y == 0].sum())}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out.update(auroc=float("nan"), auprc=float("nan"))
    else:
        out["auroc"] = float(roc_auc_score(y, s, sample_weight=w))
        out["auprc"] = float(average_precision_score(y, s, sample_weight=w))
    pred = s > threshold
    tp = float(w[(y == 1) & pred].sum())
    fp = float(w[(y == 0) & pred].sum())
    fn = float(w[(y == 1) & ~pred].sum())
    out["precision"] = tp / (tp + fp) if tp + fp else float("nan")
    out["recall"] = tp / (tp + fn) if tp + fn else float("nan")
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    out.update(tp=int(tp), fp=int(fp), fn=int(fn))
    return out


def edge_metrics(pred_scores: dict, truth: Structure, threshold: float = 0.5) -> dict:
    """pred_scores: {(a, b): score} with a < b (missing pairs = score 0)."""
    N = truth.n_nodes
    true = {(int(a), int(b)) for a, b in truth.edges}
    keys = list(pred_scores)
    labels = [k in true for k in keys]
    n_total = N * (N - 1) // 2
    n_pos_scored = sum(labels)
    imp_pos = len(true) - n_pos_scored
    imp_neg = n_total - len(keys) - imp_pos
    return _weighted_metrics([pred_scores[k] for k in keys], labels, imp_neg, imp_pos, threshold)


def hyperedge_metrics(
    pred_scores: dict, truth: Structure, threshold: float = 0.5, by_class: bool = True
) -> dict:
    """pred_scores: {(a, b, c): score}, a < b < c. Adds per-closure-class metrics
    (open = 0 internal pairwise edges, partial = 1–2, closed = 3), where negatives are the
    non-hyperedge triples *of the same class*."""
    N = truth.n_nodes
    true = {tuple(int(v) for v in h) for h in truth.hyperedges}
    keys = list(pred_scores)
    labels = [k in true for k in keys]
    n_total = N * (N - 1) * (N - 2) // 6
    imp_pos = len(true) - sum(labels)
    out = {
        "all": _weighted_metrics(
            [pred_scores[k] for k in keys],
            labels,
            n_total - len(keys) - imp_pos,
            imp_pos,
            threshold,
        )
    }
    if not by_class:
        return out
    counts = triple_class_counts(truth)  # all triples per class
    kcls = truth.closure_class(np.array(keys).reshape(-1, 3)) if keys else np.zeros(0, int)
    tlist = np.array(sorted(true)).reshape(-1, 3)
    tcls = truth.closure_class(tlist) if len(tlist) else np.zeros(0, int)
    groups = {"open": [0], "partial": [1, 2], "closed": [3]}
    for name, cl in groups.items():
        sel = [q for q in range(len(keys)) if kcls[q] in cl]
        ks = [keys[q] for q in sel]
        lab = [labels[q] for q in sel]
        n_true = int(np.isin(tcls, cl).sum())
        ip = n_true - sum(lab)
        n_cls = int(counts[cl].sum())
        out[name] = _weighted_metrics(
            [pred_scores[k] for k in ks], lab, n_cls - len(ks) - ip, ip, threshold
        )
    return out


def false_hyperedge_rate(pred_scores: dict, n_nodes: int, threshold: float = 0.5) -> dict:
    """For pairwise-only ground truth: number and rate (per possible triple) of predicted hyperedges."""
    n_pred = int(sum(s > threshold for s in pred_scores.values()))
    n_total = n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6
    return {"n_false_hyperedges": n_pred, "rate": n_pred / max(1, n_total)}


def screening_recall_report(pair_index, tri_index, truth: Structure) -> dict:
    from ..screen import screening_recall, triadic_screening_recall

    return {
        "pair_recall": screening_recall(pair_index, truth.pair_index),
        "tri_recall": triadic_screening_recall(tri_index, truth.tri_index),
        "n_pair_candidates": int(len(pair_index)),
        "n_tri_candidates": int(len(tri_index)),
    }
