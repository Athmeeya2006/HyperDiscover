"""End-to-end structure-learning pipeline: screening -> staged training -> readout.

Symbolic distillation and refitting live in ``hyperdiscover.distill`` and are orchestrated
by ``HyperDiscover`` (api.py).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.dataset import TrainData
from .model.structure_model import StructureModel
from .screen import screen_pairs, screen_triples
from .train import train_joint, train_pairwise, train_triadic
from .train.alternating import train_alternating

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "device": "cpu",
    "max_order": 3,
    "node_constants": "auto",  # auto | true | false  (auto: true for periodic systems)
    "periodic": False,
    # identifiability schedule (a) from CLAUDE.md §7 Stage 3; "joint" alone is the Phase-4 alternative
    "schedule": ["pairwise", "triadic", "joint"],
    # "hard_concrete" (CLAUDE.md §7 Stage 2, default) or "alternating" (MLP steps <-> per-node
    # non-negative BIC LASSO for the edge weights; DECISIONS.md D-029). Alternating ignores `schedule`.
    "gating": "hard_concrete",
    "data": {
        "deriv_method": "savgol",
        "deriv_kw": {"window": 9, "polyorder": 3},
        "weak_kw": {"window": 21, "stride": 4, "power": 4},
        "trim": 3,
    },
    "screen": {
        "all_pairs_max_n": 200,
        "k_p": 30,
        "all_triples_max_n": 12,
        "k_t": 50,
        "n_neighbours": 15,
    },
    "model": {"hidden": 64, "layers": 3, "init_prob_pair": 0.9, "init_prob_tri": 0.5},
    "train": {
        "pairwise": {
            "steps": 3000,
            "batch": 128,
            "loss": "derivative",
            "lam_pair": 0.01,
            "lr_net": 3e-3,
            "lr_gate": 5e-2,
        },
        "triadic": {
            "steps": 3000,
            "batch": 128,
            "loss": "derivative",
            "lam_tri": 0.02,
            "lr_net": 3e-3,
            "lr_gate": 5e-2,
            "anova": 0.0,
        },
        "joint": {
            "steps": 1500,
            "batch": 128,
            "loss": "derivative",
            "lam_pair": 0.01,
            "lam_tri": 0.02,
            "lr_net": 1e-3,
            "lr_gate": 2e-2,
            "anova": 0.0,
            "rollout_weight": 0.0,
            "rollout_horizon": 5,
            "beta_start": 0.667,
            "beta_end": 0.2,
        },
        "prune_between_stages": True,
        "prune_threshold": 0.05,
    },
    "readout": {"threshold": 0.5, "undirected": "max"},
}


def merge_config(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class StructureResult:
    model: StructureModel
    data: TrainData
    pair_index_all: np.ndarray  # screened candidates [E0, 2]
    tri_index_all: np.ndarray  # screened candidates [H0, 3]
    pair_scores: np.ndarray  # final gate probability per screened candidate [E0]
    tri_scores: np.ndarray  # [H0]
    histories: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def edges(self, threshold: float = 0.5, mode: str = "max"):
        """Undirected edges [(a, b, score)] with score > threshold."""
        und = undirected_pair_scores(self.pair_index_all, self.pair_scores, mode)
        return [(a, b, s) for (a, b), s in und.items() if s > threshold]

    def hyperedges(self, threshold: float = 0.5, mode: str = "max"):
        und = undirected_tri_scores(self.tri_index_all, self.tri_scores, mode)
        return [(a, b, c, s) for (a, b, c), s in und.items() if s > threshold]


def undirected_pair_scores(pair_index: np.ndarray, scores: np.ndarray, mode: str = "max") -> dict:
    acc: dict[tuple[int, int], list[float]] = {}
    for (i, j), s in zip(pair_index, scores):
        acc.setdefault((min(i, j), max(i, j)), []).append(float(s))
    red = np.max if mode == "max" else np.mean
    # a missing direction counts as 0 for the mean
    return {
        k: float(red(v + [0.0] * (2 - len(v)) if mode == "mean" else v)) for k, v in acc.items()
    }


def undirected_tri_scores(tri_index: np.ndarray, scores: np.ndarray, mode: str = "max") -> dict:
    acc: dict[tuple[int, int, int], list[float]] = {}
    for t, s in zip(tri_index, scores):
        acc.setdefault(tuple(sorted(int(v) for v in t)), []).append(float(s))
    red = np.max if mode == "max" else np.mean
    return {
        k: float(red(v + [0.0] * (3 - len(v)) if mode == "mean" else v)) for k, v in acc.items()
    }


def fit_structure(
    X: np.ndarray,
    t: np.ndarray,
    config: dict | None = None,
    segment_starts=None,
    ckpt_dir: str | Path | None = None,
    logger=print,
    stages: tuple[str, ...] | None = None,
) -> StructureResult:
    """Run screening and staged structure learning on training trajectories X [M, T, N, d]."""
    cfg = merge_config(DEFAULT_CONFIG, config)
    stages = tuple(stages if stages is not None else cfg["schedule"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    timings: dict[str, float] = {}
    M, T, N, d = X.shape
    periodic = bool(cfg["periodic"])
    use_c = periodic if cfg["node_constants"] == "auto" else bool(cfg["node_constants"])

    t0 = time.time()
    data = TrainData(
        X,
        t,
        cfg["data"]["deriv_method"],
        cfg["data"]["deriv_kw"],
        cfg["data"]["weak_kw"],
        segment_starts,
        cfg["data"]["trim"],
        cfg["device"],
    )
    timings["data"] = time.time() - t0

    t0 = time.time()
    sc = cfg["screen"]
    dX_masked = data.dX_np[:, data.time_mask]
    X_masked = X[:, data.time_mask]
    pair_index, pscore = screen_pairs(
        X_masked, dX_masked, sc["k_p"], sc["all_pairs_max_n"], periodic, rng=rng
    )
    if cfg["max_order"] >= 3:
        tri_index, _ = screen_triples(
            X_masked,
            dX_masked,
            pscore,
            sc["k_p"],
            sc["k_t"],
            sc["n_neighbours"],
            sc["all_triples_max_n"],
            periodic,
            rng=rng,
        )
    else:
        tri_index = np.zeros((0, 3), np.int64)
    timings["screen"] = time.time() - t0
    logger(f"[screen] {len(pair_index)} pair candidates, {len(tri_index)} triple candidates")

    mc = cfg["model"]
    model = StructureModel(
        N,
        d,
        pair_index,
        tri_index,
        mc["hidden"],
        mc["layers"],
        use_c,
        periodic,
        data.norm,
        mc["init_prob_pair"],
        mc["init_prob_tri"],
        seed,
    ).to(cfg["device"])
    # keep a map from current (possibly pruned) candidates back to the screened lists
    pair_ids = np.arange(len(pair_index))
    tri_ids = np.arange(len(tri_index))
    histories = []
    tc = cfg["train"]
    thr = tc["prune_threshold"]

    def prune(which: str):
        nonlocal pair_ids, tri_ids
        if not tc["prune_between_stages"]:
            return
        pp, pt = model.gate_probs()
        if which == "pair" and len(pp):
            keep = pp > thr
            model.prune(pair_keep=torch.as_tensor(keep))
            pair_ids = pair_ids[keep]
        if which == "tri" and len(pt):
            keep = pt > thr
            model.prune(tri_keep=torch.as_tensor(keep))
            tri_ids = tri_ids[keep]

    if cfg.get("gating", "hard_concrete") == "alternating":
        t0 = time.time()
        alt = cfg["train"].get("alternating", {})
        histories.append(_tag(train_alternating(model, data, alt, seed, logger), pair_ids, tri_ids))
        timings["alternating"] = time.time() - t0
        stages = ()  # the alternating schedule replaces the staged one
    if "pairwise" in stages:
        t0 = time.time()
        histories.append(
            _tag(
                train_pairwise(model, data, tc["pairwise"], seed, ckpt_dir, logger),
                pair_ids,
                tri_ids,
            )
        )
        prune("pair")
        timings["pairwise"] = time.time() - t0
    if cfg["max_order"] >= 3 and len(tri_index) and "triadic" in stages:
        t0 = time.time()
        histories.append(
            _tag(
                train_triadic(model, data, tc["triadic"], seed, ckpt_dir, logger), pair_ids, tri_ids
            )
        )
        prune("tri")
        timings["triadic"] = time.time() - t0
    if "joint" in stages and tc["joint"].get("steps", 0) > 0:
        t0 = time.time()
        if cfg["max_order"] < 3:
            model.use_tris = False
        histories.append(
            _tag(train_joint(model, data, tc["joint"], seed, ckpt_dir, logger), pair_ids, tri_ids)
        )
        timings["joint"] = time.time() - t0
    if cfg["max_order"] < 3:
        model.use_tris = False

    pp, pt = model.gate_probs()
    pair_scores = np.zeros(len(pair_index))
    pair_scores[pair_ids] = pp
    tri_scores = np.zeros(len(tri_index))
    if model.use_tris:
        tri_scores[tri_ids] = pt
    return StructureResult(
        model, data, pair_index, tri_index, pair_scores, tri_scores, histories, timings, cfg
    )


def _tag(hist: dict, pair_ids: np.ndarray, tri_ids: np.ndarray) -> dict:
    """Remember which screened candidates the gate snapshots in ``hist`` refer to."""
    hist["pair_ids"] = pair_ids.copy()
    hist["tri_ids"] = tri_ids.copy()
    return hist
