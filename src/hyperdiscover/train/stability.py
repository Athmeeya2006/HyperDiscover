"""Stability selection and per-triple BIC test for alternating gating (CLAUDE.md §18.4 P0).

Both run after the alternating schedule, with the message functions f, g, h fixed, so the model is
linear in the edge weights (train/alternating.py) and every step is a cheap per-node regression:

1. **Stability selection** (Meinshausen & Bühlmann 2010): repeat the per-node LASSO selection
   (``select_weights``) on ``n_boot`` independent random sample batches; keep a candidate only if it
   is selected in at least ``threshold`` of them. A false triple that wins on one batch by fitting
   noise or derivative error is rarely selected consistently.
2. **Per-triple BIC test**: for every kept triple at focal node i, refit node i's non-negative least
   squares with and without it on a fresh batch; drop it unless removing it raises BIC
   (n log(RSS/n) + k log n). Triples are tested weakest first and the support is updated after each
   drop (backward elimination restricted to triples; pairs are not tested here).

Weights of the final support are refit by NNLS on a fresh batch (unshrunk), then rescaled by the
same "typical strong weight" rule as the readout.

References (references/references.bib): meinshausen2010stability, shah2013variable, schwarz1978estimating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import nnls

from ..data.dataset import TrainData
from ..model.structure_model import StructureModel
from .alternating import AlternatingConfig, batch_design, select_on_batch


@dataclass
class StabilityConfig:
    n_boot: int = 30
    threshold: float = 0.8
    bic_test: bool = True
    samples: int = 3000

    @classmethod
    def from_dict(cls, d: dict | None) -> StabilityConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _bic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _nnls_rss(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    if A.shape[1] == 0:
        return np.zeros(0), float(y @ y)
    coef, res = nnls(A, y)
    return coef, float(res**2)


def selection_frequencies(
    model: StructureModel,
    data: TrainData,
    alt_cfg: AlternatingConfig,
    cfg: StabilityConfig,
    gen: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of ``n_boot`` sample batches in which each candidate pair / triple is selected."""
    fp = np.zeros(len(model.pair_index))
    ft = np.zeros(len(model.tri_index))
    for _ in range(cfg.n_boot):
        R, G, H = batch_design(model, data, cfg.samples, gen, alt_cfg.loss)
        wp, wt = select_on_batch(model, R, G, H, alt_cfg, True)
        fp += wp > 0
        ft += wt > 0
    return fp / cfg.n_boot, ft / cfg.n_boot


def bic_prune_triples(
    R: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    pair_t: np.ndarray,
    tri_t: np.ndarray,
    keep_p: np.ndarray,
    keep_t: np.ndarray,
    N: int,
) -> np.ndarray:
    """Backward elimination of triples by BIC, per focal node. Returns the updated triple mask."""
    keep_t = keep_t.copy()
    n = len(R)
    for i in range(N):
        pc = np.where((pair_t == i) & keep_p)[0]
        while True:
            tc = np.where((tri_t == i) & keep_t)[0]
            if len(tc) == 0:
                break
            A = np.concatenate([G[:, pc], H[:, tc]], 1)
            coef, rss = _nnls_rss(A, R[:, i])
            k = int((coef > 0).sum())
            base = _bic(rss, n, k)
            # weakest triple first: smallest contribution coef · rms(column)
            contrib = coef[len(pc) :] * np.sqrt((H[:, tc] ** 2).mean(0))
            worst = int(np.argmin(contrib))
            cols = [q for q in range(A.shape[1]) if q != len(pc) + worst]
            coef2, rss2 = _nnls_rss(A[:, cols], R[:, i])
            if _bic(rss2, n, int((coef2 > 0).sum())) <= base:
                keep_t[tc[worst]] = False  # removal does not worsen BIC: drop and retest
            else:
                break
    return keep_t


@torch.no_grad()
def stabilize(
    model: StructureModel,
    data: TrainData,
    alt_cfg: AlternatingConfig,
    cfg: StabilityConfig,
    seed: int = 0,
    logger=print,
) -> dict:
    """Replace the model's edge weights by the stable, BIC-tested support (refit by NNLS)."""
    gen = torch.Generator().manual_seed(seed + 777)
    fp, ft = selection_frequencies(model, data, alt_cfg, cfg, gen)
    keep_p = fp >= cfg.threshold
    lock = getattr(model, "pair_lock_mask", None)
    if alt_cfg.pair_lock and lock is not None:
        keep_p = lock.copy()  # pairs frozen by the Occam schedule are not re-selected
    keep_t = ft >= cfg.threshold
    if alt_cfg.fix_tris:
        keep_t = model.tri_w.cpu().numpy() > 0
    pair_t = model.pair_index[:, 0].cpu().numpy()
    tri_t = model.tri_index[:, 0].cpu().numpy() if len(model.tri_index) else np.zeros(0, int)
    R, G, H = batch_design(model, data, cfg.samples, gen, alt_cfg.loss)
    n_stable_t = int(keep_t.sum())
    if cfg.bic_test and not alt_cfg.fix_tris:
        keep_t = bic_prune_triples(R, G, H, pair_t, tri_t, keep_p, keep_t, model.N)
    # unshrunk refit of the final support on a fresh batch
    R, G, H = batch_design(model, data, cfg.samples, gen, alt_cfg.loss)
    wp = np.zeros(len(pair_t))
    wt = np.zeros(len(tri_t))
    for i in range(model.N):
        pc = np.where((pair_t == i) & keep_p)[0]
        tc = np.where((tri_t == i) & keep_t)[0]
        coef, _ = _nnls_rss(np.concatenate([G[:, pc], H[:, tc]], 1), R[:, i])
        wp[pc] = coef[: len(pc)]
        wt[tc] = coef[len(pc) :]
    if alt_cfg.fix_tris:
        wt = model.tri_w.cpu().numpy()
    dev = model.pair_w.device
    model.pair_w = torch.as_tensor(wp, dtype=torch.float32, device=dev)
    model.tri_w = torch.as_tensor(wt, dtype=torch.float32, device=dev)
    info = {
        "pairs_stable": int(keep_p.sum()),
        "tris_stable": n_stable_t,
        "tris_after_bic": int(keep_t.sum()),
        "pair_freq": fp,
        "tri_freq": ft,
    }
    logger(
        f"[stability] {cfg.n_boot} batches, π ≥ {cfg.threshold}: pairs {info['pairs_stable']}, "
        f"triples {n_stable_t} -> {info['tris_after_bic']} after BIC test"
    )
    return info
