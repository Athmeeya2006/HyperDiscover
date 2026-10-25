"""Stage 5: model selection over combinations of distilled candidates (+ drop-one variants).

For each message function we take a shortlist from its Pareto front (best PySR-style score
−Δ log(loss)/Δ complexity, plus the most accurate), form all combinations, refit constants
on the data with the inferred structure, and rank by BIC on held-out trajectories (with
validation rollout error reported alongside). Drop-one-term variants of the winner are
then tried as "nearby" models.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..data.simulate import rk4_integrate
from ..systems.base import XJ, XK, Structure
from .pysr_runner import Candidate, DistillResult
from .refit import RefitResult, SymbolicModel, refit


def shortlist(res: DistillResult, k: int = 3) -> list[Candidate]:
    front = res.pareto()
    if len(front) <= k:
        return front
    scores = []
    for a, b in zip(front[:-1], front[1:]):
        dl = np.log(max(a.val_loss, 1e-300)) - np.log(max(b.val_loss, 1e-300))
        scores.append((dl / max(1, b.complexity - a.complexity), b))
    scores.sort(key=lambda s: -s[0])
    picks = [front[-1]] + [c for _, c in scores]
    out: list[Candidate] = []
    for c in picks:
        if c not in out:
            out.append(c)
        if len(out) == k:
            break
    return sorted(out, key=lambda c: c.complexity)


def drop_one_variants(expr: sp.Expr, kind: str = "") -> list[sp.Expr]:
    """Variants with one additive term removed. For h a term is removed together with its
    (j, k)-swapped mirror, so every variant keeps h(x_i, x_j, x_k) = h(x_i, x_k, x_j)."""
    terms = list(sp.Add.make_args(sp.expand(expr)))
    if len(terms) <= 1:
        return []
    groups: list[set[int]] = []
    if kind == "h":
        seen: set[int] = set()
        for q, t in enumerate(terms):
            if q in seen:
                continue
            mirror = sp.expand(t.xreplace({XJ: XK, XK: XJ}))
            orbit = {q} | {r for r, u in enumerate(terms) if r != q and sp.expand(u - mirror) == 0}
            seen |= orbit
            groups.append(orbit)
    else:
        groups = [{q} for q in range(len(terms))]
    if len(groups) <= 1:
        return []
    return [sp.Add(*[t for q, t in enumerate(terms) if q not in g]) for g in groups]


@dataclass
class SelectionEntry:
    exprs: dict[str, sp.Expr]
    fit: RefitResult
    val_mse: float
    val_bic: float
    rollout_nrmse: float
    origin: str

    def to_dict(self) -> dict:
        return {
            "exprs": {k: str(v) for k, v in self.exprs.items()},
            "latex": {k: sp.latex(v) for k, v in self.exprs.items()},
            "train_mse": self.fit.mse,
            "val_mse": self.val_mse,
            "val_bic": self.val_bic,
            "rollout_nrmse": self.rollout_nrmse,
            "n_params": self.fit.n_params,
            "origin": self.origin,
        }


@dataclass
class SelectionResult:
    best: SelectionEntry
    entries: list[SelectionEntry] = field(default_factory=list)


def _val_metrics(fit: RefitResult, structure: Structure, Xv, dXv, tv, periodic, horizon: int = 20):
    model = SymbolicModel(structure, fit.exprs, fit.node_constants, periodic)
    # plain means (not nanmean): a candidate that is NaN anywhere must score as infinitely bad,
    # and every score must be finite or +inf so min()/sorted() never see a NaN key
    with np.errstate(all="ignore"):
        pred = model.rhs(Xv)
        mse = float(np.mean((pred - dXv) ** 2))
        mse = mse if np.isfinite(mse) else float("inf")
        n = Xv[..., 0].size
        bic = (
            float(n * np.log(max(mse, 1e-300)) + fit.n_params * np.log(n))
            if np.isfinite(mse)
            else float("inf")
        )
        H = min(horizon, Xv.shape[1] - 1)
        try:
            traj = rk4_integrate(model.rhs, Xv[:, 0], tv[: H + 1], substeps=4)
            err = np.sqrt(np.mean((traj - Xv[:, : H + 1]) ** 2)) / (Xv.std() + 1e-12)
            roll = float(err) if np.isfinite(err) else float("inf")
        except Exception:
            roll = float("inf")
    return mse, bic, roll


def select_model(
    distilled: dict[str, DistillResult],
    structure: Structure,
    X_train: np.ndarray,
    dX_train: np.ndarray,
    X_val: np.ndarray,
    dX_val: np.ndarray,
    t: np.ndarray,
    node_constants: bool = False,
    periodic: bool = False,
    k: int = 3,
    try_drop_one: bool = True,
    rng: np.random.Generator | None = None,
    logger=print,
) -> SelectionResult:
    rng = rng if rng is not None else np.random.default_rng(0)
    lists = {}
    for kind in ("f", "g", "h"):
        if kind in distilled and distilled[kind].candidates:
            lists[kind] = [c.expr for c in shortlist(distilled[kind], k)]
        else:
            lists[kind] = [sp.Integer(0)]
    entries: list[SelectionEntry] = []

    def evaluate(exprs, origin):
        fit = refit(exprs, structure, X_train, dX_train, node_constants, rng=rng)
        mse, bic, roll = _val_metrics(fit, structure, X_val, dX_val, t, periodic)
        e = SelectionEntry(fit.exprs, fit, mse, bic, roll, origin)
        entries.append(e)
        return e

    for fe, ge, he in itertools.product(lists["f"], lists["g"], lists["h"]):
        evaluate({"f": fe, "g": ge, "h": he}, "combination")
    best = min(entries, key=lambda e: e.val_bic)
    if try_drop_one:
        for kind in ("f", "g", "h"):
            for v in drop_one_variants(best.exprs[kind], kind):
                evaluate({**best.exprs, kind: v}, f"drop_one:{kind}")
        best = min(entries, key=lambda e: e.val_bic)
    logger(
        f"[select] {len(entries)} models refit; best BIC {best.val_bic:.1f}, rollout nRMSE {best.rollout_nrmse:.3g}"
    )
    return SelectionResult(best, sorted(entries, key=lambda e: e.val_bic))
