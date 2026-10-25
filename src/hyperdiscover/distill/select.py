"""Stage 5: model selection over combinations of distilled candidates (+ drop-one variants).

For each message function we take a shortlist from its Pareto front (best PySR-style score
−Δ log(loss)/Δ complexity, plus the most accurate), form all combinations, refit constants
on the data with the inferred structure, and rank by BIC on held-out trajectories, using the
AR(1) effective sample size of the residuals (validation rollout error reported alongside). Drop-one-term variants of the winner are
then tried as "nearby" models.

References (references/references.bib): schwarz1978estimating, zou2007degrees, bayley1946effective, bertsimas2016best.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ..data.simulate import rk4_integrate
from ..systems.base import XI, XJ, XK, Structure
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


def prune_small_terms(
    expr: sp.Expr, kind: str, S: np.ndarray, rel: float = 0.01, n: int = 4000, seed: int = 0
):
    """Variant with ALL additive terms whose RMS contribution on data states is < ``rel`` × the RMS of the
    whole expression removed jointly (None if nothing is that small). Catches several tiny terms that
    partly cancel each other, which one-at-a-time elimination cannot remove. S: states [n_s, N]."""
    terms = list(sp.Add.make_args(sp.expand(expr)))
    if len(terms) <= 1:
        return None
    rng = np.random.default_rng(seed)
    N = S.shape[1]
    rows = rng.integers(0, len(S), n)
    cols = (
        np.stack([rng.permutation(N)[:3] for _ in range(n)])
        if N >= 3
        else rng.integers(0, N, (n, 3))
    )
    args = {XI: S[rows, cols[:, 0]], XJ: S[rows, cols[:, 1]], XK: S[rows, cols[:, 2]]}
    syms = {"f": [XI], "g": [XI, XJ], "h": [XI, XJ, XK]}[kind]

    def rms(e):
        v = sp.lambdify(syms, e, "numpy")(*[args[x] for x in syms])
        return float(np.sqrt(np.mean(np.broadcast_to(np.asarray(v, float), (n,)) ** 2)))

    total = rms(expr)
    keep = [t for t in terms if rms(t) >= rel * total]
    if len(keep) == len(terms) or not keep:
        return None
    return sp.Add(*keep)


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


def _effective_n(res: np.ndarray) -> float:
    """AR(1) effective sample size n·(1−ρ)/(1+ρ) of residuals [M, T, N, d] (ρ = pooled lag-1
    autocorrelation along time), floored at one sample per trajectory and node. Consecutive time
    points are strongly correlated, so counting every one as independent made BIC's log(n)
    penalty too weak and let spurious small terms survive (audit E8)."""
    n = float(res[..., 0].size)
    if res.shape[1] < 3 or not np.all(np.isfinite(res)):
        return n
    a, b = res[:, 1:].ravel(), res[:, :-1].ravel()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    rho = float(np.clip(np.sum(a * b) / den, 0.0, 0.999)) if den > 0 else 0.0
    floor = float(res.shape[0] * res.shape[2])
    return max(floor, n * (1 - rho) / (1 + rho))


def _val_metrics(fit: RefitResult, structure: Structure, Xv, dXv, tv, periodic, horizon: int = 20):
    model = SymbolicModel(structure, fit.exprs, fit.node_constants, periodic)
    # plain means (not nanmean): a candidate that is NaN anywhere must score as infinitely bad,
    # and every score must be finite or +inf so min()/sorted() never see a NaN key
    with np.errstate(all="ignore"):
        pred = model.rhs(Xv)
        res = pred - dXv
        mse = float(np.mean(res**2))
        mse = mse if np.isfinite(mse) else float("inf")
        n = _effective_n(res)
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
    weak=None,
    X_weak: np.ndarray | None = None,
    max_drop_rounds: int = 10,
    drop_tol: float = 0.02,
    r2_tol: float = 1e-5,
) -> SelectionResult:
    """``weak`` (a WeakForm) + ``X_weak`` (contiguous training trajectories): refit constants on the
    weak-form residual instead of explicit derivatives (CLAUDE.md §15: noise ≥ 5%, audit E9)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    lists = {}
    for kind in ("f", "g", "h"):
        if kind in distilled and distilled[kind].candidates:
            lists[kind] = [c.expr for c in shortlist(distilled[kind], k)]
        else:
            lists[kind] = [sp.Integer(0)]
    entries: list[SelectionEntry] = []

    def evaluate(exprs, origin):
        if weak is not None:
            fit = refit(exprs, structure, X_weak, None, node_constants, weak=weak, rng=rng)
        else:
            fit = refit(exprs, structure, X_train, dX_train, node_constants, rng=rng)
        mse, bic, roll = _val_metrics(fit, structure, X_val, dX_val, t, periodic)
        e = SelectionEntry(fit.exprs, fit, mse, bic, roll, origin)
        entries.append(e)
        return e

    for fe, ge, he in itertools.product(lists["f"], lists["g"], lists["h"]):
        evaluate({"f": fe, "g": ge, "h": he}, "combination")
    best = min(entries, key=lambda e: e.val_bic)
    if try_drop_one:
        # greedy backward elimination: drop the term (h: term + its j↔k mirror) whose removal raises the
        # validation error least; accept the drop if BIC improves OR the error rises by less than
        # `drop_tol` (relative). The tolerance is needed on clean data, where the true model's residual is
        # only derivative-estimation error and BIC keeps any term that shaves a sliver off it (e.g.
        # 3e-4·cos(x_i − 2x_j + 2x_k) in h). One round only (the old rule) could never remove several
        # small spurious terms, e.g. f = −x_i + 0.015 cos x_i − 0.034 + 0.02 e^{−x_i}.
        # `r2_tol`: also drop when the error rises by < r2_tol × Var(target), i.e. the term explains a
        # negligible share of the dynamics. On clean data the residual floor is derivative-estimation
        # error, and terms like 6e-5·x_i² fit it by > drop_tol relative (Kaggle hd-gate-clean, D-036).
        var_y = float(np.var(dX_val)) if np.isfinite(np.var(dX_val)) else 0.0
        for _ in range(max_drop_rounds):
            cands = [
                evaluate({**best.exprs, kind: v}, f"drop_one:{kind}")
                for kind in ("f", "g", "h")
                for v in drop_one_variants(best.exprs[kind], kind)
            ]
            S_states = X_train[..., 0].reshape(-1, X_train.shape[2])
            for kind in ("f", "g", "h"):  # several tiny (possibly cancelling) terms removed jointly
                v = prune_small_terms(best.exprs[kind], kind, S_states)
                if v is not None:
                    cands.append(evaluate({**best.exprs, kind: v}, f"prune_small:{kind}"))
            ok = [
                e
                for e in cands
                if e.val_bic < best.val_bic
                or e.val_mse <= best.val_mse * (1 + drop_tol)
                or e.val_mse - best.val_mse <= r2_tol * var_y
            ]
            if not ok:
                break
            best = min(ok, key=lambda e: (e.val_mse, e.val_bic))
    logger(
        f"[select] {len(entries)} models refit; best BIC {best.val_bic:.1f}, rollout nRMSE {best.rollout_nrmse:.3g}"
    )
    return SelectionResult(best, sorted(entries, key=lambda e: e.val_bic))
