"""Equation metrics (CLAUDE.md §9.2).

*Exact recovery* is decided by parametric equivalence (DECISIONS.md D-012): the discovered
function must (1) lie in the ground truth's parametric family, i.e. after fitting the truth's
constants to the discovered function on held-out inputs the residual is ≤ ``tol_resid``
(relative), (2) with every refitted constant within ``tol_coef`` (5%) of its true value, and
(3) have complexity ≤ truth complexity + ``slack`` (no extra small terms hiding in the
residual tolerance).
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from ..distill.refit import parametrize
from ..distill.simplify import complexity
from ..systems.base import XI, XJ, XK

ARGS = {"f": [XI], "g": [XI, XJ], "h": [XI, XJ, XK]}


def _eval(expr, kind, Xin):
    fn = sp.lambdify(ARGS[kind], expr, "numpy")
    with np.errstate(all="ignore"):
        return np.broadcast_to(
            np.asarray(fn(*[Xin[:, q] for q in range(Xin.shape[1])]), float), (len(Xin),)
        )


def functional_nrmse(
    expr_pred, expr_true, kind: str, Xin: np.ndarray, ref_scale: float | None = None
) -> float:
    """RMSE(pred − true) / std(true) on inputs Xin [S, n_inputs] (or / RMS(true) if std = 0; or / ref_scale
    if the true function is identically 0)."""
    yt = _eval(expr_true, kind, Xin)
    yp = _eval(expr_pred, kind, Xin)
    den = yt.std() if yt.std() > 1e-12 else np.sqrt(np.mean(yt**2)) + 1e-12
    if den <= 1e-12 and ref_scale:  # true function ≡ 0: relative to the couplings' scale (D-041)
        den = ref_scale
    v = float(np.sqrt(np.mean((yp - yt) ** 2)) / den)
    return v if np.isfinite(v) else float("inf")


def parametric_match(
    expr_pred,
    expr_true,
    kind: str,
    Xin: np.ndarray,
    tol_coef=0.05,
    tol_resid=1e-3,
    slack=2,
    ref_scale: float | None = None,
    free_constant: bool = False,
) -> dict:
    """``ref_scale``: magnitude used when the true function is 0 (std of the true couplings; D-041).
    ``free_constant``: the function's constant term is not identifiable (f with per-node constants c_i: only
    f(0) + mean(c) is), so a constant offset is removed from the prediction before comparing (D-041)."""
    expr_pred, expr_true = sp.sympify(expr_pred), sp.sympify(expr_true)
    yp = _eval(expr_pred, kind, Xin)
    if free_constant:
        yt0 = _eval(expr_true, kind, Xin)
        off = float(np.mean(yp) - np.mean(yt0))
        yp = yp - off
        expr_pred = sp.simplify(expr_pred - off)
    templ, syms, th_true = parametrize({kind: expr_true})
    t = templ[kind]
    out = {"complexity_pred": complexity(expr_pred), "complexity_true": complexity(expr_true)}
    if expr_true == 0:
        # relative to the magnitude of the true dynamics (the couplings), not to 0 (D-041)
        resid = float(np.sqrt(np.mean(yp**2))) / (ref_scale if ref_scale else 1.0)
        out.update(
            exact=bool(resid <= tol_resid or expr_pred == 0),
            coef_rel_err=[],
            max_coef_rel_err=0.0,
            resid=resid,
        )
        return out
    fn = sp.lambdify(ARGS[kind] + syms, t, "numpy")
    inputs = [Xin[:, q] for q in range(Xin.shape[1])]
    scale = yp.std() if yp.std() > 1e-12 else 1.0

    def r(p):
        with np.errstate(all="ignore"):
            v = (np.broadcast_to(np.asarray(fn(*inputs, *p), float), yp.shape) - yp) / scale
        return np.where(np.isfinite(v), v, 1e6)

    if len(th_true):
        sol = least_squares(r, th_true, max_nfev=500)
        th, resid = sol.x, float(np.sqrt(np.mean(sol.fun**2)))
    else:
        th, resid = th_true, float(np.sqrt(np.mean(r(th_true) ** 2)))
    rel = np.abs(th - th_true) / np.maximum(np.abs(th_true), 1e-12) if len(th_true) else np.zeros(0)
    exact = bool(
        resid <= tol_resid
        and np.all(rel <= tol_coef)
        and out["complexity_pred"] <= out["complexity_true"] + slack
    )
    out.update(
        exact=exact,
        coef_fit=th.tolist(),
        coef_true=th_true.tolist(),
        coef_rel_err=rel.tolist(),
        max_coef_rel_err=float(rel.max()) if len(rel) else 0.0,
        resid=resid,
    )
    return out


def equation_report(
    pred: dict, truth: dict, inputs: dict[str, np.ndarray], node_constants: bool = False
) -> dict:
    """pred/truth: {"f","g","h"} sympy; inputs: per-kind held-out input arrays.

    ``node_constants``: the system has per-node constants c_i, so f is compared up to a constant (D-041).
    A true function that is identically 0 is judged against the scale of the non-zero true functions.
    """
    scales = [
        float(_eval(truth[k], k, inputs[k]).std())
        for k in ("f", "g", "h")
        if k in truth and k in inputs and sp.sympify(truth[k]) != 0
    ]
    ref = max(scales) if scales else None
    rep = {}
    for kind in ("f", "g", "h"):
        if kind not in truth or kind not in inputs:
            continue
        p = pred.get(kind, sp.Integer(0))
        rep[kind] = {
            "pred": str(p),
            "true": str(truth[kind]),
            "nrmse": functional_nrmse(p, truth[kind], kind, inputs[kind], ref),
            **parametric_match(
                p, truth[kind], kind, inputs[kind], ref_scale=ref,
                free_constant=node_constants and kind == "f",
            ),
        }
    rep["all_exact"] = bool(all(rep[k]["exact"] for k in rep))
    return rep


def combined_coupling_error(true_system, pred_system, X: np.ndarray) -> dict:
    """Errors of each part and of the summed coupling (pair + tri) and full RHS, on states X [..., N, d].

    Separates 'wrong attribution' (parts wrong, sum right) from 'wrong dynamics'.
    """
    pt, pp = true_system.coupling_parts(X), pred_system.coupling_parts(X)

    def nrmse(a, b):
        den = a.std() if a.std() > 1e-12 else 1.0
        return float(np.sqrt(np.mean((a - b) ** 2)) / den)

    ct, cp = pt["pair"] + pt["tri"], pp["pair"] + pp["tri"]
    full_t = pt["self"] + ct
    full_p = pp["self"] + cp
    return {
        "self": nrmse(pt["self"], pp["self"]),
        "pair": nrmse(pt["pair"], pp["pair"]),
        "tri": nrmse(pt["tri"], pp["tri"])
        if np.any(pt["tri"])
        else float(np.sqrt(np.mean(pp["tri"] ** 2))),
        "coupling_sum": nrmse(ct, cp),
        "full_rhs": nrmse(full_t, full_p),
    }
