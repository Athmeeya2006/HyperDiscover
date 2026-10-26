"""Truth-free hyper-parameter selection for the sparse-regression baselines (DECISIONS.md D-026).

The sparsity threshold of SINDy (STLSQ) and THIS (λ) is chosen by derivative-prediction error on
held-out *trajectories* (never by the ground-truth structure): among the grid values whose
validation MSE is within ``tol`` (default 5%) of the best, the sparsest model (largest threshold)
is selected. The same rule is applied to every method, so the comparison stays fair.
"""

from __future__ import annotations

import numpy as np

from .sindy_full import basis_terms, build_node_library, generic_terms, sindy_full

SINDY_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
THIS_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)


def _flat(X, dX, max_samples, rng):
    N = X.shape[2]
    Xs, Ys = X[..., 0].reshape(-1, N), dX[..., 0].reshape(-1, N)
    if len(Xs) > max_samples:
        sel = rng.choice(len(Xs), max_samples, replace=False)
        Xs, Ys = Xs[sel], Ys[sel]
    return Xs, Ys


def _pick(grid, mses, tol):
    mses = np.asarray(mses, dtype=float)
    best = np.nanmin(mses)
    ok = [g for g, m in zip(grid, mses) if np.isfinite(m) and m <= (1 + tol) * best]
    return max(ok)


def sindy_val_mse(
    result: dict, Xv: np.ndarray, Yv: np.ndarray, terms: dict, node_constant: bool = True
) -> float:
    """Normalised validation MSE of a ``sindy_full`` result on flattened states Xv, Yv [S, N]."""
    N = Xv.shape[1]
    err = 0.0
    for i in range(N):
        Theta, _ = build_node_library(Xv, i, terms, node_constant)
        w = np.asarray(result["coefs"][i]["w"])
        err += float(np.mean((Theta @ w - Yv[:, i]) ** 2))
    return err / N / (float(Yv.var()) + 1e-300)


def tune_sindy(
    X,
    dX,
    Xv,
    dXv,
    library: str,
    truth_exprs=None,
    periodic=False,
    grid=SINDY_GRID,
    tol=0.05,
    max_samples=5000,
    rng=None,
) -> tuple[float, list[dict]]:
    rng = rng if rng is not None else np.random.default_rng(0)
    terms = (
        {k: basis_terms(truth_exprs[k]) for k in ("f", "g", "h")}
        if library == "oracle"
        else generic_terms(periodic)
    )
    Xvs, Yvs = _flat(Xv, dXv, max(1000, max_samples // 2), rng)
    table = []
    for thr in grid:
        r = sindy_full(
            X, dX, library, truth_exprs, periodic, threshold=thr, max_samples=max_samples, rng=rng
        )
        nnz = sum(int(np.count_nonzero(c["w"])) for c in r["coefs"].values())
        table.append(
            {"threshold": thr, "val_mse": sindy_val_mse(r, Xvs, Yvs, terms), "n_nonzero": nnz}
        )
    chosen = _pick(grid, [t["val_mse"] for t in table], tol)
    return chosen, table


def tune_this(X, dX, Xv, dXv, dmax=2, x0=0.0, grid=THIS_GRID, tol=0.05, max_samples=5000, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    # GPL-3.0 port, imported lazily so the MIT package does not import it (LICENSE, DECISIONS D-040)
    from .this_reimpl import get_theta_d, my_sindy

    Xs, Ys = _flat(X, dX, max_samples, rng)
    Xvs, Yvs = _flat(Xv, dXv, max(1000, max_samples // 2), rng)
    N = Xs.shape[1]
    base = np.broadcast_to(np.asarray(x0 if x0 is not None else Xs.mean(0), float), (N,))
    theta, _ = get_theta_d((Xs - base).T, dmax)
    theta_v, _ = get_theta_d((Xvs - base).T, dmax)
    table = []
    for lam in grid:
        Xi, _ = my_sindy(theta, Ys.T, lam)
        mse = float(np.mean((Xi @ theta_v - Yvs.T) ** 2) / (Yvs.var() + 1e-300))
        table.append(
            {
                "threshold": lam,
                "val_mse": mse,
                "n_nonzero": int(np.count_nonzero(np.abs(Xi) > 1e-8)),
            }
        )
    chosen = _pick(grid, [t["val_mse"] for t in table], tol)
    return chosen, table


__all__ = ["SINDY_GRID", "THIS_GRID", "tune_sindy", "tune_this", "sindy_val_mse"]
