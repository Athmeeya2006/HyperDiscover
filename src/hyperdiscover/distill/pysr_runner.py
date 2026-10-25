"""Stage 4: symbolic regression on the learned message functions.

Primary engine: PySR (Cranmer 2023). Julia must be installed beforehand (on Ada: on the login
node, into ``JULIA_DEPOT_PATH`` on shared storage; compute nodes are offline).

Fallback engine ``library``: exhaustive sparse regression over a fixed grammar of candidate
terms (monomials ≤ degree 3, sin/cos of small integer combinations of the inputs, and
products with x_i). It is NOT symbolic regression in the PySR sense (it cannot discover
rational or nested forms) and exists only so the pipeline and CI run without Julia. Results
from it are labelled ``engine="library"`` everywhere.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from ..systems.base import XI, XJ, XK
from .simplify import clean, complexity, rename_inputs

WORKER = Path(__file__).with_name("_pysr_worker.py")

DEFAULT_PYSR = {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp", "square", "cube"],
    "maxsize": 20,
    "niterations": 200,
    "timeout_in_seconds": 600,
    "parsimony": 1e-3,
    "populations": 30,
    "restarts": 3,
    "verbosity": 0,
    "progress": False,
}


@dataclass
class Candidate:
    expr: sp.Expr
    complexity: int
    train_loss: float
    val_loss: float = float("nan")
    engine: str = "pysr"
    restart: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": str(self.expr),
            "latex": sp.latex(self.expr),
            "complexity": self.complexity,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "engine": self.engine,
            "restart": self.restart,
        }


@dataclass
class DistillResult:
    kind: str
    candidates: list[Candidate] = field(default_factory=list)

    def pareto(self) -> list[Candidate]:
        """Pareto front over (complexity, validation loss)."""
        cs = sorted(self.candidates, key=lambda c: (c.complexity, c.val_loss))
        front, best = [], np.inf
        for c in cs:
            if c.val_loss < best - 1e-15:
                front.append(c)
                best = c.val_loss
        return front


def pysr_available() -> bool:
    """True if PySR is installed. Does not import it: importing PySR starts Julia, which fails
    in a process that has already imported torch; ``run_pysr`` therefore uses a subprocess."""
    import importlib.util

    return importlib.util.find_spec("pysr") is not None


def _mse(expr: sp.Expr, X: np.ndarray, y: np.ndarray, n_inputs: int) -> float:
    vars_ = [XI, XJ, XK][:n_inputs]
    try:
        fn = sp.lambdify(vars_, expr, "numpy")
        with np.errstate(all="ignore"):
            pred = np.broadcast_to(
                np.asarray(fn(*[X[:, q] for q in range(n_inputs)]), dtype=float), y.shape
            )
        v = float(np.mean((pred - y) ** 2))
        return v if np.isfinite(v) else float("inf")
    except Exception:
        return float("inf")


def run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    options: dict | None = None,
    seed: int = 0,
    out_dir: str | Path | None = None,
    X_val=None,
    y_val=None,
    timeout: float | None = None,
) -> DistillResult:
    """Run PySR ``restarts`` times, each in a fresh subprocess (see ``_pysr_worker``).

    Returns all hall-of-fame candidates with inputs renamed to x_i, x_j, x_k. Hall-of-fame
    files are kept under ``out_dir`` (CLAUDE.md §14).
    """
    import json
    import subprocess
    import sys
    import tempfile

    opts = {**DEFAULT_PYSR, **(options or {})}
    restarts = int(opts.pop("restarts", 1))
    n_inputs = X.shape[1]
    res = DistillResult(kind)
    X_val = X if X_val is None else X_val
    y_val = y if y_val is None else y_val
    base = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="hd_pysr_"))
    procs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    for r in range(restarts):
        job = base / f"pysr_{kind}_r{r}"
        job.mkdir(parents=True, exist_ok=True)
        np.savez(job / "job.npz", X=X.astype(np.float64), y=y.ravel().astype(np.float64))
        (job / "job.json").write_text(
            json.dumps(
                {
                    "options": opts,
                    "seed": seed + r,
                    "procs": procs,
                    "output_directory": str(job / "hof"),
                }
            )
        )
        budget = (
            timeout
            if timeout is not None
            else float(opts.get("timeout_in_seconds", 600)) * 1.5 + 300
        )
        # run the worker file directly (not ``-m``): importing the hyperdiscover package would load
        # torch before juliacall; ``runpy`` from ``-c`` also keeps this directory off sys.path
        code = f"import runpy, sys; sys.argv = ['w', {str(job)!r}]; runpy.run_path({str(WORKER)!r}, run_name='__main__')"
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=budget
        )
        (job / "worker.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0 or not (job / "result.json").exists():
            raise RuntimeError(
                f"PySR worker failed (restart {r}); see {job / 'worker.log'}:\n{proc.stderr[-2000:]}"
            )
        for row in json.loads((job / "result.json").read_text())["equations"]:
            if "sympy" not in row:
                continue
            e = clean(rename_inputs(sp.sympify(row["sympy"]), n_inputs), kind)
            res.candidates.append(
                Candidate(
                    e,
                    complexity(e),
                    _mse(e, X, y.ravel(), n_inputs),
                    _mse(e, X_val, y_val.ravel(), n_inputs),
                    "pysr",
                    r,
                )
            )
    return res


# ----------------------------------------------------------------------------- fallback
def _library_terms(n_inputs: int) -> list[sp.Expr]:
    v = [XI, XJ, XK][:n_inputs]
    terms: list[sp.Expr] = []
    for deg in (1, 2, 3):
        for combo in itertools.combinations_with_replacement(v, deg):
            terms.append(sp.Mul(*combo))
    if n_inputs >= 2:
        coefs = [-2, -1, 0, 1, 2]
        seen = set()
        for c in itertools.product(coefs, repeat=n_inputs):
            if all(ci == 0 for ci in c):
                continue
            arg = sum(ci * vi for ci, vi in zip(c, v))
            key = sp.srepr(sp.expand(arg))
            neg = sp.srepr(sp.expand(-arg))
            if key in seen or neg in seen:
                continue
            seen.add(key)
            terms += [sp.sin(arg), sp.cos(arg)]
    else:
        terms += [sp.sin(XI), sp.cos(XI), sp.exp(-XI)]
    return terms


def run_library(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    max_terms: int = 4,
    X_val=None,
    y_val=None,
    lam: float = 1e-3,
) -> DistillResult:
    """Greedy forward selection over a fixed term grammar (see module docstring)."""
    n_inputs = X.shape[1]
    terms = [sp.Integer(1)] + _library_terms(n_inputs)
    vars_ = [XI, XJ, XK][:n_inputs]
    cols = []
    for tm in terms:
        f = sp.lambdify(vars_, tm, "numpy")
        cols.append(
            np.broadcast_to(np.asarray(f(*[X[:, q] for q in range(n_inputs)]), float), (len(X),))
        )
    Theta = np.stack(cols, 1)
    yv = y.ravel()
    X_val = X if X_val is None else X_val
    y_val = y if y_val is None else y_val
    res = DistillResult(kind)
    active: list[int] = []
    for _ in range(max_terms):
        best = None
        for c in range(len(terms)):
            if c in active:
                continue
            A = Theta[:, active + [c]]
            w, *_ = np.linalg.lstsq(A, yv, rcond=None)
            err = np.mean((A @ w - yv) ** 2)
            if best is None or err < best[0]:
                best = (err, c, w)
        if best is None:
            break
        active.append(best[1])
        w = best[2]
        expr = sum(
            sp.Float(float(wi)) * terms[c]
            for wi, c in zip(w, active)
            if abs(wi) > lam * (np.abs(w).max() + 1e-30)
        )
        expr = clean(sp.sympify(expr), kind)
        res.candidates.append(
            Candidate(
                expr,
                complexity(expr),
                _mse(expr, X, yv, n_inputs),
                _mse(expr, X_val, y_val.ravel(), n_inputs),
                "library",
            )
        )
    return res


def distill(
    samples: dict[str, dict[str, np.ndarray]],
    engine: str = "auto",
    options: dict | None = None,
    seed: int = 0,
    out_dir=None,
    val_frac: float = 0.25,
    logger=print,
) -> dict[str, DistillResult]:
    """Distil every sampled message function. engine: "pysr" | "library" | "auto"."""
    if engine == "auto":
        engine = "pysr" if pysr_available() else "library"
    rng = np.random.default_rng(seed)
    out: dict[str, DistillResult] = {}
    for kind, s in samples.items():
        X, y = s["X"].astype(np.float64), s["y"][:, 0].astype(np.float64)
        if np.std(y) < 1e-8 * max(1.0, np.abs(y).max(initial=0.0)):
            out[kind] = DistillResult(
                kind, [Candidate(sp.Float(float(np.mean(y))), 1, 0.0, 0.0, "constant")]
            )
            continue
        perm = rng.permutation(len(X))
        nv = int(val_frac * len(X))
        va, tr = perm[:nv], perm[nv:]
        logger(f"[distill] {kind}: {engine} on {len(tr)} samples, {X.shape[1]} inputs")
        if engine == "pysr":
            out[kind] = run_pysr(X[tr], y[tr], kind, options, seed, out_dir, X[va], y[va])
        else:
            out[kind] = run_library(X[tr], y[tr], kind, X_val=X[va], y_val=y[va])
    return out
