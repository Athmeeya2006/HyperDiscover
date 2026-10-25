#!/usr/bin/env python
"""Validate the Python THIS port against the official Julia code (CLAUDE.md §8.2).

    python scripts/validate_this.py --this-dir <unzipped THIS-v2.0> --julia <julia binary> --out runs/validate_this

Needs a Julia with the package Combinatorics (installed on first use into
``--julia-depot``). The official code is NOT vendored here: download it from Zenodo
(doi:10.5281/zenodo.10530470) or GitHub (TaylorBasedHypergraphInference/THIS, tag v2.0).

1. Equivalence: identical (X, Y) are given to the official ``this`` and to
   ``hyperdiscover.baselines.this_reimpl.this_official``; coefficient matrices must agree to
   ~1e-8 (relative).
2. Paper-style example (Fig. 2 setup as described in the paper: n = 7 Kuramoto oscillators
   with pairwise + triadic coupling, samples drawn uniformly in a hypercube of side δ around
   x0 = 0, exact derivatives, 10–150 samples). The paper's exact hypergraph, frequencies and
   coupling strengths are not given in the text, so we draw them at random and report AUROC
   vs the number of samples; this is a qualitative comparison only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import to_jsonable  # noqa: E402

JULIA_DRIVER = r"""
using DelimitedFiles
include(joinpath(ARGS[1], "this.jl"))
X = readdlm(ARGS[2], ',', Float64)
Y = readdlm(ARGS[3], ',', Float64)
dmax = parse(Int64, ARGS[5]); λ = parse(Float64, ARGS[6]); ρ = parse(Float64, ARGS[7])
Ainf, coeff, relerr = this(X, Y, collect(1:dmax+1), dmax, λ, ρ, 10)
writedlm(ARGS[4], coeff, ',')
"""


def kuramoto_samples(
    n: int,
    n_samples: int,
    delta: float,
    rng,
    n_tri: int = 3,
    p_edge: float = 0.4,
    kind: str = "mixed",
):
    from hyperdiscover.graphs import make_structure
    from hyperdiscover.systems import make_system

    st = make_structure(n, rng, "er", p_edge * (n - 1), kind, n_hyper=n_tri)
    sysm = make_system("S1", st, {"K1": 1.0, "K2": 1.0}, rng=rng)
    X = rng.uniform(-delta / 2, delta / 2, (n_samples, n, 1))
    return sysm, X, sysm.rhs(X)


def run_julia(julia, this_dir, X, Y, dmax, lam, rho, work: Path, depot: Path) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    np.savetxt(work / "X.csv", X, delimiter=",")
    np.savetxt(work / "Y.csv", Y, delimiter=",")
    (work / "driver.jl").write_text(JULIA_DRIVER)
    env = {**os.environ, "JULIA_DEPOT_PATH": str(depot), "JULIA_PROJECT": str(work / "env")}
    setup = 'import Pkg; Pkg.activate(ENV["JULIA_PROJECT"]); for p in ("Combinatorics", "DelimitedFiles") ; haskey(Pkg.project().dependencies, p) || Pkg.add(p); end'
    subprocess.run([julia, "-e", setup], check=True, env=env, capture_output=True, text=True)
    out = work / "coeff.csv"
    r = subprocess.run(
        [
            julia,
            str(work / "driver.jl"),
            str(this_dir),
            str(work / "X.csv"),
            str(work / "Y.csv"),
            str(out),
            str(dmax),
            str(lam),
            str(rho),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return np.loadtxt(out, delimiter=",", ndmin=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--this-dir", required=True)
    ap.add_argument("--julia", required=True)
    ap.add_argument("--julia-depot", default=None)
    ap.add_argument("--out", default="runs/validate_this")
    a = ap.parse_args()

    from hyperdiscover.baselines.sindy_full import directed_to_undirected
    from hyperdiscover.baselines.this_reimpl import this_official, this_reconstruct
    from hyperdiscover.eval.structure_metrics import edge_metrics, hyperedge_metrics

    out = Path(a.out)
    depot = Path(a.julia_depot or out / "julia_depot")
    rng = np.random.default_rng(0)
    report: dict = {"equivalence": [], "paper_style": []}

    # 1. equivalence on several inputs (different n, dmax, λ)
    for case, (n, T, dmax, lam, rho) in enumerate(
        [(5, 60, 2, 0.1, 1.0), (7, 150, 2, 0.05, 0.5), (4, 80, 3, 0.1, 1.0)]
    ):
        _, Xs, Ys = kuramoto_samples(n, T, 1.0, rng, n_tri=1, kind="closed")
        X, Y = Xs[..., 0].T.copy(), Ys[..., 0].T.copy()  # [n, T] like the Julia code
        _, Xi_py, _ = this_official(X, Y, dmax, lam, rho)
        Xi_jl = run_julia(
            a.julia, Path(a.this_dir).resolve(), X, Y, dmax, lam, rho, out / f"case{case}", depot
        )
        diff = float(np.abs(Xi_py - Xi_jl).max() / (np.abs(Xi_jl).max() + 1e-300))
        same_support = bool(np.array_equal(np.abs(Xi_py) > 1e-8, np.abs(Xi_jl) > 1e-8))
        report["equivalence"].append(
            {
                "n": n,
                "T": T,
                "dmax": dmax,
                "lambda": lam,
                "rho": rho,
                "max_rel_diff": diff,
                "same_support": same_support,
            }
        )
        print(
            f"[equivalence] n={n} T={T} dmax={dmax} λ={lam} ρ={rho}: max rel diff {diff:.2e}, same support {same_support}"
        )

    # 2. paper-style Kuramoto example (qualitative)
    for delta in (0.1, 1.0):
        for dmax in (2, 3):
            for n_samples in (10, 25, 50, 100, 150):
                aucs_e, aucs_h = [], []
                for rep in range(5):
                    sysm, Xs, Ys = kuramoto_samples(
                        7, n_samples, delta, np.random.default_rng(100 + rep)
                    )
                    r = this_reconstruct(Xs[:, None], Ys[:, None], dmax=dmax, lam=0.0, x0=0.0)
                    pu, tu = directed_to_undirected(r["pair_scores"], r["tri_scores"])
                    aucs_e.append(edge_metrics(pu, sysm.structure)["auroc"])
                    aucs_h.append(
                        hyperedge_metrics(tu, sysm.structure, by_class=False)["all"]["auroc"]
                    )
                row = {
                    "delta": delta,
                    "dmax": dmax,
                    "n_samples": n_samples,
                    "edge_auroc": float(np.nanmean(aucs_e)),
                    "hyper_auroc": float(np.nanmean(aucs_h)),
                }
                report["paper_style"].append(row)
                print(
                    f"[paper-style] δ={delta} dmax={dmax} samples={n_samples}: edge AUROC {row['edge_auroc']:.3f}, "
                    f"hyperedge AUROC {row['hyper_auroc']:.3f}"
                )
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(to_jsonable(report), indent=2))
    ok = all(e["max_rel_diff"] < 1e-6 and e["same_support"] for e in report["equivalence"])
    print("EQUIVALENCE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
