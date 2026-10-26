#!/usr/bin/env python
"""Run baselines on a benchmark file with the same split and metrics as HyperDiscover.

    python scripts/run_baselines.py --benchmark-file B.h5 --out runs_baselines/B \
        --methods sindy_oracle sindy_generic this arni corr pcorr granger te
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import to_jsonable, write_provenance  # noqa: E402

ALL = [
    "sindy_oracle",
    "sindy_generic",
    "this",
    "arni",
    "corr",
    "pcorr",
    "granger",
    "te",
]  # + "this_local" (paper's local protocol), "nri", "hypergnn" (trained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--deriv", default="savgol")
    ap.add_argument("--nri-steps", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-tune", action="store_true", help="use fixed default thresholds (no validation tuning)"
    )
    a = ap.parse_args()

    from hyperdiscover import baselines as B
    from hyperdiscover import benchmarks
    from hyperdiscover.baselines.this_reimpl import (
        this_reconstruct,  # GPL-3.0 port (DECISIONS D-040)
    )
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.eval.structure_metrics import (
        edge_metrics,
        false_hyperedge_rate,
        hyperedge_metrics,
    )

    data, spec, truth = benchmarks.load_file(a.benchmark_file)
    system = benchmarks.rebuild_system(truth)
    st = system.structure
    M = data.X.shape[0]
    n_test = max(1, int(round(0.15 * M)))
    X = data.X[: M - n_test]  # same fit split as run_pipeline.py
    dX = estimate_derivatives(
        X, data.t, a.deriv, data.meta.get("segment_starts"), window=9, polyorder=3
    )
    sl = slice(3, X.shape[1] - 3)
    X, dX = X[:, sl], dX[:, sl]
    per = system.periodic
    out_root = Path(a.out)
    # truth-free threshold tuning: hold out the last 20% of the fit trajectories (D-026)
    n_val = max(1, int(round(0.2 * X.shape[0])))
    X_tr, dX_tr, X_va, dX_va = X[:-n_val], dX[:-n_val], X[-n_val:], dX[-n_val:]
    from hyperdiscover.baselines.tuning import tune_sindy, tune_this

    for meth in a.methods:
        out = out_root / meth
        write_provenance(out, {"method": meth, "spec": spec}, int(spec.get("seed", 0)))
        t0 = time.time()
        try:
            tuning = None
            if meth in ("sindy_oracle", "sindy_generic"):
                lib = meth.split("_")[1]
                truth_exprs = system.sympy_exprs() if lib == "oracle" else None
                thr_fit = 0.05
                if not a.no_tune:
                    thr_fit, tuning = tune_sindy(X_tr, dX_tr, X_va, dX_va, lib, truth_exprs, per)
                r = B.sindy_full(X, dX, lib, truth_exprs, per, threshold=thr_fit)
            elif meth == "this":
                # Taylor base point: 0 for phases (as in the THIS paper's Kuramoto example), data mean
                # otherwise — around 0, multiplicative couplings (S2 a·x_i x_j, S5) have no linear x_j
                # term and THIS cannot see the edges (audit E5, DECISIONS D-033)
                x0 = 0.0 if per else None
                lam = 0.1
                if not a.no_tune:
                    lam, tuning = tune_this(X_tr, dX_tr, X_va, dX_va, x0=x0)
                r = this_reconstruct(X, dX, lam=lam, x0=x0)
            elif meth == "this_local":
                # THIS is a local Taylor method (the paper samples a small box around x0 = 0). For phases:
                # co-rotating frame (couplings depend on differences only), wrap to (−π, π], keep samples
                # inside |x| < box; non-periodic systems: samples within one std box of the data mean.
                Xl, dXl = X, dX
                if per:
                    cm = np.angle(np.exp(1j * X[..., 0]).mean(-1, keepdims=True))[..., None]
                    # ẋ_i depends only on phase differences, which the shift and the wrap leave unchanged
                    # (mod 2π), so the derivative targets stay as they are
                    Xl = np.angle(np.exp(1j * (X - cm)))
                Xf, dXf = Xl.reshape(-1, Xl.shape[-2], 1), dXl.reshape(-1, Xl.shape[-2], 1)
                ctr = 0.0 if per else Xf.mean(0, keepdims=True)
                rad = np.abs(Xf - ctr).max(axis=(1, 2))
                keep = rad <= np.quantile(rad, 0.3)  # the 30 % of samples closest to the base point
                Xk, dXk = Xf[keep][None], dXf[keep][None]
                x0 = 0.0 if per else None
                lam = 0.1
                if not a.no_tune:
                    n_k = Xk.shape[1]
                    lam, tuning = tune_this(Xk[:, : n_k * 4 // 5], dXk[:, : n_k * 4 // 5],
                                            Xk[:, n_k * 4 // 5 :], dXk[:, n_k * 4 // 5 :], x0=x0)
                r = this_reconstruct(Xk, dXk, lam=lam, x0=x0)
            elif meth == "arni":
                r = B.arni(X, dX, per)
            elif meth == "corr":
                r = {"pair_scores": B.correlation_scores(X), "tri_scores": {}}
            elif meth == "pcorr":
                r = {"pair_scores": B.partial_correlation_scores(X), "tri_scores": {}}
            elif meth == "granger":
                r = {"pair_scores": B.granger_scores(data.X[: M - n_test]), "tri_scores": {}}
            elif meth == "nri":  # learned baseline: CPU here only for small N; GPU on Ada
                r = B.nri(
                    data.X[: M - n_test],
                    data.t,
                    steps=a.nri_steps,
                    device="cuda" if a.device == "cuda" else "cpu",
                )
            elif meth == "hypergnn":  # SPHINX-style latent incidence (learned, structure only; G3)
                r = B.latent_incidence_baseline(
                    X, dX, steps=a.nri_steps, device="cuda" if a.device == "cuda" else "cpu"
                )
            elif meth == "te":
                r = {
                    "pair_scores": B.transfer_entropy_scores(data.X[: M - n_test]),
                    "tri_scores": {},
                }
            else:
                raise ValueError(meth)
        except (MemoryError, NotImplementedError) as e:
            (out / "baseline_metrics.json").write_text(
                json.dumps({"method": meth, "spec": spec, "failed": repr(e)})
            )
            print(meth, "FAILED:", e)
            continue
        pu, tu = B.directed_to_undirected(r["pair_scores"], r.get("tri_scores", {}))
        # regression baselines: a nonzero coefficient is a detection -> threshold just above 0
        thr = (
            1e-12
            if meth in ("sindy_oracle", "sindy_generic", "this", "this_local", "arni")
            else 0.5
            if meth == "hypergnn"  # memberships / adjacency are probabilities
            else float(np.median(list(pu.values()) or [0]))
        )
        m = {
            "method": meth,
            "spec": spec,
            "edges": edge_metrics(pu, st, thr),
            "seconds": time.time() - t0,
            "threshold": thr,
            "label": r.get("label", meth),
            "tuning": tuning,
        }
        if meth == "this":
            m["this_x0"] = "zero" if per else "data_mean"
        if meth == "sindy_oracle":
            m["oracle_coefficients"] = B.oracle_coefficient_report(r, system.sympy_exprs(), st)
        if len(st.hyperedges):
            # methods that can output hyperedges are always scored (all-zero scores if none survive),
            # so an empty output counts as chance-level ranking instead of being dropped
            triadic = meth in ("sindy_oracle", "sindy_generic", "this", "this_local", "hypergnn")
            m["hyperedges"] = hyperedge_metrics(tu, st, thr) if (tu or triadic) else None
        else:
            m["false_hyperedges"] = false_hyperedge_rate(tu, st.n_nodes, thr)
        (out / "baseline_metrics.json").write_text(
            json.dumps(to_jsonable(m), indent=2, default=float)
        )
        h = m.get("hyperedges") or {}
        oc = m.get("oracle_coefficients")
        print(
            f"{meth:14s} edges AUROC {m['edges']['auroc']:.3f} F1 {m['edges']['f1']:.3f}"
            + (f" | hyper AUROC {h['all']['auroc']:.3f} F1 {h['all']['f1']:.3f}" if h else "")
            + (
                f" | coefficients exact: {oc['exact']} (max rel err {oc['max_coef_rel_err']:.2e},"
                f" spurious {oc['n_spurious_terms']}, missing {oc['n_missing_terms']})"
                if oc
                else ""
            )
            + f"  ({m['seconds']:.1f}s)"
        )


if __name__ == "__main__":
    main()
