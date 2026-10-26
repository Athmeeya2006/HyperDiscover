#!/usr/bin/env python
"""E12 (CLAUDE.md §18.14): transfer discovered f, g, h to unseen networks.

    python scripts/transfer_experiment.py --runs configs/transfer_sources/recipe-n20-v3 --out paper/tables

(configs/transfer_sources/ holds copies of the equations.json files of Kaggle hd-recipe-n20-v3, so the
experiment runs from the code bundle.)

f, g, h are shared and network-independent, so a model discovered on network 1 defines the dynamics on any
network: plug the new structure (A, B) into the discovered equations and integrate. For every source run
(equations.json; the system id and parameters come from the run's benchmark spec) and every target network
(graph family × N × hyperedge kind, true structure given), we compare the discovered model with the ground
truth on the target:
  * traj_nrmse: trajectory nRMSE over ``--horizon`` saved steps from M random (sparse) initial conditions;
  * final_err: |mean final state| difference (steady state);
  * sweep: coupling multiplier κ applied to g and h in BOTH systems (g_scale, h_scale); the final mean state
    from a low and a high initial condition (hysteresis); kappa_c = smallest κ with low-IC final mean above
    ``--onset`` (transition onset, e.g. the SIS epidemic threshold); reported for truth and model.
The pairwise-only model (equations of the source system's pairwise-only run, applied to the target's edges,
no hyperedges) is the comparison that shows whether the triadic term matters for the prediction.
Node constants c_i (S1 natural frequencies) are node properties, not learned couplings: the target's true c_i
are used for both models (stated in the paper). Writes transfer.csv / transfer.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _source_runs(root: Path, noise: float) -> list[Path]:
    out = []
    for run in sorted(root.iterdir()):
        eq = run / "seed0" / "equations.json"
        if eq.exists() and f"_noise{noise:g}_" in run.name and "_pw_" not in run.name:
            out.append(run / "seed0")
    return out


def _pairwise_twin(src: Path) -> Path | None:
    """The pairwise-only run of the same system / N / noise / seed (equations fitted without hyperedges)."""
    name = src.parent.name
    for kind in ("_open_", "_closed_", "_mixed_"):
        if kind in name:
            twin = src.parent.parent / name.replace(kind, "_pw_") / "seed0"
            return twin if (twin / "equations.json").exists() else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="configs/transfer_sources/recipe-n20-v3")
    ap.add_argument("--noise", type=float, default=0.0, help="source runs with this noise level")
    ap.add_argument("--graphs", nargs="+", default=["er", "ba", "ws"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[20, 50, 100])
    ap.add_argument("--kinds", nargs="+", default=["open", "closed"])
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--kappas", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--onset", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd
    import sympy as sp

    from hyperdiscover import benchmarks
    from hyperdiscover.data.simulate import rk4_integrate, sparse_ic
    from hyperdiscover.distill.refit import SymbolicModel

    def load_exprs(run: Path) -> dict:
        eq = json.loads((run / "equations.json").read_text())
        return {k: sp.sympify(v["str"]) for k, v in eq.items()}

    def final_mean(sysm, x0, T):  # trajectories [B, T + 1, N, d]
        t = np.arange(T + 1) * sysm.default_dt
        xt = rk4_integrate(sysm.rhs, x0, t, 10, getattr(sysm, "postprocess", None))
        return xt

    rows = []
    for src in _source_runs(Path(a.runs), a.noise):
        sid = src.parent.name.split("_")[0]
        src_kind = src.parent.name.split("_")[1]
        exprs = load_exprs(src)
        twin = _pairwise_twin(src)
        exprs_pw = load_exprs(twin) if twin else None
        if exprs_pw is not None:
            exprs_pw["h"] = sp.Integer(0)
        for graph, N, kind in itertools.product(a.graphs, a.Ns, a.kinds):
            try:
                tgt = benchmarks.load(sid, N=N, hyper_kind=kind, graph=graph, M=2, T=2,
                                      seed=a.seed + 100)
            except Exception as e:  # graph family not supported for this N etc.
                print("skip", sid, graph, N, kind, e)
                continue
            truth = tgt.system
            st = tgt.structure
            c = truth.c if truth.has_node_constants else None
            per = sid == "S1"
            models = {"discovered": SymbolicModel(st, exprs, c, per)}
            if exprs_pw is not None:
                models["pairwise_only"] = SymbolicModel(st.pairwise_only(), exprs_pw, c, per)
            rng = np.random.default_rng([a.seed, N, int(sid[1:])])
            x0 = sparse_ic(truth, a.M, rng)
            xt_true = final_mean(truth, x0, a.horizon)
            scale = float(xt_true.std()) + 1e-12
            lo = np.full((1, N, 1), truth.ic_box[0] + 0.01 * (truth.ic_box[1] - truth.ic_box[0]))
            hi = np.full((1, N, 1), truth.ic_box[1] - 0.01 * (truth.ic_box[1] - truth.ic_box[0]))
            lohi = np.concatenate([lo, hi])  # both initial conditions in one integration
            T_long = 10 * a.horizon
            sweep_true = {}
            for kap in a.kappas:
                truth.g_scale = truth.h_scale = kap
                fin = final_mean(truth, lohi, T_long)[:, -1].mean(axis=(-2, -1))
                sweep_true[kap] = (float(fin[0]), float(fin[1]))
            truth.g_scale = truth.h_scale = 1.0
            kc_true = next((k for k in a.kappas if sweep_true[k][0] > a.onset), np.nan)
            for mname, model in models.items():
                with np.errstate(all="ignore"):
                    xt = final_mean(model, x0, a.horizon)
                    ok = np.isfinite(xt).all()
                    sweep = {}
                    for kap in a.kappas:
                        model.g_scale = model.h_scale = kap
                        fin = final_mean(model, lohi, T_long)[:, -1].mean(axis=(-2, -1))
                        sweep[kap] = (float(fin[0]), float(fin[1]))
                    model.g_scale = model.h_scale = 1.0
                kc = next((k for k in a.kappas if sweep[k][0] > a.onset), np.nan)
                sw_err = float(np.nanmean([abs(sweep[k][q] - sweep_true[k][q])
                                           for k in a.kappas for q in (0, 1)]))
                row = dict(
                    system=sid, source_kind=src_kind, source_N=int(src.parent.name.split("_N")[1].split("_")[0]),
                    graph=graph, N=N, kind=kind, model=mname,
                    traj_nrmse=float(np.sqrt(np.mean((xt - xt_true) ** 2)) / scale) if ok else np.nan,
                    final_err=float(abs(xt[:, -1].mean() - xt_true[:, -1].mean())) if ok else np.nan,
                    sweep_mae=sw_err, kappa_c_true=kc_true, kappa_c_pred=kc,
                    hysteresis_true=float(np.mean([sweep_true[k][1] - sweep_true[k][0] for k in a.kappas])),
                    hysteresis_pred=float(np.nanmean([sweep[k][1] - sweep[k][0] for k in a.kappas])),
                )
                rows.append(row)
                print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items()), flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "transfer.csv", index=False)
    (out / "transfer.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "transfer.csv")


if __name__ == "__main__":
    main()
