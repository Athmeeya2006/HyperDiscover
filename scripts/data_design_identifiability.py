#!/usr/bin/env python
"""How much does the data design change the pure-interaction fraction of the true h? (no training)

    python scripts/data_design_identifiability.py --out paper/tables

For the S4 sanity instance and every system / hyperedge kind at N, simulates the same structure under
several observation designs (baseline, state kicks, many short trajectories, sparse initial conditions)
and reports the held-out pure-interaction fraction (eval/identifiability.py). Higher = triadic effect
less mimicable by pairwise functions on the hyperedge's member pairs.
Writes data_design_identifiability.csv / .md. (Supersedes scripts/identifiability_data_design.py.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# name -> (benchmark overrides, sparse initial conditions)
DESIGNS = {
    "base_M50_T200": ({}, False),
    "kicks_0.2_every20": ({"kick_std": 0.2, "kick_every": 20}, False),
    "kicks_0.3_every20": ({"kick_std": 0.3, "kick_every": 20}, False),
    "kicks_0.5_every10": ({"kick_std": 0.5, "kick_every": 10}, False),
    "short_M400_T25": ({"M": 400, "T": 25}, False),
    "short_M500_T20": ({"M": 500, "T": 20}, False),
    "short_kicks_M200_T50": ({"M": 200, "T": 50, "kick_std": 0.3, "kick_every": 10}, False),
    "sparse_ic": ({}, True),
    "sparse_ic_short_M500_T20": ({"M": 500, "T": 20}, True),
}
SANITY = {"N": 10, "hyper_kind": "open", "n_hyper": 2, "graph": "er", "mean_degree": 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["S2", "S4", "S5"])
    ap.add_argument("--kinds", nargs="+", default=["open", "closed"])
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--no-sanity", action="store_true", help="skip the S4 N=10 sanity instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    import pandas as pd

    from hyperdiscover import benchmarks
    from hyperdiscover.data.simulate import simulate_system, sparse_ic
    from hyperdiscover.eval.identifiability import identifiability_report

    cases = [] if a.no_sanity else [("S4", {**SANITY, "case": "sanity"})]
    cases += [(s, {"N": a.N, "hyper_kind": k}) for s in a.systems for k in a.kinds]
    rows = []
    for sid, case in cases:
        case = dict(case)
        label = case.pop("case", "grid")
        for name, (design, sparse) in DESIGNS.items():
            spec = {"M": 50, "T": 200, **design}
            b = benchmarks.load(sid, **case, **spec, seed=a.seed)
            X = b.data.X_clean
            if sparse:
                rng = np.random.default_rng([a.seed, 7])
                x0 = sparse_ic(b.system, spec["M"], rng)
                X = simulate_system(b.system, spec["M"], spec["T"], None, rng, x0=x0).X_clean
            rep = identifiability_report(b.system, X, a.n)
            rows.append(
                {
                    "system": sid,
                    "case": label,
                    "N": case["N"],
                    "kind": case["hyper_kind"],
                    "design": name,
                    "M": spec["M"],
                    "T": spec["T"],
                    "pure_fraction": rep["pure_fraction"],
                    "tri_var_ratio": rep["tri_var_ratio"],
                    "h_std": rep["h_std"],
                }
            )
            print(
                f"{sid} {label:6s} N{case['N']} {case['hyper_kind']:6s} {name:26s} "
                f"pure {rep['pure_fraction']:.3f}  h_std {rep['h_std']:.3f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "data_design_identifiability.csv", index=False)
    (out / "data_design_identifiability.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("wrote", out / "data_design_identifiability.csv")


if __name__ == "__main__":
    main()
