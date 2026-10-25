#!/usr/bin/env python
"""Visual walkthrough of every stage of HyperDiscover.

    # no training: animate S1–S5 on their hypergraphs + Stage 0/1 views (fast, CPU)
    python scripts/visualize_pipeline.py --systems-only --out runs/viz_systems

    # tiny CPU fit with the "fast" preset, then all stage figures + training GIF
    python scripts/visualize_pipeline.py --experiment configs/experiment/sanity_s4.yaml --preset fast --out runs/viz_s4

    # figures for an existing run directory (e.g. a job finished on Ada)
    python scripts/visualize_pipeline.py --run runs/sanity_s4

Writes PNG/GIF files and an index.html that shows them in pipeline order.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = [
    ("00_dynamics.gif", "Data: the system evolving on its hypergraph (node colour = state)."),
    ("01_trajectories.png", "Data: several short trajectories from random initial conditions."),
    ("02_targets.png", "Stage 0: derivative targets and weak-form test functions."),
    ("03_screening.png", "Stage 1: pairwise screening scores (candidates to keep)."),
    (
        "04_gate_evolution.png",
        "Stages 2–3: gate probabilities during pairwise → triadic → joint training.",
    ),
    ("05_training.gif", "Stages 2–3: the inferred hypergraph as gates open/close."),
    ("06_messages.png", "Learned message functions g, h (MLPs) vs ground truth."),
    (
        "07_structure.png",
        "Readout: inferred structure (green = correct, orange = false, purple dashed = missed).",
    ),
    ("08_pareto.png", "Stage 4: symbolic-regression Pareto fronts."),
    (
        "09_rollout.png",
        "Stage 5: discovered equations simulated from a held-out initial condition.",
    ),
    ("10_transition.png", "Out-of-sample test: transition sweep, true vs discovered."),
]


def write_index(out: Path, title: str, extra: str = "") -> None:
    cards = []
    for fname, cap in STEPS:
        if (out / fname).exists():
            cards.append(
                f'<figure><img src="{fname}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
            )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;background:#fff;color:#111}}
figure{{margin:24px 0}}img{{max-width:100%;border:1px solid #ddd}}figcaption{{color:#444}}pre{{background:#f5f5f5;padding:8px;overflow:auto}}</style>
</head><body><h1>{html.escape(title)}</h1>{extra}{"".join(cards)}</body></html>"""
    (out / "index.html").write_text(page)


def systems_only(out: Path, N: int = 12, seed: int = 0) -> None:
    from hyperdiscover import benchmarks
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.data.weak_form import make_weak_form
    from hyperdiscover.eval import figures as F
    from hyperdiscover.screen import screen_pairs

    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        d = out / sid
        d.mkdir(parents=True, exist_ok=True)
        b = benchmarks.load(
            sid, N=N, noise=0.01, hyper_kind="mixed", n_hyper=4, M=20, T=200, seed=seed
        )
        per = b.system.periodic
        F.animate_dynamics(
            b.structure, b.data.X, b.data.t, d / "00_dynamics.gif", periodic=per, title=sid
        )
        F.plot_system_trajectories(
            b.data, path=d / "01_trajectories.png", title=f"{sid}: {b.system.name}"
        )
        dX = estimate_derivatives(b.data.X, b.data.t, "savgol", window=9, polyorder=3)
        F.plot_derivative_targets(b.data, dX, make_weak_form(b.data.t), d / "02_targets.png")
        _, score = screen_pairs(b.data.X, dX, periodic=per)
        F.plot_screening(score, b.structure, d / "03_screening.png")
        F.plot_hypergraph(
            N,
            b.structure.edges,
            b.structure.hyperedges,
            title=f"{sid} ground truth",
            path=d / "07_structure.png",
        )
        eqs = "".join(
            f"<li><code>{k} = {html.escape(v)}</code></li>" for k, v in b.truth["equations"].items()
        )
        write_index(
            d,
            f"{sid} ({b.system.name}) – data and early stages",
            f"<p>Ground truth:</p><ul>{eqs}</ul><p>Closure classes of hyperedges: "
            f"{b.structure.closure_class().tolist()} (0 = open, 3 = closed).</p>",
        )
        print("wrote", d / "index.html")
    links = "".join(
        f'<li><a href="{s}/index.html">{s}</a></li>' for s in ["S1", "S2", "S3", "S4", "S5"]
    )
    (out / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><h1>Benchmark systems</h1><ul>{links}</ul>"
    )


def walkthrough(hd, system, data, out: Path) -> None:
    from hyperdiscover.data.derivatives import estimate_derivatives
    from hyperdiscover.eval import figures as F
    from hyperdiscover.eval.bifurcation import compare_transitions

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = hd.structure_result_
    truth = system.structure
    N = truth.n_nodes
    per = system.periodic
    F.animate_dynamics(
        truth, data.X, data.t, out / "00_dynamics.gif", periodic=per, title=system.system_id
    )
    F.plot_system_trajectories(data, path=out / "01_trajectories.png")
    dX = estimate_derivatives(
        data.X, data.t, hd.config["data"]["deriv_method"], **hd.config["data"]["deriv_kw"]
    )
    F.plot_derivative_targets(data, dX, res.data.wf, out / "02_targets.png")
    from hyperdiscover.screen import screen_pairs

    _, score = screen_pairs(data.X, dX, periodic=per)
    F.plot_screening(score, truth, out / "03_screening.png")
    if res.histories:
        F.plot_gate_evolution(
            res.histories,
            res.pair_index_all,
            res.tri_index_all,
            truth,
            out / "04_gate_evolution.png",
        )
        F.animate_training(
            res.histories, res.pair_index_all, res.tri_index_all, N, truth, out / "05_training.gif"
        )
    F.plot_message_functions(res.model, system, data.X, out / "06_messages.png")
    F.plot_hypergraph(
        N,
        [(a, b) for a, b, _ in hd.pairwise_edges_],
        [(a, b, c) for a, b, c, _ in hd.hyperedges_],
        truth=truth,
        title="inferred vs true",
        path=out / "07_structure.png",
    )
    extra = ""
    if hd.distilled_:
        F.plot_pareto(hd.distilled_, out / "08_pareto.png")
    if hd.equations_:
        disc = hd.discovered_system()
        x0 = data.X_clean[-1:, 0]
        from hyperdiscover.data.simulate import rk4_integrate

        true_traj = rk4_integrate(system.rhs, x0, data.t, 10, system.postprocess)
        with np.errstate(all="ignore"):
            pred = rk4_integrate(disc.rhs, x0, data.t, 10)
        F.plot_rollout(
            data.t, true_traj, pred, nodes=tuple(range(min(3, N))), path=out / "09_rollout.png"
        )
        if system.system_id in ("S4", "S1"):
            kind = "kuramoto" if system.system_id == "S1" else "mean"
            scales = np.linspace(
                0.0 if kind == "kuramoto" else 0.1, 3.0 if kind == "kuramoto" else 2.5, 15
            )
            cmp = compare_transitions(system, disc, scales, "g", kind, t_relax=30.0)
            F.plot_bifurcation(
                cmp, "pairwise coupling scale", "order parameter", out / "10_transition.png"
            )
        extra = (
            "<h2>Discovered equations</h2><pre>"
            + html.escape(json.dumps({k: str(v) for k, v in hd.equations_.items()}, indent=2))
            + "</pre><h2>Ground truth</h2><pre>"
            + html.escape(json.dumps(system.ground_truth()["equations"], indent=2))
            + "</pre>"
        )
    write_index(out, f"HyperDiscover walkthrough: {system.system_id}", extra)
    print("wrote", out / "index.html")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--systems-only", action="store_true")
    g.add_argument("--experiment")
    g.add_argument("--run", help="existing run directory (from run_pipeline.py)")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=12)
    a = ap.parse_args()

    from hyperdiscover import HyperDiscover, benchmarks

    if a.systems_only:
        systems_only(Path(a.out or "runs/viz_systems"), a.N)
        return
    if a.experiment:
        exp = yaml.safe_load(Path(a.experiment).read_text())
        b = benchmarks.load(**exp["benchmark"])
        M = b.data.X.shape[0]
        n_test = max(1, int(round(0.15 * M)))
        hd = HyperDiscover(config=a.preset, periodic=b.system.periodic, distill_engine=a.engine)
        hd.fit(b.data.X[: M - n_test], b.data.t, b.data.meta.get("segment_starts"))
        walkthrough(hd, b.system, b.data, Path(a.out or "runs/viz"))
        return
    run = Path(a.run)
    cfg = json.loads((run / "config.resolved.json").read_text())
    b = benchmarks.load(
        **{
            k: cfg["spec"][k]
            for k in (
                "system_id",
                "N",
                "noise",
                "hyper_kind",
                "graph",
                "mean_degree",
                "pairwise_only",
                "M",
                "T",
                "seed",
            )
        },
        n_hyper=cfg["spec"]["n_hyper"] or None,
    )
    M = b.data.X.shape[0]
    n_test = max(1, int(round(0.15 * M)))
    hd = HyperDiscover.from_structure_checkpoint(
        run / "structure.pt", b.data.X[: M - n_test], b.data.t, a.engine
    )
    if (run / "equations.json").exists():
        import sympy as sp

        eq = json.loads((run / "equations.json").read_text())
        hd.equations_ = {k: sp.sympify(v["str"]) for k, v in eq.items()}
        summ = json.loads((run / "summary.json").read_text())
        if summ.get("node_constants") is not None:
            hd.node_constants_ = np.array(summ["node_constants"]).reshape(-1, 1)
    walkthrough(hd, b.system, b.data, Path(a.out or run / "viz"))


if __name__ == "__main__":
    main()
