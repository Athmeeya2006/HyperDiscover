"""Gradio demo: upload time series (or pick a benchmark) -> hypergraph + equations.

Run:  python -m hyperdiscover.demo.app      (CPU-sized: N ≤ 30, "fast" preset)

CSV format: columns ``traj, t, x0, x1, ..., x{N-1}`` (one row per trajectory and time point,
uniform t within each trajectory).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

MAX_N = 30


def csv_to_array(path: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    xs = [c for c in df.columns if c not in ("traj", "t")]
    trajs = [g.sort_values("t") for _, g in df.groupby("traj")]
    T = min(len(g) for g in trajs)
    X = np.stack([g[xs].to_numpy()[:T] for g in trajs])[..., None]
    t = trajs[0]["t"].to_numpy()[:T]
    return X, t - t[0]


def run(source, system_id, N, noise, hyper_kind, periodic, csv_file, engine):
    import matplotlib.pyplot as plt
    import sympy as sp

    from hyperdiscover import HyperDiscover, benchmarks
    from hyperdiscover.eval.figures import plot_hypergraph, plot_pareto, plot_rollout

    truth = None
    if source == "benchmark":
        b = benchmarks.load(
            system_id,
            N=int(N),
            noise=float(noise),
            hyper_kind=hyper_kind,
            n_hyper=max(1, int(N) // 5),
            M=30,
            T=150,
            seed=0,
        )
        X, t, truth = b.data.X, b.data.t, b.structure
        periodic = b.system.periodic
    else:
        if csv_file is None:
            return None, "Upload a CSV first.", None, None
        X, t = csv_to_array(csv_file.name if hasattr(csv_file, "name") else csv_file)
    if X.shape[2] > MAX_N:
        return None, f"The demo is limited to N ≤ {MAX_N} nodes (got {X.shape[2]}).", None, None
    hd = HyperDiscover(
        config="fast", periodic=bool(periodic), distill_engine=engine, verbose=False
    ).fit(X, t)
    s = hd.summary()
    tmp = Path(tempfile.mkdtemp())
    edges = [(a, b) for a, b, _ in s["pairwise_edges"]]
    hyps = [(a, b, c) for a, b, c, _ in s["hyperedges"]]
    plot_hypergraph(
        X.shape[2], edges, hyps, truth=truth, title="inferred structure", path=tmp / "graph.png"
    )
    eq_md = "### Discovered equations\n\n$$\\dot x_i = f(x_i) + c_i + \\sum_j A_{ij} g + \\sum_{j<k} B_{ijk} h$$\n\n"
    for k, v in hd.equations_.items():
        eq_md += f"- ${k} = {sp.latex(v)}$\n"
    if truth is not None:
        eq_md += "\n### Ground truth\n\n" + "".join(
            f"- ${k} = {v}$\n" for k, v in b.truth["latex"].items()
        )
    pareto_path = None
    if hd.distilled_:
        plot_pareto(hd.distilled_, path=tmp / "pareto.png")
        pareto_path = str(tmp / "pareto.png")
    roll_path = None
    if hd.equations_:
        x0 = X[hd.val_idx_[:1], 0]
        pred = hd.simulate(x0, t)
        plot_rollout(
            t,
            X[hd.val_idx_[:1]],
            pred,
            nodes=tuple(range(min(3, X.shape[2]))),
            path=tmp / "rollout.png",
        )
        roll_path = str(tmp / "rollout.png")
    plt.close("all")
    return str(tmp / "graph.png"), eq_md + "\n\n" + hd.report(), pareto_path, roll_path


def build():
    import gradio as gr

    with gr.Blocks(title="HyperDiscover demo") as demo:
        gr.Markdown(
            "# HyperDiscover\nInfer pairwise edges, 3-node hyperedges and closed-form equations "
            "from node time series. CPU-sized fits only (N ≤ 30, 'fast' preset)."
        )
        with gr.Row():
            with gr.Column():
                source = gr.Radio(["benchmark", "csv"], value="benchmark", label="data source")
                system_id = gr.Dropdown(
                    ["S1", "S2", "S3", "S4", "S5"], value="S4", label="benchmark system"
                )
                N = gr.Slider(6, MAX_N, value=10, step=1, label="N nodes")
                noise = gr.Slider(
                    0.0, 0.1, value=0.0, step=0.01, label="noise (fraction of signal std)"
                )
                hyper_kind = gr.Dropdown(
                    ["open", "closed", "mixed"], value="open", label="hyperedge type"
                )
                periodic = gr.Checkbox(False, label="phase data (CSV only)")
                csv = gr.File(label="CSV (traj, t, x0..x{N-1})")
                engine = gr.Dropdown(
                    ["auto", "library", "pysr"], value="auto", label="symbolic engine"
                )
                btn = gr.Button("Run", variant="primary")
            with gr.Column():
                graph = gr.Image(
                    label="hypergraph (green = correct, orange = false, purple dashed = missed)"
                )
                eqs = gr.Markdown()
        with gr.Row():
            pareto = gr.Image(label="Pareto fronts")
            roll = gr.Image(label="true vs predicted trajectories")
        btn.click(
            run,
            [source, system_id, N, noise, hyper_kind, periodic, csv, engine],
            [graph, eqs, pareto, roll],
        )
    return demo


if __name__ == "__main__":
    build().launch()
