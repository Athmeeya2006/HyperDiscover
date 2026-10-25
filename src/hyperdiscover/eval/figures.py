"""Figures: every plot is generated from saved arrays/metrics, never hand-edited.

Visual walkthrough helpers (used by ``scripts/visualize_pipeline.py`` and the demo):
  * ``plot_system_trajectories``  – what the observed data look like;
  * ``plot_hypergraph``           – the (true / inferred) structure, triangles shaded,
                                    TP / FP / FN colour-coded when a truth is given;
  * ``plot_gate_evolution``       – gate probabilities over training steps per stage;
  * ``animate_training``          – GIF of the hypergraph as gates open and close;
  * ``plot_message_functions``    – learned g/h (MLP) vs ground truth on data slices;
  * ``plot_pareto``               – symbolic-regression Pareto fronts;
  * ``plot_rollout``              – true vs discovered trajectories;
  * ``plot_bifurcation``          – out-of-sample transition sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C_TP, C_FP, C_FN, C_GREY = "#1b9e77", "#d95f02", "#7570b3", "#999999"


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def layout(n_nodes: int, edges, seed: int = 0) -> dict:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(map(int, e)) for e in edges])
    return nx.spring_layout(G, seed=seed)


def plot_system_trajectories(data, n_traj: int = 3, nodes=None, path=None, title=""):
    X, t = data.X, data.t
    nodes = list(range(min(X.shape[2], 10))) if nodes is None else nodes
    fig, axes = plt.subplots(1, n_traj, figsize=(4 * n_traj, 3), sharey=True, squeeze=False)
    for m in range(n_traj):
        ax = axes[0, m]
        for n in nodes:
            ax.plot(t, X[m, :, n, 0], lw=1)
        ax.set_xlabel("t")
        ax.set_title(f"trajectory {m}")
    axes[0, 0].set_ylabel("x")
    fig.suptitle(title)
    return _save(fig, path)


def plot_hypergraph(
    n_nodes: int,
    edges,
    hyperedges,
    pos=None,
    truth=None,
    ax=None,
    edge_alpha=None,
    tri_alpha=None,
    title: str = "",
    path=None,
):
    """Draw edges and shaded triangles. With ``truth`` (a Structure), predicted items are coloured
    TP (green) / FP (orange) and missed true items are drawn dashed (FN, purple)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 5))
    pos = pos if pos is not None else layout(n_nodes, edges)
    t_edges = {tuple(sorted(map(int, e))) for e in truth.edges} if truth is not None else set()
    t_hyp = {tuple(sorted(map(int, h))) for h in truth.hyperedges} if truth is not None else set()
    p_edges = [tuple(sorted(map(int, e))) for e in edges]
    p_hyp = [tuple(sorted(map(int, h))) for h in hyperedges]
    for q, h in enumerate(p_hyp):
        col = C_GREY if truth is None else (C_TP if h in t_hyp else C_FP)
        a = 0.35 if tri_alpha is None else 0.05 + 0.45 * float(tri_alpha[q])
        ax.add_patch(Polygon([pos[v] for v in h], closed=True, fc=col, ec=col, alpha=a, lw=1))
    for h in t_hyp - set(p_hyp):
        ax.add_patch(
            Polygon([pos[v] for v in h], closed=True, fill=False, ec=C_FN, ls="--", lw=1.5)
        )
    for q, e in enumerate(p_edges):
        col = "k" if truth is None else (C_TP if e in t_edges else C_FP)
        a = 0.9 if edge_alpha is None else float(edge_alpha[q])
        ax.plot(*zip(pos[e[0]], pos[e[1]]), color=col, alpha=a, lw=1.5)
    for e in t_edges - set(p_edges):
        ax.plot(*zip(pos[e[0]], pos[e[1]]), color=C_FN, ls="--", lw=1)
    xy = np.array([pos[v] for v in range(n_nodes)])
    ax.scatter(xy[:, 0], xy[:, 1], s=180, c="white", edgecolors="k", zorder=3)
    for v in range(n_nodes):
        ax.text(*pos[v], str(v), ha="center", va="center", fontsize=7, zorder=4)
    ax.set_axis_off()
    ax.set_title(title)
    return _save(ax.figure, path) if own else ax


def _stage_snapshots(histories):
    """Concatenate gate snapshots across stages on the *screened* candidate index space."""
    steps, pairs, tris, stages = [], [], [], []
    offset = 0
    n_pair = max((int(h["pair_ids"].max()) + 1 if len(h["pair_ids"]) else 0) for h in histories)
    n_tri = max((int(h["tri_ids"].max()) + 1 if len(h["tri_ids"]) else 0) for h in histories)
    for h in histories:
        for s, pp, pt in zip(h["step"], h["pair_probs"], h["tri_probs"]):
            full_p = np.zeros(n_pair)
            full_t = np.zeros(n_tri)
            if len(pp) == len(h["pair_ids"]):
                full_p[h["pair_ids"]] = pp
            if len(pt) == len(h["tri_ids"]):
                full_t[h["tri_ids"]] = pt
            steps.append(offset + s)
            pairs.append(full_p)
            tris.append(full_t)
            stages.append(h["stage"])
        offset += (h["step"][-1] + 1) if h["step"] else 0
    return np.array(steps), np.array(pairs), np.array(tris), stages


def plot_gate_evolution(histories, pair_index, tri_index, truth=None, path=None):
    """Gate probability trajectories; true items in green, others grey."""
    steps, P, T, stages = _stage_snapshots(histories)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    t_pairs = {tuple(map(int, p)) for p in truth.pair_index} if truth is not None else set()
    t_tris = {tuple(map(int, p)) for p in truth.tri_index} if truth is not None else set()
    for ax, M, idx, tset, name in (
        (axes[0], P, pair_index, t_pairs, "pairwise gates"),
        (axes[1], T, tri_index, t_tris, "triadic gates"),
    ):
        if M.size == 0:
            ax.set_title(f"{name} (none)")
            continue
        for q in range(M.shape[1]):
            is_true = tuple(map(int, idx[q])) in tset
            ax.plot(
                steps,
                M[:, q],
                color=C_TP if is_true else C_GREY,
                alpha=0.9 if is_true else 0.25,
                lw=1.4 if is_true else 0.6,
                zorder=3 if is_true else 1,
            )
        bounds = [steps[i] for i in range(1, len(stages)) if stages[i] != stages[i - 1]]
        for b in bounds:
            ax.axvline(b, color="k", ls=":", lw=0.8)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training step (all stages)")
        ax.set_ylabel("P(gate on)")
        ax.set_title(f"{name}: green = true, grey = candidate")
    return _save(fig, path)


def animate_training(
    histories, pair_index, tri_index, n_nodes, truth=None, path="training.gif", fps=6, max_frames=60
):
    """GIF: the hypergraph with edge/triangle opacity = current gate probability."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ..pipeline import undirected_pair_scores, undirected_tri_scores

    steps, P, T, stages = _stage_snapshots(histories)
    frames = np.unique(np.linspace(0, len(steps) - 1, min(max_frames, len(steps))).astype(int))
    pos = layout(n_nodes, truth.edges if truth is not None else pair_index)
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(f):
        ax.clear()
        ep = undirected_pair_scores(pair_index[: P.shape[1]], P[f])
        tp = undirected_tri_scores(tri_index[: T.shape[1]], T[f]) if T.size else {}
        edges = [k for k, v in ep.items() if v > 0.05]
        hyps = [k for k, v in tp.items() if v > 0.05]
        plot_hypergraph(
            n_nodes,
            edges,
            hyps,
            pos,
            truth,
            ax,
            [ep[e] for e in edges],
            [tp[h] for h in hyps],
            title=f"{stages[f]} stage, step {steps[f]}",
        )

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_message_functions(model, true_system, X, path=None, n_grid: int = 60):
    """Compare learned g, h with ground truth on 1-D slices through the data range.

    g: vary x_j at fixed x_i ∈ {q25, q50, q75}. h: vary x_j = x_k at fixed x_i (diagonal slice)
    and vary x_j at fixed x_k (off-diagonal). Learned curves include gate amplitude 1.
    """
    import torch

    lo, hi = np.quantile(X, [0.02, 0.98])
    qs = np.quantile(X, [0.25, 0.5, 0.75])
    grid = np.linspace(lo, hi, n_grid)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    def learned(fn, *cols):
        with torch.no_grad():
            ts = [torch.as_tensor(c[:, None], dtype=torch.float32) for c in cols]
            return fn(*ts).numpy()[:, 0]

    for q, xi in enumerate(qs):
        a = np.full_like(grid, xi)
        col = f"C{q}"
        axes[0].plot(
            grid,
            true_system.g(a[:, None], grid[:, None])[:, 0],
            color=col,
            lw=2,
            alpha=0.5,
            label=f"true, x_i={xi:.2f}",
        )
        axes[0].plot(grid, learned(model.msg.g, a, grid), color=col, ls="--", label="learned")
        axes[1].plot(
            grid,
            true_system.h(a[:, None], grid[:, None], grid[:, None])[:, 0],
            color=col,
            lw=2,
            alpha=0.5,
        )
        axes[1].plot(grid, learned(model.msg.h, a, grid, grid), color=col, ls="--")
        xk = np.full_like(grid, qs[1])
        axes[2].plot(
            grid,
            true_system.h(a[:, None], grid[:, None], xk[:, None])[:, 0],
            color=col,
            lw=2,
            alpha=0.5,
        )
        axes[2].plot(grid, learned(model.msg.h, a, grid, xk), color=col, ls="--")
    axes[0].set_title("g(x_i, x_j) vs x_j")
    axes[1].set_title("h(x_i, x_j, x_k = x_j)")
    axes[2].set_title(f"h(x_i, x_j, x_k = {qs[1]:.2f})")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("x_j")
    fig.suptitle("solid = ground truth, dashed = learned MLP (before symbolic distillation)")
    return _save(fig, path)


def plot_pareto(distilled: dict, path=None):
    kinds = [k for k in ("f", "g", "h") if k in distilled]
    fig, axes = plt.subplots(
        1, max(1, len(kinds)), figsize=(4.5 * max(1, len(kinds)), 3.5), squeeze=False
    )
    for ax, k in zip(axes[0], kinds):
        cs = distilled[k].candidates
        ax.scatter([c.complexity for c in cs], [max(c.val_loss, 1e-16) for c in cs], s=10, c=C_GREY)
        fr = distilled[k].pareto()
        ax.plot([c.complexity for c in fr], [max(c.val_loss, 1e-16) for c in fr], "o-", color=C_TP)
        ax.set_yscale("log")
        ax.set_xlabel("complexity")
        ax.set_ylabel("validation MSE")
        ax.set_title(f"{k}: Pareto front")
    return _save(fig, path)


def plot_rollout(t, true_traj, pred_traj, nodes=(0, 1, 2), traj: int = 0, path=None):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for q, n in enumerate(nodes):
        ax.plot(t, true_traj[traj, :, n, 0], color=f"C{q}", lw=2, alpha=0.5, label=f"node {n} true")
        ax.plot(t, pred_traj[traj, :, n, 0], color=f"C{q}", ls="--", label=f"node {n} discovered")
    ax.set_xlabel("t")
    ax.legend(fontsize=7)
    ax.set_title("held-out initial condition: true vs discovered model")
    return _save(fig, path)


def plot_bifurcation(cmp: dict, xlabel="coupling scale", ylabel="order parameter", path=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for key, ls in (("true", "-"), ("pred", "--")):
        sw = cmp[key]
        ax.plot(sw["scales"], sw["forward"], ls, color="C0", label=f"{key} forward")
        ax.plot(sw["scales"], sw["backward"], ls, color="C3", label=f"{key} backward")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_metric_curve(df, x: str, y: str, hue: str, path=None, logx=False):
    """Robustness / scaling curve from a metrics DataFrame (mean ± std over seeds)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for key, grp in df.groupby(hue):
        g = grp.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.errorbar(g[x], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=2, label=str(key))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=7)
    return _save(fig, path)


def animate_dynamics(
    structure,
    X: np.ndarray,
    t: np.ndarray,
    path="dynamics.gif",
    traj: int = 0,
    fps: int = 10,
    max_frames: int = 80,
    periodic: bool = False,
    title: str = "",
):
    """GIF: node colours = state x_i(t) on the hypergraph (left) + the time series (right)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    N = structure.n_nodes
    pos = layout(N, structure.edges)
    Y = X[traj, :, :, 0]
    if periodic:
        Y = np.mod(Y, 2 * np.pi)
    vmin, vmax = (0, 2 * np.pi) if periodic else (float(Y.min()), float(Y.max()))
    cmap = plt.get_cmap("twilight" if periodic else "viridis")
    frames = np.unique(np.linspace(0, len(t) - 1, min(max_frames, len(t))).astype(int))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    xy = np.array([pos[v] for v in range(N)])
    for h in structure.hyperedges:
        a1.add_patch(
            Polygon([pos[int(v)] for v in h], closed=True, fc="#e7298a", alpha=0.18, ec="#e7298a")
        )
    for e in structure.edges:
        a1.plot(*zip(pos[int(e[0])], pos[int(e[1])]), color="k", lw=0.8, alpha=0.6)
    sc = a1.scatter(
        xy[:, 0], xy[:, 1], s=200, c=Y[0], cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="k", zorder=3
    )
    fig.colorbar(sc, ax=a1, fraction=0.04, label="x_i (mod 2π)" if periodic else "x_i")
    a1.set_axis_off()
    for n in range(min(N, 12)):
        a2.plot(t, Y[:, n], lw=0.8, alpha=0.8)
    line = a2.axvline(t[0], color="k")
    a2.set_xlabel("t")
    a2.set_title("node time series (first 12 nodes)")

    def draw(f):
        sc.set_array(Y[f])
        line.set_xdata([t[f], t[f]])
        a1.set_title(f"{title}  t = {t[f]:.2f}  (pink = 3-node hyperedges)")
        return sc, line

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_derivative_targets(
    data, dX_est: np.ndarray, weak, path=None, node: int = 0, traj: int = 0
):
    """Stage 0: true ẋ vs estimated ẋ, and the weak-form test functions on the same axis."""
    t = data.t
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(t, data.X[traj, :, node, 0], color="C0", lw=1, label="observed x")
    if data.X_clean is not None:
        axes[0].plot(t, data.X_clean[traj, :, node, 0], color="k", lw=0.8, ls=":", label="clean x")
    for s in weak.starts[::3]:
        tt = t[s : s + weak.length]
        axes[0].fill_between(
            tt,
            data.X[traj, :, node, 0].min(),
            data.X[traj, :, node, 0].min()
            + weak.phi / weak.phi.max() * 0.15 * np.ptp(data.X[traj, :, node, 0]),
            color="C2",
            alpha=0.25,
        )
    axes[0].set_title("states + weak-form test functions φ (green)")
    axes[0].legend(fontsize=7)
    if data.dX_clean is not None:
        axes[1].plot(
            t, data.dX_clean[traj, :, node, 0], color="k", lw=1.5, alpha=0.6, label="true ẋ"
        )
    axes[1].plot(t, dX_est[traj, :, node, 0], color="C3", ls="--", lw=1, label="estimated ẋ")
    axes[1].set_title("derivative targets (Stage 0, option A)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("t")
    return _save(fig, path)


def plot_screening(score: np.ndarray, structure, path=None):
    """Stage 1: pairwise screening score matrix with true edges marked."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    S = np.where(np.isfinite(score), score, np.nan)
    im = ax.imshow(S, cmap="Greys")
    for a, b in structure.pair_index:
        ax.scatter(b, a, marker="s", s=12, facecolors="none", edgecolors=C_TP, lw=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="screening score")
    ax.set_xlabel("source j")
    ax.set_ylabel("target i")
    ax.set_title("pairwise screening (green = true edge)")
    return _save(fig, path)
