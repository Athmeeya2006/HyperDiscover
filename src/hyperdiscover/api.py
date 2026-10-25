"""Public sklearn-style estimator.

from hyperdiscover import HyperDiscover
hd = HyperDiscover(state_dim=1, max_order=3, config="default").fit(X, t)   # X: [M, T, N, d]
hd.pairwise_edges_, hd.hyperedges_, hd.equations_
hd.simulate(x0, t); hd.report()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
import yaml

from .data.observe import train_val_test_split
from .data.simulate import rk4_integrate
from .distill import distill, sample_messages, select_model
from .distill.refit import SymbolicModel
from .pipeline import DEFAULT_CONFIG, StructureResult, fit_structure, merge_config
from .systems.base import Structure

PRESETS: dict[str, dict] = {
    "default": {},
    # CPU-sized preset for the demo, tests and quick looks (not for paper numbers)
    "fast": {
        "train": {"pairwise": {"steps": 800}, "triadic": {"steps": 800}, "joint": {"steps": 400}},
        "model": {"hidden": 32, "layers": 2},
    },
}


def resolve_config(config: str | dict | None) -> dict:
    if config is None:
        return merge_config(DEFAULT_CONFIG, {})
    if isinstance(config, dict):
        return merge_config(DEFAULT_CONFIG, config)
    if config in PRESETS:
        return merge_config(DEFAULT_CONFIG, PRESETS[config])
    p = Path(config)
    if p.exists():
        return merge_config(DEFAULT_CONFIG, yaml.safe_load(p.read_text()) or {})
    raise ValueError(f"unknown config {config!r}: preset name {sorted(PRESETS)} or a YAML path")


class HyperDiscover:
    """Joint discovery of hypergraph structure and symbolic equations.

    Args:
        state_dim: d (only d = 1 is supported end-to-end for now).
        max_order: 2 (pairwise only) or 3 (pairwise + triadic).
        config: preset name, YAML path or dict overriding ``pipeline.DEFAULT_CONFIG``.
        periodic: inputs are phases (sin/cos embedding; node constants on by default).
        distill_engine: "pysr" | "library" | "auto" | "none" (skip symbolic stages).
    """

    def __init__(
        self,
        state_dim: int = 1,
        max_order: int = 3,
        config: str | dict | None = "default",
        periodic: bool = False,
        distill_engine: str = "auto",
        pysr_options: dict | None = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.state_dim = state_dim
        self.max_order = max_order
        self.config = resolve_config(config)
        self.config.update({"max_order": max_order, "periodic": periodic, "seed": seed})
        self.periodic = periodic
        self.distill_engine = distill_engine
        self.pysr_options = pysr_options
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        segment_starts=None,
        val_fraction: float = 0.2,
        out_dir: str | Path | None = None,
    ) -> HyperDiscover:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X[..., None]
        if X.shape[-1] != self.state_dim:
            raise ValueError(f"X has d={X.shape[-1]}, expected state_dim={self.state_dim}")
        if self.state_dim != 1:
            raise NotImplementedError("d > 1 is a stretch goal; only d = 1 is supported end-to-end")
        rng = np.random.default_rng(self.seed)
        M = X.shape[0]
        tr, va, te = (
            train_val_test_split(M, rng, (1 - val_fraction, val_fraction, 0.0))
            if M > 2
            else (np.arange(M), np.arange(M), np.arange(0))
        )
        va = va if len(va) else tr
        self.train_idx_, self.val_idx_ = tr, va
        ckpt = Path(out_dir) / "checkpoints" if out_dir else None
        res: StructureResult = fit_structure(X[tr], t, self.config, segment_starts, ckpt, self._log)
        self.structure_result_ = res
        self._readout(X.shape[2])
        if self.distill_engine != "none":
            # fresh generator: `fit` and the split `fit_symbolic` path give identical Stage 4–5 results
            self._symbolic(
                X, t, tr, va, np.random.default_rng(self.seed + 1), out_dir, segment_starts
            )
        return self

    def fit_symbolic(
        self, X: np.ndarray, t: np.ndarray, segment_starts=None, out_dir=None
    ) -> HyperDiscover:
        """Run only Stages 4–5 on an already fitted (or loaded) structure model."""
        X = np.asarray(X, dtype=np.float64)
        X = X[..., None] if X.ndim == 3 else X
        rng = np.random.default_rng(self.seed + 1)
        self._symbolic(X, t, self.train_idx_, self.val_idx_, rng, out_dir, segment_starts)
        return self

    def save_structure(self, path: str | Path) -> None:
        """Checkpoint everything Stage 4 needs (lets GPU fitting and CPU distillation be separate jobs)."""
        import torch

        res = self.structure_result_
        torch.save(
            {
                "model_state": res.model.state_dict(),
                "pair_index_all": res.pair_index_all,
                "tri_index_all": res.tri_index_all,
                "pair_scores": res.pair_scores,
                "tri_scores": res.tri_scores,
                "histories": res.histories,
                "timings": res.timings,
                "config": self.config,
                "train_idx": self.train_idx_,
                "val_idx": self.val_idx_,
                "max_order": self.max_order,
                "periodic": self.periodic,
                "seed": self.seed,
                "use_c": bool(res.model.use_c),
                "use_tris": bool(res.model.use_tris),
                "shape": res.model.N,
            },
            path,
        )

    @classmethod
    def from_structure_checkpoint(
        cls,
        path: str | Path,
        X: np.ndarray,
        t: np.ndarray,
        distill_engine="auto",
        pysr_options=None,
        segment_starts=None,
        verbose=True,
    ) -> HyperDiscover:
        import torch

        from .data.dataset import TrainData
        from .model.structure_model import StructureModel
        from .train.checkpoint import load_model_state

        ck = torch.load(path, map_location="cpu", weights_only=False)
        hd = cls(
            1,
            ck["max_order"],
            ck["config"],
            ck["periodic"],
            distill_engine,
            pysr_options,
            ck["seed"],
            verbose,
        )
        X = np.asarray(X, dtype=np.float64)
        X = X[..., None] if X.ndim == 3 else X
        cfg = hd.config
        hd.train_idx_, hd.val_idx_ = ck["train_idx"], ck["val_idx"]
        dc = cfg["data"]
        data = TrainData(
            X[hd.train_idx_],
            t,
            dc["deriv_method"],
            dc["deriv_kw"],
            dc["weak_kw"],
            segment_starts,
            dc["trim"],
        )
        mc = cfg["model"]
        model = StructureModel(
            X.shape[2],
            1,
            ck["pair_index_all"],
            ck["tri_index_all"],
            mc["hidden"],
            mc["layers"],
            ck["use_c"],
            ck["periodic"],
            data.norm,
        )
        load_model_state(model, ck["model_state"])
        model.use_tris = ck["use_tris"]
        model.eval()
        hd.structure_result_ = StructureResult(
            model,
            data,
            ck["pair_index_all"],
            ck["tri_index_all"],
            ck["pair_scores"],
            ck["tri_scores"],
            ck["histories"],
            ck["timings"],
            cfg,
        )
        hd._readout(X.shape[2])
        return hd

    def _readout(self, N: int) -> None:
        res = self.structure_result_
        thr = self.config["readout"]["threshold"]
        mode = self.config["readout"]["undirected"]
        self.pairwise_edges_ = res.edges(thr, mode)
        self.hyperedges_ = res.hyperedges(thr, mode) if self.max_order >= 3 else []
        self.structure_ = Structure(
            N,
            np.array([(a, b) for a, b, _ in self.pairwise_edges_]).reshape(-1, 2),
            np.array([(a, b, c) for a, b, c, _ in self.hyperedges_]).reshape(-1, 3),
        )
        self.node_constants_ = None
        self.equations_ = {}
        self.distilled_ = {}
        self.selection_ = None

    def _symbolic(self, X, t, tr, va, rng, out_dir, segment_starts):
        from .data.derivatives import estimate_derivatives

        res = self.structure_result_
        # configs/distill/*.yaml overlay: `distill:` (n_samples, pysr options) and `select:`;
        # explicit constructor `pysr_options` take precedence over the config
        dist_cfg = self.config.get("distill", {})
        sel_cfg = self.config.get("select", {})
        pysr_opts = {**dist_cfg.get("pysr", {}), **(self.pysr_options or {})}
        samples = sample_messages(res.model, X[tr], dist_cfg.get("n_samples", 4000), rng=rng)
        if self.max_order < 3:
            samples.pop("h", None)
        self.distilled_ = distill(
            samples,
            self.distill_engine,
            pysr_opts or None,
            self.seed,
            Path(out_dir) / "pysr" if out_dir else None,
            logger=self._log,
        )
        dcfg = self.config["data"]
        dX = estimate_derivatives(X, t, dcfg["deriv_method"], segment_starts, **dcfg["deriv_kw"])
        trim = dcfg["trim"]
        # refit on the same segment-aware mask as training (drops points next to every kick);
        # validate on the trimmed first continuous segment so rollouts never cross a kick
        mask = res.data.time_mask
        starts = list(segment_starts or [0]) + [X.shape[1]]
        sl = slice(trim, starts[1] - trim)
        use_c = bool(res.model.use_c)
        self.selection_ = select_model(
            self.distilled_,
            self.structure_,
            X[tr][:, mask],
            dX[tr][:, mask],
            X[va][:, sl],
            dX[va][:, sl],
            t[sl] - t[sl][0],
            use_c,
            self.periodic,
            k=sel_cfg.get("shortlist_k", 3),
            try_drop_one=sel_cfg.get("drop_one", True),
            rng=rng,
            logger=self._log,
        )
        best = self.selection_.best
        self.equations_ = dict(best.exprs)
        self.node_constants_ = best.fit.node_constants

    # ------------------------------------------------------------------ use
    def discovered_system(self) -> SymbolicModel:
        if not self.equations_:
            raise RuntimeError("no symbolic model: fit with distill_engine != 'none'")
        return SymbolicModel(self.structure_, self.equations_, self.node_constants_, self.periodic)

    def simulate(self, x0: np.ndarray, t: np.ndarray, substeps: int = 10) -> np.ndarray:
        """Integrate the discovered symbolic model. x0: [N, d] or [B, N, d] -> [(B,) T, N, d]."""
        x0 = np.asarray(x0, dtype=np.float64)
        single = x0.ndim == 2
        traj = rk4_integrate(
            self.discovered_system().rhs, x0[None] if single else x0, np.asarray(t), substeps
        )
        return traj[0] if single else traj

    def simulate_neural(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Integrate the learned neural model (deterministic gates), for comparison."""
        import torch

        model = self.structure_result_.model

        def rhs(x):
            with torch.no_grad():
                return (
                    model(torch.as_tensor(x, dtype=torch.float32), "deterministic").double().numpy()
                )

        x0 = np.asarray(x0, dtype=np.float64)
        single = x0.ndim == 2
        traj = rk4_integrate(rhs, x0[None] if single else x0, np.asarray(t), 4)
        return traj[0] if single else traj

    def summary(self) -> dict[str, Any]:
        res = self.structure_result_
        return {
            "pairwise_edges": [(int(a), int(b), float(s)) for a, b, s in self.pairwise_edges_],
            "hyperedges": [(int(a), int(b), int(c), float(s)) for a, b, c, s in self.hyperedges_],
            "equations": {k: str(v) for k, v in self.equations_.items()},
            "latex": {k: sp.latex(v) for k, v in self.equations_.items()},
            "node_constants": None
            if self.node_constants_ is None
            else np.asarray(self.node_constants_).ravel().tolist(),
            "pareto": {k: [c.to_dict() for c in r.pareto()] for k, r in self.distilled_.items()},
            "selection": [e.to_dict() for e in self.selection_.entries[:10]]
            if self.selection_
            else [],
            "timings": res.timings,
            "n_candidates": {
                "pairs": int(len(res.pair_index_all)),
                "triples": int(len(res.tri_index_all)),
            },
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            "# HyperDiscover report",
            "",
            "## Structure",
            f"- pairwise edges: {len(s['pairwise_edges'])} (from {s['n_candidates']['pairs']} candidates)",
            f"- hyperedges: {len(s['hyperedges'])} (from {s['n_candidates']['triples']} candidates)",
            "",
        ]
        if s["hyperedges"]:
            lines += ["| i | j | k | P(gate) |", "|---|---|---|---|"]
            lines += [f"| {a} | {b} | {c} | {p:.3f} |" for a, b, c, p in s["hyperedges"]]
            lines.append("")
        lines += [
            "## Equations",
            "",
            "dx_i/dt = f(x_i) + c_i + Σ_j A_ij g(x_i, x_j) + Σ_{j<k} B_ijk h(x_i, x_j, x_k)",
            "",
        ]
        for k, v in s["equations"].items():
            lines.append(f"- **{k}** = `{v}`")
        for k, front in s["pareto"].items():
            lines += [
                "",
                f"### Pareto front: {k}",
                "",
                "| complexity | val MSE | expression | engine |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| {c['complexity']} | {c['val_loss']:.3e} | `{c['expr']}` | {c['engine']} |"
                for c in front
            ]
        lines += ["", "## Timings (s)", ""] + [f"- {k}: {v:.1f}" for k, v in s["timings"].items()]
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=float))
        (path / "report.md").write_text(self.report())
        torch.save(self.structure_result_.model.state_dict(), path / "structure_model.pt")
