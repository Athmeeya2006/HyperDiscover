"""Stand-alone benchmark suite: ``hyperdiscover.benchmarks.load("S4", N=50, noise=0.05)``.

A benchmark instance is fully determined by its arguments and seed (structure, parameters,
initial conditions and noise all come from one ``numpy.random.Generator``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .data.observe import add_noise
from .data.simulate import TrajectoryData, simulate_system
from .graphs.hypergraphs import make_structure
from .systems import SYSTEMS, make_system
from .systems.base import HigherOrderSystem, Structure

BENCH_VERSION = "hd-bench-v0.1"


@dataclass
class Benchmark:
    system: HigherOrderSystem
    structure: Structure
    data: TrajectoryData
    spec: dict[str, Any]

    @property
    def truth(self) -> dict[str, Any]:
        return self.system.ground_truth()


def load(
    system_id: str,
    N: int = 50,
    noise: float = 0.0,
    hyper_kind: str = "open",
    graph: str = "er",
    mean_degree: float = 4.0,
    n_hyper: int | None = None,
    hyper_density: float = 0.3,
    pairwise_only: bool = False,
    M: int = 50,
    T: int = 200,
    dt: float | None = None,
    seed: int = 0,
    params: dict | None = None,
    kick_std: float = 0.0,
    kick_every: int = 0,
    disjoint_hyperedges: bool = False,
) -> Benchmark:
    """Generate (deterministically) one benchmark instance."""
    if system_id not in SYSTEMS:
        raise KeyError(system_id)
    rng = np.random.default_rng([seed, int(system_id[1:]), N])
    structure = make_structure(
        N,
        rng,
        graph,
        mean_degree,
        hyper_kind,
        n_hyper,
        hyper_density,
        pairwise_only=pairwise_only,
        disjoint=disjoint_hyperedges,
    )
    kw = {"rng": rng} if system_id == "S1" else {}
    system = make_system(system_id, structure, params, **kw)
    data = simulate_system(system, M, T, dt, rng, kick_std=kick_std, kick_every=kick_every)
    data = add_noise(data, noise, rng)
    spec = {
        "version": BENCH_VERSION,
        "system_id": system_id,
        "N": N,
        "noise": noise,
        "hyper_kind": hyper_kind,
        "graph": graph,
        "mean_degree": mean_degree,
        "n_hyper": int(len(structure.hyperedges)),
        "n_edges": int(len(structure.edges)),
        "pairwise_only": pairwise_only,
        "M": M,
        "T": T,
        "dt": data.meta["dt"],
        "seed": seed,
        "params": system.params,
        "kick_std": kick_std,
        "kick_every": kick_every,
    }
    return Benchmark(system, structure, data, spec)


def save(bench: Benchmark, path: str | Path) -> None:
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # atomic: a killed job never leaves a partial .h5
    with h5py.File(tmp, "w") as f:
        f.create_dataset("X", data=bench.data.X.astype(np.float32), compression="gzip")
        f.create_dataset("X_clean", data=bench.data.X_clean.astype(np.float32), compression="gzip")
        f.create_dataset(
            "dX_clean", data=bench.data.dX_clean.astype(np.float32), compression="gzip"
        )
        f.create_dataset("t", data=bench.data.t)
        f.attrs["spec"] = json.dumps(bench.spec, default=float)
        f.attrs["truth"] = json.dumps(bench.truth, default=float)
        f.attrs["meta"] = json.dumps(bench.data.meta, default=float)
    os.replace(tmp, path)


def load_file(path: str | Path) -> tuple[TrajectoryData, dict, dict]:
    """Returns (data, spec, truth) from a saved benchmark file."""
    with h5py.File(path, "r") as f:
        data = TrajectoryData(
            f["X"][()].astype(np.float64),
            f["t"][()],
            f["X_clean"][()].astype(np.float64),
            f["dX_clean"][()].astype(np.float64),
            json.loads(f.attrs["meta"]),
        )
        return data, json.loads(f.attrs["spec"]), json.loads(f.attrs["truth"])


def rebuild_system(truth: dict) -> HigherOrderSystem:
    """Reconstruct the ground-truth system object from a saved ``truth`` dict."""
    st = Structure.from_dict(truth["structure"])
    return make_system(
        truth["system_id"], st, truth["params"], node_constants=np.array(truth["node_constants"])
    )
