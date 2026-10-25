"""Stage 4 input: sample (input, output) pairs of the learned f, g, h on the data distribution
of *active* edges / hyperedges (gates already multiplied in)."""

from __future__ import annotations

import numpy as np
import torch

from ..model.structure_model import StructureModel


@torch.no_grad()
def sample_messages(
    model: StructureModel,
    X: np.ndarray,
    n_samples: int = 4000,
    rng: np.random.Generator | None = None,
    threshold: float = 0.5,
) -> dict[str, dict[str, np.ndarray]]:
    """Returns {"f": {"X": [S, d], "y": [S, d]}, "g": {"X": [S, 2d], ...}, "h": {"X": [S, 3d], ...}}.

    g and h outputs are multiplied by the deterministic gate value of the sampled candidate,
    so a partially open gate is reflected in the distilled amplitude. For "f" the per-node
    constant c_i is *not* included (it is reported separately).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    model.eval()
    dev = next(model.parameters()).device
    M, T, N, d = X.shape
    states = X.reshape(-1, N, d)
    out: dict[str, dict[str, np.ndarray]] = {}
    s_idx = rng.integers(0, len(states), n_samples)
    n_idx = rng.integers(0, N, n_samples)
    xi = torch.as_tensor(states[s_idx, n_idx], dtype=torch.float32, device=dev)
    out["f"] = {"X": xi.cpu().numpy(), "y": model.msg.f(xi).cpu().numpy()}

    pp, pt = model.gate_probs()
    zp = model.pair_gate("deterministic").cpu().numpy()
    act = np.where(pp > threshold)[0]
    if len(act):
        e = act[rng.integers(0, len(act), n_samples)]
        pi = model.pair_index.cpu().numpy()[e]
        xs = states[s_idx]
        a = torch.as_tensor(xs[np.arange(n_samples), pi[:, 0]], dtype=torch.float32, device=dev)
        b = torch.as_tensor(xs[np.arange(n_samples), pi[:, 1]], dtype=torch.float32, device=dev)
        y = model.msg.g(a, b).cpu().numpy() * zp[e][:, None]
        out["g"] = {"X": np.concatenate([a.cpu().numpy(), b.cpu().numpy()], -1), "y": y}
    if model.tri_gate.n and model.use_tris:
        zt = model.tri_gate("deterministic").cpu().numpy()
        act = np.where(pt > threshold)[0]
        if len(act):
            e = act[rng.integers(0, len(act), n_samples)]
            ti = model.tri_index.cpu().numpy()[e]
            xs = states[s_idx]
            r = np.arange(n_samples)
            a, b, c = (
                torch.as_tensor(xs[r, ti[:, q]], dtype=torch.float32, device=dev) for q in range(3)
            )
            y = model.msg.h(a, b, c).cpu().numpy() * zt[e][:, None]
            out["h"] = {"X": np.concatenate([v.cpu().numpy() for v in (a, b, c)], -1), "y": y}
    return out
