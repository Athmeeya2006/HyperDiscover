"""GNN-encoder (amortized) gates: symmetry, trainability per stage, and an end-to-end smoke fit."""

import numpy as np
import torch

from hyperdiscover import benchmarks
from hyperdiscover.model.gnn_gates import node_features
from hyperdiscover.model.structure_model import StructureModel
from hyperdiscover.pipeline import fit_structure
from hyperdiscover.train.checkpoint import load_model_state


def _model(N=6):
    torch.manual_seed(0)
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    tris = np.array([(0, 1, 2), (0, 2, 1), (3, 4, 5)])
    feats = torch.randn(N, 10)
    return StructureModel(N, 1, pairs, tris, 16, 2, node_feats=feats, gnn_hidden=8)


def test_triple_logits_symmetric_in_jk():
    m = _model()
    with torch.no_grad():
        for p in m.tri_gate.head.parameters():
            torch.nn.init.normal_(p)  # the head starts at zero; make it non-trivial
        la = m.tri_gate.log_alpha
    assert torch.allclose(la[0], la[1])  # (0, 1, 2) and (0, 2, 1)
    assert not torch.allclose(la[0], la[2])


def test_encoder_trainable_when_either_gate_trains():
    m = _model()
    m.set_trainable(f=False, g=False, h=True, pair_gates=False, tri_gates=True, c=False)
    assert all(p.requires_grad for p in m.gate_encoder.parameters())
    assert not any(p.requires_grad for p in m.pair_gate.head.parameters())
    m.set_trainable(pair_gates=False, tri_gates=False)
    assert not any(p.requires_grad for p in m.gate_encoder.parameters())
    groups = m.param_groups(1e-3, 1e-2, 0.0)
    n_gate = sum(len(g["params"]) for g in groups if g["lr"] == 1e-2)
    assert n_gate == 0


def test_node_features_shape():
    S, Y = torch.randn(500, 7, 1), torch.randn(500, 7, 1)
    assert node_features(S, Y, 32, periodic=False).shape == (7, 64)
    assert node_features(S, Y, 32, periodic=True).shape == (7, 96)


def test_smoke_fit_and_checkpoint_roundtrip(tmp_path):
    b = benchmarks.load("S4", N=8, hyper_kind="open", M=6, T=40, seed=0)
    cfg = {
        "gating": "gnn_encoder",
        "model": {"hidden": 16, "layers": 2, "gnn": {"snapshots": 16, "hidden": 8}},
        "train": {
            "pairwise": {"steps": 20},
            "triadic": {"steps": 20},
            "joint": {"steps": 10},
        },
    }
    res = fit_structure(b.data.X, b.data.t, cfg, logger=lambda *_: None)
    assert res.model.gnn
    pp, pt = res.model.gate_probs()
    assert np.all((pp >= 0) & (pp <= 1)) and np.all((pt >= 0) & (pt <= 1))
    sd = res.model.state_dict()
    torch.save(sd, tmp_path / "m.pt")
    m2 = StructureModel(
        res.model.N,
        1,
        res.pair_index_all,
        res.tri_index_all,
        16,
        2,
        node_feats=res.model.gate_encoder.feats.clone(),
        gnn_hidden=8,
        norm=res.data.norm,
    )
    load_model_state(m2, torch.load(tmp_path / "m.pt"))
    x = torch.as_tensor(b.data.X[:2, 0], dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(m2(x, "deterministic"), res.model(x, "deterministic"), atol=1e-6)
