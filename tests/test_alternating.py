"""Alternating convex gating: weight mode is consistent and the per-node LASSO recovers the support."""

import numpy as np
import torch

from hyperdiscover.model.structure_model import StructureModel
from hyperdiscover.train.alternating import AlternatingConfig, lasso_weights


class _Data:
    """Minimal stand-in for TrainData: states and targets made by the model itself."""

    def __init__(self, S, Y):
        self.S, self.Y = S, Y

    def derivative_batch(self, batch, gen):
        idx = torch.randint(0, len(self.S), (batch,), generator=gen)
        return self.S[idx], self.Y[idx]


def _model(N=6):
    torch.manual_seed(0)
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    tris = np.array([(0, 1, 2), (3, 4, 5), (1, 0, 3)])
    m = StructureModel(N, 1, pairs, tris, 16, 2)
    for p in m.msg.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return m


def test_weight_mode_uses_weights():
    m = _model()
    m.weight_mode = True
    m.pair_w = torch.zeros(len(m.pair_index))
    m.tri_w = torch.zeros(len(m.tri_index))
    x = torch.randn(4, 6, 1)
    assert torch.allclose(m(x), m.msg.f(x))  # all weights zero -> self term only


def test_lasso_recovers_true_support():
    m = _model()
    m.weight_mode = True
    true_p = torch.zeros(len(m.pair_index))
    true_p[[0, 7, 13, 20]] = 1.0
    true_t = torch.tensor([1.0, 0.0, 1.0])
    m.pair_w, m.tri_w = true_p.clone(), true_t.clone()
    S = torch.randn(4000, 6, 1)
    with torch.no_grad():
        Y = m(S, "deterministic")
    m.pair_w = torch.ones_like(true_p)  # start dense, as the trainer does
    m.tri_w = torch.ones_like(true_t)
    lasso_weights(
        m,
        _Data(S, Y),
        AlternatingConfig(lasso_samples=3000, tri_penalty=2.0),
        True,
        torch.Generator().manual_seed(0),
    )
    assert torch.equal(m.pair_w > 1e-3, true_p > 0), m.pair_w
    assert torch.equal(m.tri_w > 1e-3, true_t > 0), m.tri_w
    assert torch.allclose(m.pair_w[true_p > 0], torch.ones(4), atol=0.05)
