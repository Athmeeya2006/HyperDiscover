"""CLAUDE.md §12 test 4: h symmetric in (j, k); node relabelling permutes outputs consistently."""

import numpy as np
import torch

from hyperdiscover.model.messages import Messages
from hyperdiscover.model.structure_model import StructureModel


def test_h_symmetric():
    torch.manual_seed(0)
    m = Messages(1, 32, 2)
    for p in m.h_net.parameters():  # zero-init last layer would make this trivial
        torch.nn.init.normal_(p, std=0.5)
    a, b, c = (torch.randn(100, 1) for _ in range(3))
    assert torch.allclose(m.h(a, b, c), m.h(a, c, b), atol=1e-6)
    assert not torch.allclose(m.h(a, b, c), m.h(b, a, c), atol=1e-3)


def test_periodic_embedding():
    m = Messages(1, 16, 2, periodic=True)
    for p in m.g_net.parameters():
        torch.nn.init.normal_(p, std=0.5)
    a, b = torch.randn(50, 1), torch.randn(50, 1)
    assert torch.allclose(m.g(a, b), m.g(a + 2 * np.pi, b - 4 * np.pi), atol=1e-4)


def test_node_relabelling_equivariance():
    torch.manual_seed(0)
    N = 7
    rng = np.random.default_rng(0)
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j and rng.random() < 0.4])
    tris = np.array([(0, 1, 2), (1, 0, 2), (2, 0, 1), (3, 4, 6)])
    model = StructureModel(N, 1, pairs, tris, 16, 2)
    for p in model.msg.parameters():
        torch.nn.init.normal_(p, std=0.3)
    perm = rng.permutation(N)
    inv = np.argsort(perm)  # node v is relabelled to inv[v]
    pairs_p = inv[pairs]
    tris_p = inv[tris]
    tris_p[:, 1:] = np.sort(tris_p[:, 1:], axis=1)
    model_p = StructureModel(N, 1, pairs_p, tris_p, 16, 2)
    model_p.msg.load_state_dict(model.msg.state_dict())
    with torch.no_grad():
        model_p.pair_gate.log_alpha.copy_(model.pair_gate.log_alpha)
        model_p.tri_gate.log_alpha.copy_(model.tri_gate.log_alpha)
    x = torch.randn(10, N, 1)
    y = model(x, "deterministic")
    x_p = x[:, perm]  # new node q holds old node perm[q]
    y_p = model_p(x_p, "deterministic")
    assert torch.allclose(y_p, y[:, perm], atol=1e-5)
