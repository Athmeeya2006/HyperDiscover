"""CLAUDE.md §12 test 3: hard-concrete gates hit exact 0/1 at evaluation; L0 expectation is right."""

import math

import torch

from hyperdiscover.model.gates import HardConcreteGate


def test_exact_zero_and_one():
    g = HardConcreteGate(4)
    with torch.no_grad():
        g.log_alpha.copy_(torch.tensor([-10.0, -3.0, 3.0, 10.0]))
    z = g("deterministic")
    assert z[0].item() == 0.0 and z[1].item() == 0.0
    assert z[2].item() == 1.0 and z[3].item() == 1.0
    assert torch.equal(g("hard"), torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_l0_expectation_matches_monte_carlo():
    torch.manual_seed(0)
    g = HardConcreteGate(5)
    with torch.no_grad():
        g.log_alpha.copy_(torch.tensor([-2.0, -0.5, 0.0, 0.7, 2.0]))
    samples = torch.stack([g("sample") for _ in range(40000)])
    mc = (samples > 0).float().mean(0)
    assert torch.allclose(mc, g.prob_nonzero().detach(), atol=0.01), (mc, g.prob_nonzero())


def test_closed_form():
    g = HardConcreteGate(1, beta=0.5, gamma=-0.1, zeta=1.1)
    with torch.no_grad():
        g.log_alpha.fill_(0.3)
    want = 1 / (1 + math.exp(-(0.3 - 0.5 * math.log(0.1 / 1.1))))
    assert abs(g.prob_nonzero().item() - want) < 1e-6


def test_init_prob():
    g = HardConcreteGate(1000, init_prob=0.8, init_std=0.0)
    assert torch.allclose(g.prob_nonzero(), torch.full((1000,), 0.8), atol=1e-5)


def test_samples_in_unit_interval_with_mass_at_ends():
    torch.manual_seed(1)
    g = HardConcreteGate(1)
    s = torch.stack([g("sample") for _ in range(5000)])
    assert s.min() >= 0 and s.max() <= 1
    assert (s == 0).any() and (s == 1).any()


def test_warmup_keeps_gates_fixed_and_fits():
    import numpy as np

    from hyperdiscover import benchmarks
    from hyperdiscover.data.dataset import TrainData
    from hyperdiscover.model.structure_model import StructureModel
    from hyperdiscover.train.loop import StageConfig, run_stage

    b = benchmarks.load("S4", N=6, n_hyper=1, mean_degree=2.0, M=6, T=60, seed=0)
    data = TrainData(b.data.X, b.data.t, trim=3, weak_kw={"window": 11, "stride": 4})
    pairs = np.array([(i, j) for i in range(6) for j in range(6) if i != j])
    m = StructureModel(6, 1, pairs, b.structure.tri_index, 16, 2, norm=data.norm)
    before = m.pair_gate.log_alpha.detach().clone()
    cfg = StageConfig(name="w", steps=20, warmup_steps=20, log_every=1000, lam_pair=1.0)
    run_stage(m, data, cfg, logger=lambda *_: None)
    assert torch.equal(m.pair_gate.log_alpha.detach(), before)  # no gate updates during warm-up
