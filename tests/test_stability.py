"""Stability selection / per-triple BIC test (train/stability.py) on a linear toy problem."""

import numpy as np

from hyperdiscover.train.stability import bic_prune_triples


def test_bic_prune_drops_spurious_triple_keeps_true_one():
    rng = np.random.default_rng(0)
    n = 2000
    G = rng.normal(size=(n, 2))  # two candidate pairs at node 0, both true
    H = rng.normal(size=(n, 3))  # three candidate triples, only the first is true
    H[:, 2] = np.abs(H[:, 2])
    R = (G @ np.array([1.0, 1.0]) + H[:, 0] + 0.05 * rng.normal(size=n))[:, None]
    keep = bic_prune_triples(
        R,
        G,
        H,
        pair_t=np.zeros(2, int),
        tri_t=np.zeros(3, int),
        keep_p=np.ones(2, bool),
        keep_t=np.ones(3, bool),
        N=1,
    )
    assert keep.tolist() == [True, False, False]


def test_pair_lock_selection_runs_per_target():
    """pair_lock: frozen pairs are subtracted per target node before triples are selected."""
    import torch

    from hyperdiscover.train.alternating import AlternatingConfig, select_on_batch

    class _M:  # minimal stand-in for StructureModel
        N = 3
        pair_index = torch.tensor([[0, 1], [1, 2], [2, 0]])
        tri_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
        pair_lock_mask = np.array([True, True, False])

    rng = np.random.default_rng(1)
    n = 1500
    G = rng.normal(size=(n, 3))
    H = rng.normal(size=(n, 2))
    R = np.stack([G[:, 0] + H[:, 0], G[:, 1], 0.01 * rng.normal(size=n)], 1)
    wp, wt = select_on_batch(_M(), R, G, H, AlternatingConfig(pair_lock=True), True)
    assert wp[2] == 0 and wp[0] > 0.9 and wp[1] > 0.9
    assert wt[0] > 0.9 and wt[1] == 0
