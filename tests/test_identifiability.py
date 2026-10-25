"""The pure-interaction fraction separates additive (pairwise-mimicable) from joint triadic terms."""

import numpy as np

from hyperdiscover.eval.identifiability import pure_interaction_fraction


def test_additive_h_is_fully_mimicable(rng):
    Z = rng.uniform(0, 1, (6000, 3))
    hv = np.sin(Z[:, 1]) * Z[:, 0] + np.sin(Z[:, 2]) * Z[:, 0]  # = u(x_i,x_j) + u(x_i,x_k)
    assert pure_interaction_fraction(Z, hv)["pure_fraction"] < 0.05


def test_product_h_is_largely_joint(rng):
    Z = rng.uniform(-1, 1, (6000, 3))  # independent, zero-mean inputs: x_j x_k has no main effects
    hv = Z[:, 1] * Z[:, 2]
    assert pure_interaction_fraction(Z, hv)["pure_fraction"] > 0.8
