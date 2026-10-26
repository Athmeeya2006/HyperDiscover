"""Delayed-coupling simulator (data/delays.py): zero lags reproduce the ordinary integrator."""

import numpy as np

from hyperdiscover import benchmarks
from hyperdiscover.data.delays import simulate_delayed
from hyperdiscover.data.simulate import rk4_integrate


def test_zero_lag_matches_rk4():
    b = benchmarks.load("S4", N=8, hyper_kind="open", M=2, T=2, seed=0)
    s = b.system
    rng = np.random.default_rng(0)
    x0 = s.sample_ic(3, rng)
    hist = x0[:, None]
    xd = simulate_delayed(s, np.zeros(len(s._pairs)), np.zeros(len(s._tris)), hist, 20)
    t = np.arange(21) * s.default_dt
    xr = rk4_integrate(s.rhs, x0, t, 10, getattr(s, "postprocess", None))
    xr = np.moveaxis(xr, 0, 1) if xr.shape[0] == 21 else xr
    assert np.allclose(xd, xr, atol=1e-8)
