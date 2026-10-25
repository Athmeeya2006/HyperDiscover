"""Oracle check of the alternating-gating LASSO step: TRUE f, g, h as messages, S4 sanity instance.

If BIC-LASSO cannot recover the exact support even with perfect messages, the structure step itself
(not MLP training) is the problem.
"""

import itertools
import sys

import numpy as np
from sklearn.linear_model import LassoLarsIC

from hyperdiscover import benchmarks

data_kw = dict(
    N=10, noise=0.0, hyper_kind="open", n_hyper=2, graph="er", mean_degree=3.0, M=50, T=200, seed=0
)
b = benchmarks.load("S4", **data_kw)
sysm, st = b.system, b.system.structure
X = b.data.X_clean.reshape(-1, sysm.N, 1)
Y = b.data.dX_clean.reshape(-1, sysm.N, 1)
rng = np.random.default_rng(0)
sel = rng.choice(len(X), 3000, replace=False)
X, Y = X[sel], Y[sel]
N = sysm.N
true_pairs = {tuple(p) for p in st.pair_index}
true_tris = {(t[0], *sorted(t[1:])) for t in st.tri_index}
print("true edges", len(true_pairs), "true focal triples", len(true_tris))

msg_err = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # relative perturbation of messages
R = (Y - sysm.f(X) - sysm.c)[..., 0]


def run(tri_penalty, sigma2_floor):
    fp = fn = tfp = tfn = 0
    for i in range(N):
        js = [j for j in range(N) if j != i]
        ks = [(j, k) for j, k in itertools.combinations(js, 2)]
        G = np.stack([sysm.g(X[:, i], X[:, j])[:, 0] for j in js], 1)
        H = np.stack([sysm.h(X[:, i], X[:, j], X[:, k])[:, 0] for j, k in ks], 1)
        if msg_err:
            G = G * (1 + msg_err * rng.standard_normal(G.shape[1]))
            H = H * (1 + msg_err * rng.standard_normal(H.shape[1]))
        D = np.concatenate([G, H / tri_penalty], 1)
        y = R[:, i]
        nv = None
        if sigma2_floor is not None:
            # floor relative to the target variance, so a near-perfect OLS fit cannot make BIC permissive
            from numpy.linalg import lstsq

            res = y - D @ lstsq(D, y, rcond=None)[0]
            nv = max(res.var() * len(y) / max(len(y) - D.shape[1], 1), sigma2_floor * y.var())
        m = LassoLarsIC(criterion="bic", positive=True, fit_intercept=False, noise_variance=nv).fit(
            D, y
        )
        w = m.coef_
        for q, j in enumerate(js):
            on, tr = w[q] > 1e-8, (i, j) in true_pairs
            fp += on and not tr
            fn += tr and not on
        for q, (j, k) in enumerate(ks):
            on, tr = w[len(js) + q] > 1e-8, (i, j, k) in true_tris
            tfp += on and not tr
            tfn += tr and not on
    print(
        f"tri_pen {tri_penalty} floor {sigma2_floor}: edges FP {fp} FN {fn} | triples FP {tfp} FN {tfn}"
    )


for tp in (1.0, 2.0, 4.0):
    for fl in (None, 1e-4, 1e-3, 1e-2):
        run(tp, fl)
