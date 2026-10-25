"""Model-free hyperedge detection from joint two-node kicks (DECISIONS.md D-030).

For the model class dx_i/dt = f(x_i) + c_i + Σ_j A_ij g(x_i, x_j) + Σ_{j<k} B_ijk h(x_i, x_j, x_k),
the pairwise part is additively separable in the neighbours, so for j ≠ k (both ≠ i)

    ∂²ẋ_i / ∂x_j ∂x_k = Σ_{hyperedges ∋ i,j,k} ∂²h(x_i, x_j, x_k) / ∂x_j ∂x_k,

whatever g is. A joint kick measures this directly: from an operating state x*, kick j by δ, k by
δ, and both, integrate a short time τ, and form the mixed difference of node i's state

    D_ijk = x_i^{jk}(τ) − x_i^{j}(τ) − x_i^{k}(τ) + x_i^{0}(τ)  ≈  τ δ² ∂²ẋ_i/∂x_j∂x_k + O(τ²).

The O(τ²) term carries indirect paths (j → l → i, k → l → i); it is cancelled by Richardson
extrapolation, 2·D(τ) − D(2τ)/2, from readouts at τ and 2τ of the same intervention. The same idea, applied to (i, j) with single kicks, gives the
pairwise response D_ij = x_i^{j}(τ) − x_i^{0}(τ) ≈ τ δ ∂ẋ_i/∂x_j (edge detection).

Interventions per operating state: N single kicks + N(N−1)/2 joint kicks (one joint kick of (j, k)
is read out at every i simultaneously).

References (references/references.bib): richardson1911approximate, richardson1927deferred, dunn1961multiple, timme2007revealing.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..data.simulate import rk4_integrate


def kick_responses(
    system,
    x_star: np.ndarray,
    delta: float,
    tau: float,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    substeps: int = 10,
    n_repeats: int = 1,
    richardson: bool = True,
) -> dict[str, np.ndarray]:
    """Responses to no kick, single kicks and joint kicks from each operating state.

    Args:
        x_star: operating states [S, N, d] (d = 1 is used for the scores).
        delta: kick size in units of the IC-box width.
        tau: response time.
        noise: measurement noise std of one readout, in units of the IC-box width.
        n_repeats: independent noisy readouts averaged per intervention.
        richardson: read at τ and 2τ and use 2·x(τ) − x(2τ)/2, which cancels the O(τ²) term of every
            difference (the indirect paths j → l → i) at no extra intervention cost.
    Returns:
        {"base": [S, N], "base_rep": [S, N], "single": [S, N(source), N(target)], "pairs": [P, 2],
         "joint": [S, P, N(target)]}: effective readouts of node states (first state component);
        ``base_rep`` is an independent effective readout of the unkicked states (noise calibration).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    S, N, _ = x_star.shape
    width = system.ic_box[1] - system.ic_box[0]
    d = delta * width
    pairs = np.array(list(itertools.combinations(range(N), 2)), dtype=np.int64)
    # batch every intervention of every operating state into one integration
    x0 = [x_star[:, None]]  # base
    single = np.repeat(x_star[:, None], N, 1).copy()
    single[:, np.arange(N), np.arange(N)] += d
    x0.append(single)
    joint = np.repeat(x_star[:, None], len(pairs), 1).copy()
    joint[:, np.arange(len(pairs)), pairs[:, 0]] += d
    joint[:, np.arange(len(pairs)), pairs[:, 1]] += d
    x0.append(joint)
    X0 = np.concatenate(x0, 1)  # [S, 1 + N + P, N, d]
    post = getattr(system, "postprocess", None)
    xt = rk4_integrate(system.rhs, X0, np.array([0.0, tau, 2 * tau]), substeps, post)[..., 0]
    xt = np.moveaxis(xt, 2, 0)  # [T=3, S, 1 + N + P, N]

    def read(a):  # a: [T=3, ...] noiseless states -> one effective noisy readout
        if noise > 0:
            eps = rng.standard_normal((n_repeats,) + a.shape).mean(0)
            a = a + noise * width * eps
        return 2 * a[1] - 0.5 * a[2] if richardson else a[1]

    base_rep = read(xt[:, :, 0])  # second, independent readout of the unkicked states
    xt = read(xt)
    return {
        "base": xt[:, 0],
        "base_rep": base_rep,
        "single": xt[:, 1 : 1 + N],
        "pairs": pairs,
        "joint": xt[:, 1 + N :],
    }


def mixed_difference_scores(
    resp: dict[str, np.ndarray], delta: float, tau: float, width: float = 1.0
):
    """Scores from kick responses.

    Returns:
        pair_score [N(target), N(source)]: RMS over operating states of D_ij / (τ δ) (diag = 0).
        tri_score [N(target), P]: RMS over operating states of D_ijk / (τ δ²) for every source pair
            (j, k) in ``resp["pairs"]``; entries with i ∈ {j, k} are 0.
    """
    d = delta * width
    base, single, joint, pairs = resp["base"], resp["single"], resp["joint"], resp["pairs"]
    S, N = base.shape
    Dij = (single - base[:, None, :]) / (tau * d)  # [S, source, target]
    pair_score = np.sqrt((Dij**2).mean(0)).T
    np.fill_diagonal(pair_score, 0.0)
    j, k = pairs[:, 0], pairs[:, 1]
    Dijk = (joint - single[:, j] - single[:, k] + base[:, None, :]) / (
        tau * d * d
    )  # [S, P, target]
    tri_score = np.sqrt((Dijk**2).mean(0)).T  # [target, P]
    tri_score[j, np.arange(len(pairs))] = 0.0
    tri_score[k, np.arange(len(pairs))] = 0.0
    return pair_score, tri_score


def calibrated_pvalues(
    resp: dict[str, np.ndarray], sigma_read: float
) -> tuple[np.ndarray, np.ndarray]:
    """Truth-free significance of the pairwise and joint responses (χ² test against readout noise).

    Under "no direct effect" each D_ij is pure readout noise with variance 2σ² and each D_ijk with
    variance 4σ² (σ = std of one averaged readout, estimated from replicate unkicked readouts, see
    ``estimate_readout_noise``). Summing (D/σ_D)² over the S operating states gives χ²_S.
    Returns p-values [N, N] (pairs, target × source) and [N, P] (joint, target × source pair).
    """
    from scipy.stats import chi2

    base, single, joint, pairs = resp["base"], resp["single"], resp["joint"], resp["pairs"]
    S = base.shape[0]
    Dij = single - base[:, None, :]
    j, k = pairs[:, 0], pairs[:, 1]
    Dijk = joint - single[:, j] - single[:, k] + base[:, None, :]
    q_pair = (Dij**2).sum(0).T / (2 * sigma_read**2)
    q_tri = (Dijk**2).sum(0).T / (4 * sigma_read**2)
    p_pair = chi2.sf(q_pair, S)
    p_tri = chi2.sf(q_tri, S)
    np.fill_diagonal(p_pair, 1.0)
    p_tri[j, np.arange(len(pairs))] = 1.0
    p_tri[k, np.arange(len(pairs))] = 1.0
    return p_pair, p_tri


def estimate_readout_noise(xt_a: np.ndarray, xt_b: np.ndarray) -> float:
    """σ of one readout from two independent readouts of the same unkicked states: std(a − b)/√2."""
    return float(np.std(xt_a - xt_b) / np.sqrt(2))


def hyperedge_scores(tri_score: np.ndarray, pairs: np.ndarray) -> dict[tuple[int, int, int], float]:
    """Unordered hyperedge {a, b, c} score = max over its three focal readouts."""
    N = tri_score.shape[0]
    col = {tuple(p): q for q, p in enumerate(pairs.tolist())}
    out = {}
    for a, b, c in itertools.combinations(range(N), 3):
        out[(a, b, c)] = max(
            tri_score[a, col[(b, c)]], tri_score[b, col[(a, c)]], tri_score[c, col[(a, b)]]
        )
    return out


def kick_candidates(
    resp: dict[str, np.ndarray], alpha: float = 0.01, sigma_read: float | None = None
) -> tuple:
    """Truth-free structure from kick responses (χ² against readout noise, Bonferroni at ``alpha``).

    Returns (Structure of candidates, info). Candidate edges = undirected pairs with a significant
    direct single-kick influence in either direction (this includes influence that acts only through
    a hyperedge, so they are CANDIDATES for the observational fit to select from); hyperedges = node
    triples with a significant mixed (joint-kick) response at any focal node. Noiseless responses
    (sigma_read = 0) fall back to a 1e-6 relative floor.
    """
    from ..systems.base import Structure

    N = resp["base"].shape[1]
    sigma = (
        estimate_readout_noise(resp["base"], resp["base_rep"]) if sigma_read is None else sigma_read
    )
    n_tri_tests = N * (N - 1) * (N - 2) // 2
    if sigma > 0:
        pp, pt = calibrated_pvalues(resp, sigma)
        infl = pp < alpha / (N * (N - 1))
        hs = hyperedge_scores(-pt, resp["pairs"])
        hyper = [k for k, v in hs.items() if -v < alpha / n_tri_tests]
    else:
        ps, ts = mixed_difference_scores(resp, 1.0, 1.0)
        infl = ps > 1e-6 * ps.max()
        hs = hyperedge_scores(ts, resp["pairs"])
        top = max(hs.values()) if hs else 0.0
        hyper = [k for k, v in hs.items() if v > 1e-6 * top]
    edges = sorted({tuple(sorted((i, j))) for i, j in zip(*np.where(infl)) if i != j})
    st = Structure(
        N,
        np.array(edges, dtype=np.int64).reshape(-1, 2),
        np.array(hyper, dtype=np.int64).reshape(-1, 3),
    )
    return st, {"sigma_read": sigma, "n_candidate_edges": len(edges), "n_hyperedges": len(hyper)}
