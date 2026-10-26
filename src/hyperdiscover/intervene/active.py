"""Budgeted joint-kick design: screened kick pairs + adaptive readout repeats (CLAUDE.md §18.12 G1, §18.4 P1).

The exhaustive design (joint_kick.kick_responses) kicks all N(N−1)/2 node pairs. Here:

1. **Single kicks** of every node (N interventions, ``r0`` readouts each) give, per target i, the set
   C_i of the ``max_per_target`` strongest direct influences on i (χ² statistic vs readout noise).
   Hyperedge partners are in C_i whenever ∂h/∂x_j ≠ 0 at the operating states, so a hyperedge
   {i, j, k} needs j, k ∈ C_i (or the analogous condition at another member).
2. **Screened joint kicks**: only pairs (j, k) with j, k ∈ C_i for some i. For sparse networks
   |C_i| is bounded, so the number of joint kicks grows ~linearly in N instead of quadratically.
   |C_i| = ``max_per_target`` keeps the joint-kick count ≤ N·C(m, 2).
3. **Adaptive repeats**: every joint kick starts with ``r0`` readouts; in each round, kicks whose
   best focal p-value is undecided (between the Bonferroni level and ``p_hi``/(N − 2)) get their readouts
   doubled, until ``max_rounds`` or the readout budget is used. Decided kicks get no more readouts.

Readout noise is i.i.d. per readout, so the mean of r readouts is the noiseless state plus noise of
std σ/√r; responses are integrated once (deterministic dynamics) and readouts are drawn on demand.
Costs are counted as interventions (distinct kicked states) and readouts (interventions × repeats).

References (references/references.bib): timme2007revealing, richardson1911approximate (Richardson
readout as in joint_kick.py); the screened design is ours (§17.1 "Open").
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ..data.simulate import rk4_integrate


@dataclass
class ActiveKickConfig:
    delta: float = 0.3
    tau: float = 1.0  # absolute time (caller converts from units of dt)
    r0: int = 1
    max_rounds: int = 5
    p_hi: float = 0.05
    alpha: float = 0.01
    max_per_target: int = 8  # candidate influencers per target: significant ones, else the strongest
    richardson: bool = True
    substeps: int = 10
    readout_budget: int | None = None  # total readouts for joint kicks (None = unlimited)
    screen: bool = True  # False: joint-kick every pair (exhaustive design, same readout logic)
    adaptive: bool = True  # False: no extra readouts (r0 per kick)
    random_pairs: int | None = None  # baseline: this many uniformly random pairs instead of screening
    bonferroni: str = "performed"  # "performed" (P·(N−2) tests) | "exhaustive" (N(N−1)(N−2)/2)


class _Kicker:
    """Noiseless responses at τ, 2τ for a set of initial states; noisy averaged readouts on demand."""

    def __init__(self, system, x_star, cfg: ActiveKickConfig, noise: float, rng):
        self.system, self.x_star, self.cfg, self.rng = system, x_star, cfg, rng
        self.width = system.ic_box[1] - system.ic_box[0]
        self.noise = noise
        self.d = cfg.delta * self.width

    def integrate(self, kicked: list[tuple[int, ...]]) -> np.ndarray:
        """kicked: list of node tuples (() = no kick). Returns noiseless [3, S, K, N]."""
        S, N, _ = self.x_star.shape
        X0 = np.repeat(self.x_star[:, None], len(kicked), 1).copy()
        for q, nodes in enumerate(kicked):
            for v in nodes:
                X0[:, q, v] += self.d
        post = getattr(self.system, "postprocess", None)
        xt = rk4_integrate(
            self.system.rhs, X0, np.array([0.0, self.cfg.tau, 2 * self.cfg.tau]),
            self.cfg.substeps, post,
        )[..., 0]
        return np.moveaxis(xt, 2, 0)

    def readout_sum(self, clean: np.ndarray, r: int) -> np.ndarray:
        """Sum of r noisy effective readouts of clean [3, ...] (Richardson combination applied)."""
        eff = 2 * clean[1] - 0.5 * clean[2] if self.cfg.richardson else clean[1]
        if self.noise <= 0 or r == 0:
            return r * eff
        # sum of r i.i.d. readouts at τ and 2τ: noise std √r σ at each time
        e1 = self.rng.standard_normal(eff.shape) * np.sqrt(r)
        e2 = self.rng.standard_normal(eff.shape) * np.sqrt(r)
        sig = self.noise * self.width
        if self.cfg.richardson:
            return r * eff + sig * (2 * e1 - 0.5 * e2)
        return r * eff + sig * e1


def _readout_var(sigma: float, richardson: bool) -> float:
    """Variance of one effective readout given per-time readout std σ."""
    return sigma**2 * (4 + 0.25 if richardson else 1.0)


def active_joint_kicks(
    system,
    x_star: np.ndarray,
    cfg: ActiveKickConfig,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run the budgeted design from operating states x_star [S, N, d].

    Returns dict with ``hyperedges`` (list of sorted triples, truth-free detections), ``hyper_scores``
    {triple: −log10 p (∞ when noiseless and non-zero)}, ``influence`` [N, N] bool (target × source),
    ``pairs_kicked`` [P, 2], and costs ``n_interventions``, ``n_readouts``, ``n_pairs_exhaustive``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    S, N, _ = x_star.shape
    k = _Kicker(system, x_star, cfg, noise, rng)
    sigma = noise * k.width
    v1 = _readout_var(sigma, cfg.richardson)

    # --- 1. base + single kicks, r0 readouts each
    clean_bs = k.integrate([()] + [(v,) for v in range(N)])  # [3, S, 1 + N, N]
    r_bs = cfg.r0
    sums = k.readout_sum(clean_bs, r_bs)
    base, single = sums[:, 0] / r_bs, sums[:, 1:] / r_bs  # [S, N], [S, src, tgt]
    Dij = single - base[:, None, :]
    if sigma > 0:
        q = (Dij**2).sum(0).T / (2 * v1 / r_bs)  # target × source
        p_pair = chi2.sf(q, S)
        np.fill_diagonal(p_pair, 1.0)
        infl = p_pair < cfg.alpha / (N * (N - 1))
        strength = -np.log10(np.maximum(p_pair, 1e-300))
    else:
        mag = np.sqrt((Dij**2).mean(0)).T
        np.fill_diagonal(mag, 0.0)
        infl = mag > 1e-6 * max(mag.max(), 1e-300)
        strength = mag
    # --- 2. screened joint-kick pairs
    cand = set() if cfg.screen else set(itertools.combinations(range(N), 2))
    for i in range(N if cfg.screen else 0):
        # recall matters more than precision here (a missed partner is never kicked): take the
        # max_per_target strongest sources whether or not they pass the Bonferroni test
        order = [v for v in np.argsort(-strength[i]) if v != i]
        src = np.array(order[: cfg.max_per_target])
        cand |= set(itertools.combinations(sorted(src.tolist()), 2))
    if cfg.random_pairs is not None:  # random-selection baseline with the same joint-kick budget
        allp = list(itertools.combinations(range(N), 2))
        pick = rng.choice(len(allp), min(cfg.random_pairs, len(allp)), replace=False)
        cand = {allp[q] for q in pick}
    pairs = np.array(sorted(cand), dtype=np.int64).reshape(-1, 2)
    P = len(pairs)
    out = {
        "influence": infl,
        "pairs_kicked": pairs,
        "n_pairs_exhaustive": N * (N - 1) // 2,
        "sigma_read": sigma,
    }
    n_interventions = 1 + N + P
    n_readouts = (1 + N) * r_bs
    if P == 0:
        out.update(hyperedges=[], hyper_scores={}, n_interventions=n_interventions,
                   n_readouts=n_readouts, rounds=0)
        return out
    clean_j = k.integrate([tuple(p) for p in pairs])  # [3, S, P, N]
    reps = np.full(P, cfg.r0)
    jsum = k.readout_sum(clean_j, cfg.r0)
    n_readouts += P * cfg.r0
    j_, k_ = pairs[:, 0], pairs[:, 1]
    # Bonferroni over the tests actually performed: valid because the kicked pairs were chosen from
    # independent single-kick readouts, so selection does not depend on the joint-kick noise
    n_tests = P * (N - 2) if cfg.bonferroni == "performed" else N * (N - 1) * (N - 2) // 2
    level = cfg.alpha / max(n_tests, 1)
    budget = cfg.readout_budget
    rounds = 0

    def stats():
        joint = jsum / reps[None, :, None]
        D = joint - single[:, j_] - single[:, k_] + base[:, None, :]  # [S, P, tgt]
        if sigma > 0:
            var = v1 * (1.0 / reps[None, :, None] + 3.0 / r_bs)
            qq = (D**2 / var).sum(0)  # [P, tgt]
            p = chi2.sf(qq, S)
        else:  # noiseless: no noise model; "p" is 1 − relative magnitude (ranking only, as in
            # joint_kick_experiment.py), detections use the 1e-6 relative floor of kick_candidates
            mag = np.sqrt((D**2).mean(0))
            p = 1.0 - mag / max(mag.max(), 1e-300)
        p[np.arange(P), j_] = 1.0
        p[np.arange(P), k_] = 1.0
        return p

    p = stats()
    while cfg.adaptive and sigma > 0 and rounds < cfg.max_rounds:
        # p_hi is per kick: Bonferroni over its N − 2 readout targets, so null kicks rarely qualify
        undecided = (p.min(1) > level) & (p.min(1) < cfg.p_hi / max(N - 2, 1))
        if not undecided.any():
            break
        add = reps[undecided].copy()  # double their readouts
        if budget is not None and n_readouts + add.sum() > budget:
            break
        extra = np.zeros_like(jsum)
        for q_idx in np.where(undecided)[0]:
            extra[:, q_idx] = k.readout_sum(clean_j[:, :, q_idx], int(reps[q_idx]))
        jsum = jsum + extra
        n_readouts += int(add.sum())
        reps[undecided] *= 2
        rounds += 1
        p = stats()
    # base/single readouts are shared by all joint kicks; their error enters through 3/r_bs above
    hyper_p: dict[tuple[int, int, int], float] = {}
    for q_idx, (a, b) in enumerate(pairs):
        for i in range(N):
            if i in (a, b):
                continue
            key = tuple(sorted((i, int(a), int(b))))
            hyper_p[key] = min(hyper_p.get(key, 1.0), float(p[q_idx, i]))
    out.update(
        hyperedges=sorted(
            t for t, pv in hyper_p.items() if (pv < level if sigma > 0 else pv < 1.0 - 1e-6)
        ),
        hyper_scores={
            t: float(-np.log10(max(pv, 1e-300)) if sigma > 0 else 1.0 - pv) for t, pv in hyper_p.items()
        },
        n_interventions=n_interventions,
        n_readouts=n_readouts,
        rounds=rounds,
        repeats=reps,
    )
    return out
