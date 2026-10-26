"""Alternating convex gating: an alternative to stochastic hard-concrete gates (DECISIONS.md D-029).

The model is linear in the edge weights once the message functions are fixed:

    ẋ_i − f(x_i) − c_i = Σ_j w_ij g(x_i, x_j) + Σ_{(j,k)} w_ijk h(x_i, x_j, x_k),

so we alternate
  (a) gradient steps on the message MLPs f, g, h (and c_i) with the weights held fixed, and
  (b) for every node, a NON-NEGATIVE LASSO for its incoming weights on a large batch, with the
      penalty chosen by BIC (LARS path, scikit-learn ``LassoLarsIC``).
Step (b) is convex and returns exact zeros, which the hard-concrete gates failed to reach on the
S4 sanity case (RESEARCH_LOG 2026-10-25). The Occam prior λ_t > λ_p is implemented as a column
rescaling: triadic columns are divided by ``tri_penalty`` (> 1), which multiplies their effective
L1 penalty by the same factor. Rounds start pairwise-only for ``pair_only_rounds`` (staged schedule).

Readout: weights normalised by the typical strong weight of their kind (median of weights ≥ 10% of the
max; true edges ≈ 1 because g and h are shared), NOT clipped (StructureModel.normalized_weights);
``> 0.5`` = present.

BIC needs a noise-variance estimate; scikit-learn's default (OLS residual) is 0 on clean data, or
undefined when a node has more candidates than samples, and LassoLarsIC then fails. We pass it
explicitly, floored at ``noise_floor`` × Var(target) (``_noise_variance``).

References (references/references.bib): meinshausen2007relaxed, belloni2013least, tibshirani1996regression, efron2004least, zou2007degrees, lawson1995solving, brunton2016discovering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.linear_model import LassoLarsIC

from ..data.dataset import TrainData
from ..model.structure_model import StructureModel
from .losses import derivative_loss, weak_loss


@dataclass
class AlternatingConfig:
    rounds: int = 8
    steps_per_round: int = 400
    first_round_steps: int = 1500
    batch: int = 128
    lr_net: float = 3e-3
    weight_decay: float = 1e-5
    lasso_samples: int = 3000
    tri_penalty: float = 2.0
    pair_only_rounds: int = 2
    criterion: str = "bic"
    log_every: int = 1
    # audit 2026-10-26 (docs/AUDIT): "relaxed" = standardised LASSO selection, then a non-negative
    # least-squares refit on the selected columns and an absolute threshold on scale-normalised weights;
    # "lasso" = the original rule (support = LASSO weight > 0); "fixed" = no selection (known structure).
    readout: str = "relaxed"
    threshold: float = 0.5
    # "sindy": round-0 support from a generic-library SINDy selection (a good basin for g); "ones": all on
    warm_start: str = "sindy"
    # keep the triple weights as given (e.g. hyperedges proven by joint kicks) and select only pairs
    fix_tris: bool = False
    sindy_threshold: float = 0.05
    # "derivative" (Savitzky–Golay targets) or "weak" (weak form over sliding windows: both the MLP loss and
    # the LASSO design are projected on the test function, no explicit differentiation; §18.15 W2)
    loss: str = "derivative"
    # functional-ANOVA pure-interaction penalty on h (identifiability option (b), §7 Stage 3; §18.15 W9):
    # weight of model.anova_penalty in the MLP steps of the triadic rounds (0 = off)
    anova: float = 0.0
    # §18.15 W8: after training, keep the SINDy warm-start support instead of the final one if it has the lower
    # held-out BIC (weights refit by NNLS through the learned messages) — never end worse than the warm start
    warm_guard: bool = False
    # Occam schedule (CLAUDE.md §7 Stage 3): after the pair-only rounds the pair support is frozen;
    # triples are selected on the residual of an NNLS fit of the frozen pairs, then pairs are refit
    pair_lock: bool = False
    # BIC noise-variance floor, relative to the variance of the node's residual target (§18.4 P0)
    noise_floor: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> AlternatingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@torch.no_grad()
def _design(model: StructureModel, S: torch.Tensor, Y: torch.Tensor):
    """Residual targets [n, N] and per-candidate message values: G [n, E], H [n, Ht] (d = 1)."""
    base = model.msg.f(S)
    if model.use_c:
        base = base + model.c
    R = (Y - base)[..., 0]
    i, j = model.pair_index[:, 0], model.pair_index[:, 1]
    G = model.msg.g(S[:, i], S[:, j])[..., 0] if len(i) else torch.zeros(len(S), 0)
    if len(model.tri_index):
        a, b, c = (model.tri_index[:, q] for q in range(3))
        H = model.msg.h(S[:, a], S[:, b], S[:, c])[..., 0]
    else:
        H = torch.zeros(len(S), 0)
    return R.cpu().numpy(), G.cpu().numpy(), H.cpu().numpy()


def _noise_variance(Xd: np.ndarray, y: np.ndarray, floor: float) -> float:
    """OLS residual variance of y on Xd [n, p] (dof n − p), floored at ``floor`` × Var(y).

    Falls back to the floor when n ≤ p (OLS interpolates). The floor keeps BIC finite on clean data.
    """
    n, p = Xd.shape
    vy = float(np.var(y))
    lo = max(floor * vy, 1e-30)
    if n <= p + 1:
        return lo
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    res = y - Xd @ coef
    return max(float(res @ res) / (n - p), lo)


def select_weights(
    R: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    pair_t: np.ndarray,
    tri_t: np.ndarray,
    N: int,
    cfg: AlternatingConfig,
    use_tris: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node structure selection from residual targets R [n, N] and candidate message columns
    G [n, E] (pairs, target ``pair_t``) and H [n, Ht] (triples, focal ``tri_t``). Returns (w_pair, w_tri)."""
    w_pair = np.zeros(G.shape[1])
    w_tri = np.zeros(H.shape[1])
    for i in range(N):
        pc = np.where(pair_t == i)[0]
        tc = np.where(tri_t == i)[0] if use_tris else np.zeros(0, int)
        Xd = np.concatenate([G[:, pc], H[:, tc] / cfg.tri_penalty], 1)
        if Xd.shape[1] == 0:
            continue
        scale = np.sqrt((Xd**2).mean(0))
        keep = scale > 1e-10  # columns that are identically zero carry no information
        if not keep.any():
            continue
        nv = _noise_variance(Xd[:, keep], R[:, i], cfg.noise_floor)
        lasso = LassoLarsIC(
            criterion=cfg.criterion, positive=True, fit_intercept=False, noise_variance=nv
        )
        if cfg.readout == "relaxed":
            # standardised columns: every candidate pays the same L1 price (× tri_penalty for triples)
            lasso.fit(Xd[:, keep] / scale[keep], R[:, i])
            sel = np.where(keep)[0][lasso.coef_ > 0]
            coef = np.zeros(Xd.shape[1])
            if len(sel):  # relaxed LASSO: unshrunk non-negative refit on the selected support
                coef[sel] = nnls(Xd[:, sel], R[:, i])[0]
        else:
            lasso.fit(Xd[:, keep], R[:, i])
            coef = np.zeros(Xd.shape[1])
            coef[keep] = lasso.coef_
        w_pair[pc] = coef[: len(pc)]
        w_tri[tc] = coef[len(pc) :] / cfg.tri_penalty
    if cfg.readout == "relaxed":
        # g and h are shared, so true weights share one scale: normalise by the typical strong weight
        # and drop everything below threshold × scale (absolute, not "any non-zero")
        w_pair = _threshold(w_pair, cfg.threshold)
        w_tri = _threshold(w_tri, cfg.threshold)
    return w_pair, w_tri


def select_on_batch(
    model: StructureModel,
    R: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    cfg: AlternatingConfig,
    use_tris: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Selection rule on one batch's design (R, G, H), honouring ``pair_lock`` and ``fix_tris``."""
    pair_t = model.pair_index[:, 0].cpu().numpy()
    tri_t = model.tri_index[:, 0].cpu().numpy() if len(model.tri_index) else np.zeros(0, int)
    lock = getattr(model, "pair_lock_mask", None)
    if cfg.pair_lock and use_tris and lock is not None:
        wp0 = _nnls_per_node(R, G, pair_t, lock, model.N)
        R1 = R.copy()
        np.add.at(R1.T, pair_t, -(G * wp0[None, :]).T)  # per-target residual of the frozen pairs
        _, w_tri = select_weights(R1, G[:, :0], H, pair_t[:0], tri_t, model.N, cfg, True)
        R2 = R.copy()
        np.add.at(R2.T, tri_t, -(H * w_tri[None, :]).T)
        w_pair = _nnls_per_node(R2, G, pair_t, lock, model.N)
    elif cfg.fix_tris and use_tris:
        # triadic part is given: subtract it per focal node, select pairs on the remainder
        w_tri = model.tri_w.cpu().numpy()
        R2 = R.copy()
        np.add.at(R2.T, tri_t, -(H * w_tri[None, :]).T)
        w_pair, _ = select_weights(R2, G, H[:, :0], pair_t, tri_t[:0], model.N, cfg, False)
    else:
        w_pair, w_tri = select_weights(R, G, H, pair_t, tri_t, model.N, cfg, use_tris)
    return w_pair, w_tri


@torch.no_grad()
def batch_design(model: StructureModel, data: TrainData, n: int, gen: torch.Generator, loss: str):
    """Residual targets R and message columns G, H on a random batch, for the derivative or weak form.

    Weak form: with test function φ on a window of L states, Σ_l φ_l ẋ(t_l) = −Σ_l φ'_l x(t_l) (the stored
    weak target), so R = target − Σ_l φ_l (f + c)(x_l) and every column is Σ_l φ_l g(x_i, x_j)(t_l): the model
    stays linear in the edge weights. n samples ↔ n // L windows (same number of evaluated states).
    """
    if loss != "weak":
        S, Y = data.derivative_batch(n, gen)
        return _design(model, S, Y)
    L = len(data.phi)
    Xw, Yw = data.weak_batch(max(n // L, 64), gen)  # [B, L, N, d], [B, N, d]
    B, _, N, d = Xw.shape
    R0, G0, H0 = _design(model, Xw.reshape(B * L, N, d), torch.zeros(B * L, N, d, device=Xw.device))
    phi = data.phi.cpu().numpy()

    def proj(A):
        return np.einsum("l,blc->bc", phi, A.reshape(B, L, -1))

    R = Yw[..., 0].cpu().numpy() + proj(R0)  # R0 = −(f + c) on every state
    return R, proj(G0), proj(H0)


def lasso_weights(
    model: StructureModel,
    data: TrainData,
    cfg: AlternatingConfig,
    use_tris: bool,
    gen: torch.Generator,
) -> dict:
    """Per-node structure selection (``select_weights``) on a large batch; updates the model buffers."""
    R, G, H = batch_design(model, data, cfg.lasso_samples, gen, cfg.loss)
    w_pair, w_tri = select_on_batch(model, R, G, H, cfg, use_tris)
    dev = model.pair_w.device
    model.pair_w = torch.as_tensor(w_pair, dtype=torch.float32, device=dev)
    model.tri_w = torch.as_tensor(w_tri, dtype=torch.float32, device=dev)
    return {"pairs_on": int((w_pair > 0).sum()), "tris_on": int((w_tri > 0).sum())}


def _nnls_per_node(R, G, pair_t, mask, N) -> np.ndarray:
    """Non-negative least-squares weights of the masked pair columns, per target node."""
    w = np.zeros(G.shape[1])
    for i in range(N):
        pc = np.where((pair_t == i) & mask)[0]
        if len(pc):
            w[pc] = nnls(G[:, pc], R[:, i])[0]
    return w


def _threshold(w: np.ndarray, thr: float) -> np.ndarray:
    """Zero weights below ``thr`` × the typical strong weight (median of weights ≥ 10% of the max)."""
    if not len(w) or w.max() <= 0:
        return w
    ref = np.median(w[w >= 0.1 * w.max()])
    return np.where(w >= thr * ref, w, 0.0)


def warm_start_support(model: StructureModel, data: TrainData, threshold: float) -> tuple:
    """Round-0 support {0, 1} per candidate from a generic-library SINDy selection (Phase-2 baseline).

    Nodes where SINDy selects no pair keep all their candidate pairs on (no evidence either way).
    """
    from ..baselines.sindy_full import sindy_full

    X = data.X_np[:, data.time_mask]
    dX = data.dX_np[:, data.time_mask]
    r = sindy_full(X, dX, "generic", None, bool(model.msg.embed.periodic), threshold)
    pi = model.pair_index.cpu().numpy()
    ti = model.tri_index.cpu().numpy()
    wp = np.array([float(r["pair_scores"].get((i, j), 0.0) > 0) for i, j in pi])
    wt = np.array([float(r["tri_scores"].get((i, j, k), 0.0) > 0) for i, j, k in ti])
    for i in range(model.N):
        rows = pi[:, 0] == i
        if rows.any() and wp[rows].sum() == 0:
            wp[rows] = 1.0
    return wp, wt


def _support_bic(R, G, H, pair_t, tri_t, keep_p, keep_t, N) -> tuple[float, np.ndarray, np.ndarray]:
    """Held-out BIC of a support (NNLS weights per node, summed over nodes) and the fitted weights."""
    n = len(R)
    bic, wp, wt = 0.0, np.zeros(G.shape[1]), np.zeros(H.shape[1])
    for i in range(N):
        pc = np.where((pair_t == i) & keep_p)[0]
        tc = np.where((tri_t == i) & keep_t)[0]
        A = np.concatenate([G[:, pc], H[:, tc]], 1)
        if A.shape[1]:
            coef, res = nnls(A, R[:, i])
            rss = float(res**2)
            wp[pc], wt[tc] = coef[: len(pc)], coef[len(pc) :]
            k = int((coef > 0).sum())
        else:
            rss, k = float(R[:, i] @ R[:, i]), 0
        bic += n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
    return bic, wp, wt


@torch.no_grad()
def _warm_guard(model, data, cfg, warm_pair, warm_tri, seed, logger) -> dict:
    """Choose between the final and the warm-start support by BIC on a fresh batch (W8)."""
    gen = torch.Generator().manual_seed(seed + 4242)
    R, G, H = batch_design(model, data, cfg.lasso_samples, gen, cfg.loss)
    pair_t = model.pair_index[:, 0].cpu().numpy()
    tri_t = model.tri_index[:, 0].cpu().numpy() if len(model.tri_index) else np.zeros(0, int)
    fin_p, fin_t = model.pair_w.cpu().numpy() > 0, model.tri_w.cpu().numpy() > 0
    b_fin, _, _ = _support_bic(R, G, H, pair_t, tri_t, fin_p, fin_t, model.N)
    b_warm, wp, wt = _support_bic(R, G, H, pair_t, tri_t, warm_pair > 0, warm_tri > 0, model.N)
    use_warm = b_warm < b_fin
    if use_warm:
        dev = model.pair_w.device
        model.pair_w = torch.as_tensor(wp, dtype=torch.float32, device=dev)
        model.tri_w = torch.as_tensor(wt, dtype=torch.float32, device=dev)
    logger(f"[alternating] warm guard: BIC final {b_fin:.1f} vs warm start {b_warm:.1f} -> "
           f"{'WARM START' if use_warm else 'final'}")
    return {"bic_final": b_fin, "bic_warm": b_warm, "used_warm": bool(use_warm)}


def train_alternating(
    model: StructureModel, data: TrainData, cfg_dict: dict, seed: int = 0, logger=print
) -> dict:
    """Run the alternating schedule. Returns a history compatible with the gate-evolution figures."""
    cfg = AlternatingConfig.from_dict(cfg_dict)
    gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 12345)
    torch.manual_seed(seed)
    model.weight_mode = True
    model.set_trainable(f=True, g=True, h=True, pair_gates=False, tri_gates=False, c=True)
    params = [p for p in model.msg.parameters()] + ([model.c] if model.c is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.lr_net, weight_decay=cfg.weight_decay)
    history = {"step": [], "loss": [], "data": [], "pair_probs": [], "tri_probs": []}
    t0 = time.time()
    step = 0
    warm_tri = None
    warm_pair = None
    if cfg.warm_start == "sindy":
        wp, warm_tri = warm_start_support(model, data, cfg.sindy_threshold)
        warm_pair = wp.copy()
        model.pair_w = torch.as_tensor(wp, dtype=torch.float32, device=model.pair_w.device)
        logger(
            f"[alternating] SINDy warm start: {int(wp.sum())}/{len(wp)} pairs, "
            f"{int(warm_tri.sum())}/{len(warm_tri)} triples"
        )
    for r in range(cfg.rounds):
        use_tris = r >= cfg.pair_only_rounds
        model.use_tris = use_tris
        if not use_tris:
            model.tri_w = torch.zeros_like(model.tri_w)
        elif r == cfg.pair_only_rounds:
            # first triadic round: h (zero-initialised) learns the residual before the LASSO selects;
            # start from the SINDy triples when warm-started (if any), else all candidates on
            if cfg.fix_tris:
                model.tri_w = torch.ones_like(model.tri_w)
            elif warm_tri is not None and warm_tri.sum() > 0:
                model.tri_w = torch.as_tensor(
                    warm_tri, dtype=torch.float32, device=model.tri_w.device
                )
            else:
                model.tri_w = torch.ones_like(model.tri_w)
        n_steps = cfg.first_round_steps if r == 0 else cfg.steps_per_round
        model.train()
        for _ in range(n_steps):
            if cfg.loss == "weak":
                ld, _ = weak_loss(model, data, max(cfg.batch // 4, 16), gen, gate_mode="deterministic")
            else:
                ld, _ = derivative_loss(model, data, cfg.batch, gen, gate_mode="deterministic")
            loss = ld
            if cfg.anova > 0 and use_tris:
                S_an, _ = data.derivative_batch(256, gen)
                loss = loss + cfg.anova * model.anova_penalty(S_an)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            step += 1
        if cfg.readout == "fixed":  # known structure: weights stay as given, only messages train
            info = {
                "pairs_on": int((model.pair_w > 0).sum()),
                "tris_on": int((model.tri_w > 0).sum()) if use_tris else 0,
            }
        else:
            info = lasso_weights(model, data, cfg, use_tris, gen)
            if cfg.pair_lock and r == cfg.pair_only_rounds - 1:
                model.pair_lock_mask = model.pair_w.cpu().numpy() > 0
                logger(f"[alternating] pair support locked: {int(model.pair_lock_mask.sum())} pairs")
        with torch.no_grad():
            evf = weak_loss if cfg.loss == "weak" else derivative_loss
            ev, _ = evf(model, data, 512 if cfg.loss != "weak" else 128, eval_gen, gate_mode="deterministic")
        eval_gen.manual_seed(seed + 12345)
        pp, pt = model.gate_probs()
        history["step"].append(step)
        history["loss"].append(float(ev))
        history["data"].append(float(ld.detach()))
        history["pair_probs"].append(pp.astype(np.float32))
        history["tri_probs"].append(pt.astype(np.float32))
        if r % cfg.log_every == 0 or r == cfg.rounds - 1:
            logger(
                f"[alternating] round {r} ({'pairs+tris' if use_tris else 'pairs only'}) steps {step} "
                f"data {float(ev):.3e} pairs_on {info['pairs_on']}/{len(pp)} tris_on {info['tris_on']}/{len(pt)} "
                f"{time.time() - t0:.1f}s"
            )
    model.use_tris = True
    if cfg.warm_guard and warm_pair is not None and cfg.readout != "fixed" and not cfg.fix_tris:
        history["warm_guard"] = _warm_guard(model, data, cfg, warm_pair, warm_tri, seed, logger)
    model.eval()
    history["stage"] = "alternating"
    history["seconds"] = time.time() - t0
    return history
