#!/usr/bin/env python
"""Interventional structure + observational equations (DECISIONS.md D-030, CLAUDE.md §17.1).

    python scripts/kick_then_fit.py --experiment configs/experiment/sanity_s4.yaml --out runs/kick_fit/s4

1. Joint-kick interventions from sparse-IC (low-activity) operating states → truth-free candidates:
   significant direct-influence pairs and significant joint (mixed-difference) triples.
2. HyperDiscover fit on observational trajectories with those candidates replacing Stage-1 screening
   ``--mode hybrid`` (default): kicked hyperedges fixed, pairwise edges selected from data among the
   influence pairs; ``candidates``: selection over both; ``known``: everything kicked taken as given.
3. Symbolic distillation + metrics, written like any run directory (metrics.json, equations.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment", required=True, help="YAML with a `benchmark:` spec (+ `config:`)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["hybrid", "candidates", "known"], default="hybrid")
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--tau-dt", type=float, default=4.0)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--readout-noise", type=float, default=0.05, help="× mean node signal std")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--engine", default="library")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from _common import to_jsonable, write_provenance
    from hyperdiscover import HyperDiscover, benchmarks
    from hyperdiscover.data.simulate import sparse_ic
    from hyperdiscover.eval.run_metrics import run_metrics
    from hyperdiscover.intervene.joint_kick import kick_candidates, kick_responses

    exp = yaml.safe_load(Path(a.experiment).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = benchmarks.load(**exp["benchmark"])
    write_provenance(out, {"experiment": exp, "args": vars(a)}, a.seed)
    sysm, st = b.system, b.structure
    width = sysm.ic_box[1] - sysm.ic_box[0]
    sig = float(b.data.X_clean[..., 0].std(axis=(0, 1)).mean())
    rng = np.random.default_rng([a.seed, 99])

    # 1) interventions
    t0 = time.time()
    xs = sparse_ic(sysm, a.states, rng)
    resp = kick_responses(
        sysm,
        xs,
        a.delta,
        a.tau_dt * sysm.default_dt,
        a.readout_noise * sig / width,
        rng,
        n_repeats=a.repeats,
    )
    cand, info = kick_candidates(resp, a.alpha)
    true_h = {tuple(sorted(h)) for h in np.asarray(st.hyperedges).tolist()}
    got_h = {tuple(h) for h in np.asarray(cand.hyperedges).tolist()}
    true_e = {tuple(e) for e in np.asarray(st.edges).tolist()}
    got_e = {tuple(e) for e in np.asarray(cand.edges).tolist()}
    kick = {
        **info,
        "seconds": time.time() - t0,
        "interventions": a.states * (1 + st.n_nodes + st.n_nodes * (st.n_nodes - 1) // 2),
        "readouts_per_intervention": a.repeats,
        "hyper_tp": len(got_h & true_h),
        "hyper_fp": len(got_h - true_h),
        "hyper_fn": len(true_h - got_h),
        "edge_candidate_recall": len(got_e & true_e) / max(1, len(true_e)),
        "edge_candidates_false": len(got_e - true_e),
    }
    print("[kicks]", kick, flush=True)

    # 2) observational fit on the kicked candidates / structure
    cfg = exp.get("config", {})
    cfg = {**cfg, "gating": cfg.get("gating", "alternating")}
    hd = HyperDiscover(config=cfg, distill_engine=a.engine, seed=a.seed)
    seg = b.data.meta.get("segment_starts")
    M = b.data.X.shape[0]
    n_test = max(1, int(round(0.15 * M)))
    X_fit, X_test = b.data.X[: M - n_test], b.data.X[M - n_test :]
    kw = {
        "known": {"known_structure": cand},
        "candidates": {"candidates": cand},
        "hybrid": {"candidates": cand, "fixed_hyperedges": True},
    }[a.mode]
    hd.fit(X_fit, b.data.t, segment_starts=seg, out_dir=out, **kw)

    # 3) metrics
    metrics = run_metrics(hd, sysm, X_test, b.data.t)
    metrics["spec"] = b.spec
    metrics["kicks"] = kick
    metrics["mode"] = a.mode
    metrics["engine"] = a.engine
    (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2, default=float))
    eq = {k: str(v) for k, v in hd.equations_.items()}
    (out / "equations.json").write_text(json.dumps(eq, indent=2))
    e, h = metrics["edges"], metrics.get("hyperedges", {}).get("all", {})
    print(
        f"[result] kicks: hyper {kick['hyper_tp']}/{kick['hyper_fp']}/{kick['hyper_fn']} (TP/FP/FN), "
        f"edge-candidate recall {kick['edge_candidate_recall']:.2f} | fit: edges F1 {e.get('f1')} "
        f"hyper F1 {h.get('f1')} | equations exact {metrics['equations']['all_exact']} {eq}",
        flush=True,
    )


if __name__ == "__main__":
    main()
