#!/usr/bin/env python
"""Measure peak memory and step time of the structure model vs N (decide --per-gpu, batch, K_t).

    python scripts/profile_memory.py --device cuda --N 20 100 500 1000
    python scripts/profile_memory.py --device cpu --N 10 20 50

Candidates mimic the defaults: all pairs for N ≤ 200 (else K_p per node), all triples for
N ≤ 20 (else K_t per node). Reports per-step forward+backward time and peak memory (CUDA:
max_memory_allocated; CPU: peak RSS increase, approximate). Results go to RESEARCH_LOG.md by hand
only after they ran on the real hardware.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from itertools import combinations

import numpy as np
import torch


def candidates(N, k_p, k_t, rng):
    if N <= 200:
        pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    else:
        pairs = np.array(
            [
                (i, j)
                for i in range(N)
                for j in rng.choice(np.delete(np.arange(N), i), k_p, replace=False)
            ]
        )
    tris = []
    for i in range(N):
        others = [v for v in range(N) if v != i]
        if N <= 20:
            tris += [(i, j, k) for j, k in combinations(others, 2)]
        else:
            nb = rng.choice(others, min(15, N - 1), replace=False)
            pool = list(combinations(sorted(nb), 2))
            for q in rng.choice(len(pool), min(k_t, len(pool)), replace=False):
                tris.append((i, *pool[q]))
    return pairs, np.array(tris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--N", type=int, nargs="+", default=[20, 100, 500])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--k-p", type=int, default=30)
    ap.add_argument("--k-t", type=int, default=50)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()

    from hyperdiscover.model.structure_model import StructureModel

    rng = np.random.default_rng(0)
    rows = []
    for N in a.N:
        pairs, tris = candidates(N, a.k_p, a.k_t, rng)
        m = StructureModel(N, 1, pairs, tris, a.hidden, 3).to(a.device)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        x = torch.randn(a.batch, N, 1, device=a.device)
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        times = []
        for _ in range(a.steps):
            t0 = time.time()
            loss = (m(x) ** 2).mean() + m.l0_pair() + m.l0_tri()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if a.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        peak = (
            torch.cuda.max_memory_allocated() / 2**20
            if a.device == "cuda"
            else (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
        )
        row = {
            "N": N,
            "pairs": len(pairs),
            "triples": len(tris),
            "batch": a.batch,
            "device": a.device,
            "step_s_median": float(np.median(times[1:] or times)),
            "peak_MiB": float(peak),
        }
        rows.append(row)
        print(json.dumps(row))
        del m, opt, x
        if a.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
