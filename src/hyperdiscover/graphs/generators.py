"""Pairwise random-graph generators (ER, BA, WS, SBM, RGG), all seeded explicitly."""

from __future__ import annotations

import networkx as nx
import numpy as np


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def make_graph(
    kind: str, n: int, rng: np.random.Generator, mean_degree: float = 4.0, **kw
) -> np.ndarray:
    """Undirected edges [E, 2] (a < b) for a random graph of the given kind.

    Args:
        kind: one of "er", "ba", "ws", "sbm", "rgg", "empty".
        n: number of nodes.
        rng: numpy Generator (the only source of randomness).
        mean_degree: target mean degree ⟨k⟩.
    """
    kind = kind.lower()
    s = _seed(rng)
    k = max(mean_degree, 0.0)
    if kind == "empty":
        G = nx.empty_graph(n)
    elif kind == "er":
        G = nx.gnp_random_graph(n, min(1.0, k / max(n - 1, 1)), seed=s)
    elif kind == "ba":
        m = max(1, int(round(k / 2)))
        G = nx.barabasi_albert_graph(n, min(m, n - 1), seed=s)
    elif kind == "ws":
        kk = max(2, int(round(k / 2)) * 2)
        G = nx.watts_strogatz_graph(n, min(kk, n - 1), kw.get("rewire_p", 0.1), seed=s)
    elif kind == "sbm":
        n_blocks = int(kw.get("n_blocks", 2))
        sizes = [n // n_blocks + (1 if b < n % n_blocks else 0) for b in range(n_blocks)]
        ratio = float(kw.get("in_out_ratio", 8.0))
        # mean degree k = p_in (s-1) + p_out (n - s) with p_in = ratio p_out
        s_avg = n / n_blocks
        p_out = k / (ratio * (s_avg - 1) + (n - s_avg))
        p_in = min(1.0, ratio * p_out)
        P = [[p_in if a == b else p_out for b in range(n_blocks)] for a in range(n_blocks)]
        G = nx.stochastic_block_model(sizes, P, seed=s)
    elif kind == "rgg":
        # expected degree ≈ n π r² in the unit square (ignoring boundary effects)
        r = float(kw.get("radius", np.sqrt(k / (np.pi * max(n - 1, 1)))))
        G = nx.random_geometric_graph(n, r, seed=s)
    else:
        raise ValueError(f"unknown graph kind {kind!r}")
    edges = np.array(sorted((min(a, b), max(a, b)) for a, b in G.edges() if a != b), dtype=np.int64)
    return edges.reshape(-1, 2)
