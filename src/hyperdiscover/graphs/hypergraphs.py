"""3-node hyperedge generators with controlled closure (open / closed / mixed).

Closure class of a triple = number of pairwise edges among its three nodes:
0 = open, 3 = closed, 1–2 = partially closed. The "mixed" mode draws a mixture of all
classes (see DECISIONS.md, D-007).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..systems.base import Structure
from .generators import make_graph


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in edges}


def _n_internal(t: tuple[int, int, int], es: set[tuple[int, int]]) -> int:
    a, b, c = sorted(t)
    return ((a, b) in es) + ((a, c) in es) + ((b, c) in es)


def _triangles(edges: np.ndarray, n: int) -> list[tuple[int, int, int]]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        nbrs[a].add(int(b))
        nbrs[b].add(int(a))
    tris = []
    for a, b in edges:
        a, b = int(a), int(b)
        for c in nbrs[a] & nbrs[b]:
            if c > b:
                tris.append((a, b, c))
    return tris


def sample_hyperedges(
    edges: np.ndarray,
    n: int,
    n_hyper: int,
    kind: str,
    rng: np.random.Generator,
    max_tries: int = 200_000,
    disjoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_hyper`` hyperedges of the given closure kind, adding pairwise edges if needed.

    Args:
        edges: undirected pairwise edges [E, 2].
        kind: "open" (no internal edges), "closed" (all three internal edges), "mixed".
        disjoint: if True, hyperedges share no nodes (useful for tiny sanity systems).

    Returns:
        (edges, hyperedges): ``edges`` may be augmented (closed kind adds missing edges
        to close sampled triples; open kind never adds edges) and hyperedges [H, 3].
    """
    es = _edge_set(edges)
    chosen: list[tuple[int, int, int]] = []
    used: set[int] = set()
    kinds = {"open": [0], "closed": [3], "mixed": [0, 1, 2, 3]}[kind]
    tries = 0
    tri_pool = _triangles(edges, n) if kind == "closed" else []
    if tri_pool:
        rng.shuffle(tri_pool)
    while len(chosen) < n_hyper and tries < max_tries:
        tries += 1
        target = kinds[len(chosen) % len(kinds)]
        if target == 3 and tri_pool:
            t = tri_pool.pop()
        else:
            t = tuple(sorted(int(v) for v in rng.choice(n, 3, replace=False)))
        if t in chosen or (disjoint and used & set(t)):
            continue
        m = _n_internal(t, es)
        if target == 0:
            if m != 0:
                continue
        new_edges: list[tuple[int, int]] = []
        if target == 3:  # close the triple by adding the missing edges
            new_edges = [p for p in combinations(t, 2) if p not in es]
        elif target != 0:  # partial: want exactly `target` internal edges
            if m > target:
                continue
            missing = [p for p in combinations(t, 2) if p not in es]
            rng.shuffle(missing)
            new_edges = [(int(p[0]), int(p[1])) for p in missing[: target - m]]
        # an added edge inside an earlier hyperedge would change that hyperedge's closure class
        if any(a in c and b in c for a, b in new_edges for c in map(set, chosen)):
            continue
        es.update(new_edges)
        chosen.append(t)
        used |= set(t)
    if len(chosen) < n_hyper:
        raise RuntimeError(
            f"could only place {len(chosen)}/{n_hyper} {kind} hyperedges; lower the density"
        )
    out_edges = np.array(sorted(es), dtype=np.int64).reshape(-1, 2)
    return out_edges, np.array(chosen, dtype=np.int64).reshape(-1, 3)


def make_structure(
    n: int,
    rng: np.random.Generator,
    graph: str = "er",
    mean_degree: float = 4.0,
    hyper_kind: str = "open",
    n_hyper: int | None = None,
    hyper_density: float = 0.5,
    max_hyper: int = 2000,
    pairwise_only: bool = False,
    disjoint: bool = False,
    **graph_kw,
) -> Structure:
    """Build a random higher-order structure.

    ``n_hyper`` defaults to ``round(hyper_density * n)`` capped at ``max_hyper``.
    """
    edges = make_graph(graph, n, rng, mean_degree, **graph_kw)
    if pairwise_only:
        hyp = np.zeros((0, 3), dtype=np.int64)
    else:
        nh = n_hyper if n_hyper is not None else min(max_hyper, int(round(hyper_density * n)))
        edges, hyp = sample_hyperedges(edges, n, nh, hyper_kind, rng, disjoint=disjoint)
    meta = {
        "graph": graph,
        "mean_degree": mean_degree,
        "hyper_kind": hyper_kind,
        "pairwise_only": pairwise_only,
    }
    return Structure(n, edges, hyp, meta)


def triple_class_counts(structure: Structure) -> np.ndarray:
    """Number of all C(N,3) triples in each closure class 0..3 (closed form, O(E + N))."""
    n = structure.n_nodes
    edges = structure.edges
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.int64)
    t3 = len(_triangles(edges, n))
    paths = int(np.sum(deg * (deg - 1) // 2))  # pairs of edges sharing a node
    t2 = paths - 3 * t3
    t1 = len(edges) * (n - 2) - 2 * t2 - 3 * t3
    total = n * (n - 1) * (n - 2) // 6
    t0 = total - t1 - t2 - t3
    return np.array([t0, t1, t2, t3], dtype=np.int64)
