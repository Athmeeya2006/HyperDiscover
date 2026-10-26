"""Loaders for real higher-order structures (Phase 7; ask the human before use).

Real hypergraphs are fetched with XGI's data loader, which needs internet; on Ada, download
on the login node and cache to disk first. Dataset names must be verified against the XGI
version in use (CLAUDE.md §10).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from ..systems.base import Structure


def structure_from_hyperedge_list(
    n_nodes: int, hyperedges: list[list[int]], pair_from_large: bool = True
) -> Structure:
    """Convert arbitrary-size hyperedges into our order-≤3 Structure.

    Size-2 hyperedges become pairwise edges; size-3 become 3-hyperedges; larger ones are
    either projected to their pairwise clique (``pair_from_large``) or dropped.
    """
    edges, tris = set(), set()
    for he in hyperedges:
        he = sorted(set(int(v) for v in he))
        if len(he) == 2:
            edges.add(tuple(he))
        elif len(he) == 3:
            tris.add(tuple(he))
        elif len(he) > 3 and pair_from_large:
            edges.update(combinations(he, 2))
    return Structure(
        n_nodes,
        np.array(sorted(edges), dtype=np.int64).reshape(-1, 2),
        np.array(sorted(tris), dtype=np.int64).reshape(-1, 3),
        {"source": "hyperedge_list"},
    )


def load_xgi(
    name: str, cache_dir: str | Path | None = None, max_nodes: int | None = None
) -> Structure:
    """Load a hypergraph from XGI's data repository (requires ``xgi``; verify dataset names)."""
    import xgi  # optional dependency

    H = (
        xgi.load_xgi_data(name, cache=True)
        if cache_dir is None
        else xgi.load_xgi_data(name, path=str(cache_dir))
    )
    H.cleanup(isolates=False, singletons=False, multiedges=False, relabel=True)
    nodes = sorted(H.nodes)
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    idx = {v: i for i, v in enumerate(nodes)}
    # induced sub-hypergraph: keep a hyperedge only if ALL its members survive the truncation
    # (dropping members would invent smaller, spurious hyperedges/edges)
    hes = [[idx[v] for v in e] for e in H.edges.members() if all(v in idx for v in e)]
    hes = [e for e in hes if len(e) >= 2]
    s = structure_from_hyperedge_list(len(nodes), hes)
    s.meta.update({"source": f"xgi:{name}"})
    return s


def connected_subhypergraph(
    structure: Structure, n: int, rng: np.random.Generator, max_hyper: int | None = None
) -> Structure:
    """Induced sub-structure on n nodes grown by BFS from a random high-degree node (keeps it connected).

    Only edges / hyperedges with ALL members inside are kept (no invented smaller interactions);
    ``max_hyper`` caps the number of hyperedges (random subset) so dense contact data stay simulable.
    """
    N = structure.n_nodes
    nbr: list[set] = [set() for _ in range(N)]
    for a, b in np.asarray(structure.edges).tolist():
        nbr[a].add(b)
        nbr[b].add(a)
    for a, b, c in np.asarray(structure.hyperedges).tolist():
        nbr[a] |= {b, c}
        nbr[b] |= {a, c}
        nbr[c] |= {a, b}
    deg = np.array([len(s) for s in nbr])
    top = np.argsort(-deg)[: max(1, N // 10)]
    start = int(rng.choice(top))
    order, seen, queue = [], {start}, [start]
    while queue and len(order) < n:
        v = queue.pop(0)
        order.append(v)
        nxt = sorted(nbr[v] - seen)
        rng.shuffle(nxt)
        for u in nxt:
            seen.add(u)
            queue.append(u)
    idx = {v: i for i, v in enumerate(order)}
    E = [[idx[a], idx[b]] for a, b in np.asarray(structure.edges).tolist() if a in idx and b in idx]
    H = [[idx[a], idx[b], idx[c]] for a, b, c in np.asarray(structure.hyperedges).tolist()
         if a in idx and b in idx and c in idx]
    if max_hyper is not None and len(H) > max_hyper:
        H = [H[q] for q in sorted(rng.choice(len(H), max_hyper, replace=False))]
    return Structure(len(order), np.array(E, dtype=np.int64).reshape(-1, 2),
                     np.array(H, dtype=np.int64).reshape(-1, 3),
                     {**structure.meta, "subgraph_nodes": len(order)})
