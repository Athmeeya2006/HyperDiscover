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
