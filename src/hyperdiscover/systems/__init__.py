"""Ground-truth higher-order benchmark systems S1–S6."""

from __future__ import annotations

from .base import XI, XJ, XK, HigherOrderSystem, Structure, directed_pairs, focal_triples
from .chaotic import RosslerHO
from .consensus_ho import ConsensusHO
from .kuramoto_ho import KuramotoHO
from .lotka_volterra_ho import LotkaVolterraHO
from .mutualistic_ho import MutualisticHO
from .sis_ho import SISHO

SYSTEMS: dict[str, type[HigherOrderSystem]] = {
    "S1": KuramotoHO,
    "S2": LotkaVolterraHO,
    "S3": ConsensusHO,
    "S4": SISHO,
    "S5": MutualisticHO,
    "S6": RosslerHO,
}


def make_system(system_id: str, structure: Structure, params=None, **kw) -> HigherOrderSystem:
    """Instantiate a benchmark system by ID (S1..S6)."""
    try:
        cls = SYSTEMS[system_id]
    except KeyError as e:
        raise KeyError(f"unknown system {system_id!r}; choose from {sorted(SYSTEMS)}") from e
    return cls(structure, params, **kw)


__all__ = [
    "XI",
    "XJ",
    "XK",
    "HigherOrderSystem",
    "Structure",
    "directed_pairs",
    "focal_triples",
    "SYSTEMS",
    "make_system",
    "KuramotoHO",
    "LotkaVolterraHO",
    "ConsensusHO",
    "SISHO",
    "MutualisticHO",
    "RosslerHO",
]
