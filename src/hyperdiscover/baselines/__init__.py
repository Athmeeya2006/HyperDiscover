"""Baselines (CLAUDE.md §8).

The THIS port (``this_reimpl.py``) is GPL-3.0 (derived from the official THIS code) and is NOT imported
here; load it explicitly: ``from hyperdiscover.baselines.this_reimpl import this_reconstruct``.
"""

from .arni import arni
from .granger_te import (
    correlation_scores,
    granger_scores,
    partial_correlation_scores,
    transfer_entropy_scores,
)
from .hypergnn import latent_incidence_baseline
from .nri import nri
from .sindy_full import directed_to_undirected, oracle_coefficient_report, sindy_full

__all__ = [
    "arni",
    "latent_incidence_baseline",
    "nri",
    "correlation_scores",
    "granger_scores",
    "partial_correlation_scores",
    "transfer_entropy_scores",
    "directed_to_undirected",
    "oracle_coefficient_report",
    "sindy_full",
]
